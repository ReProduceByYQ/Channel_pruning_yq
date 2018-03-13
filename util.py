import numpy as np
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
import sys, json
sys.path.insert(0, 'D:/Deep_Learning/Win_Caffe/caffe-blvc/python')
import caffe
from google.protobuf import text_format
from sklearn.linear_model import *


"""
  Some function For Low Rank decompose
"""
# read low rank configure file
# input is configure_file_name
def load_config(config_file):
    with open(config_file, 'r') as fp:
        conf = json.load(fp)
    return conf

# vh_decompose:
#    input   layer_name,  rank=K
def vh_decompose(conv, K):
    def _create_new(name):
        new_ = LayerParameter()
        new_.CopyFrom(conv)
        new_.name = name
        new_.convolution_param.ClearField('kernel_size')
        new_.convolution_param.ClearField('pad')
        new_.convolution_param.ClearField('stride')
        return new_
    conv_param = conv.convolution_param
    # vertical
    v = _create_new(conv.name + '_v')
    del(v.top[:])
    v.top.extend([v.name])
    v.param[1].lr_mult = 0
    v_param = v.convolution_param
    v_param.num_output = K
    v_param.kernel_h, v_param.kernel_w = conv_param.kernel_size[0], 1
    v_param.pad_h, v_param.pad_w = conv_param.pad[0], 0
    if conv_param.stride==[]:
        v_param.stride_h, v_param.stride_w = 1, 1
    else:
        v_param.stride_h, v_param.stride_w = conv_param.stride[0], 1
    # horizontal
    h = _create_new(conv.name + '_h')
    del(h.bottom[:])
    h.bottom.extend(v.top)
    h_param = h.convolution_param
    h_param.kernel_h, h_param.kernel_w = 1, conv_param.kernel_size[0]
    h_param.pad_h, h_param.pad_w = 0, conv_param.pad[0]
    if conv_param.stride==[]:
        h_param.stride_h, h_param.stride_w = 1,1
    else:
        h_param.stride_h, h_param.stride_w = 1, conv_param.stride[0]
    return v, h

# input_file  : original prototxt
# config      : config.json return
# output_file : low rank generator prototxt
def make_lowrank_model(input_file, conf, output_file):
    with open(input_file, 'r') as fp:
        net = NetParameter()
        text_format.Parse(fp.read(), net)
    new_layers = []
    for layer in net.layer:
        if not layer.name in conf.keys():
            new_layers.append(layer)
            continue
        v, h = vh_decompose(layer, conf[layer.name])
        new_layers.extend([v, h])
    new_net = NetParameter()
    new_net.CopyFrom(net)
    del(new_net.layer[:])
    new_net.layer.extend(new_layers)
    with open(output_file, 'w') as fp:
        fp.write(text_format.MessageToString(new_net))

#
#
#
def approx_lowrank_weights(orig_model, orig_weights, conf,
                           lowrank_model, lowrank_weights):
    orig_net = caffe.Net(orig_model, orig_weights, caffe.TEST)
    lowrank_net = caffe.Net(lowrank_model, orig_weights, caffe.TEST)
    for layer_name in conf:
        W, b = [p.data for p in orig_net.params[layer_name]]
        v_weights, v_bias = \
            [p.data for p in lowrank_net.params[layer_name + '_v']]
        h_weights, h_bias = \
            [p.data for p in lowrank_net.params[layer_name + '_h']]
        # Set biases
        v_bias[...] = 0
        h_bias[...] = b.copy()
        # Get the shapes
        num_groups = v_weights.shape[0] // h_weights.shape[1]
        N, C, D, D = W.shape
        N = N // num_groups
        K = h_weights.shape[1]
        # SVD approximation
        for g in range(num_groups):
            W_ = W[N*g:N*(g+1)].transpose(1, 2, 3, 0).reshape((C*D, D*N))
            U, S, V = np.linalg.svd(W_)
            v = U[:, :K] * np.sqrt(S[:K])
            v = v[:, :K].reshape((C, D, 1, K)).transpose(3, 0, 1, 2)
            v_weights[K*g:K*(g+1)] = v.copy()
            h = V[:K, :] * np.sqrt(S)[:K, np.newaxis]
            h = h.reshape((K, 1, D, N)).transpose(3, 0, 1, 2)
            h_weights[N*g:N*(g+1)] = h.copy()
    lowrank_net.save(lowrank_weights)



"""
    Some function For Channel decompose
"""

def extract_layer_name(net, layer_name):
    
    convs_name = [ blob_name for blob_name in net._layer_names if 'conv' in blob_name or 'fc' in blob_name]
    pr_idx = convs_name.index(layer_name)
    top_layer = convs_name[pr_idx+1]
    bottom_layer = net.bottom_names[top_layer][0]


    return bottom_layer, top_layer

def rel_error(A, B):
    return np.mean((A-B)**2)**0.5/ np.mean(A**2)**.5
def relu(x):
    return np.maximum(x, 0.0)

def get_layer_params(input_file, layer_name):
    with open(input_file, 'r') as fp:
        net_file = NetParameter()
        text_format.Parse(fp.read(), net_file)

    pad_w =0
    pad_h =0
    k_w = 0
    k_h = 0
    for layer in net_file.layer:
        if layer.name == layer_name:
            pad_w = layer.convolution_param.pad_w
            pad_h = layer.convolution_param.pad_h
            k_w   = layer.convolution_param.kernel_w
            k_h   = layer.convolution_param.kernel_h

    return pad_h, pad_w, k_h, k_w

def get_layer_weights(net, name):
    return net.params[name][0].data, net.params[name][1].data

def extract_feat(prune_name, cfg, net):

    nBatch = cfg['nBatch']
    nPointsPerLayer = cfg['nPointsPerLayer']
    N = nBatch * nPointsPerLayer
    pad_h = cfg['params'][0]
    pad_w = cfg['params'][1]
    k_h   = cfg['params'][2]
    k_w   = cfg['params'][3]

    bottom_layer_name, top_layer_name = extract_layer_name(net, prune_name)  # pool1, conv2_1_v
    net.forward()

    w1, b1 = get_layer_weights(net, prune_name)          # conv1_2_h (64, 24, 1, 3) (64,)
    w2, b2 = get_layer_weights(net, top_layer_name)      # conv2_1_v (48 , 64, 3, 1) (48,)
    rank = int(w1.shape[0] / cfg['c_ratio'])             # 64 /1.15 = 55

    if 1:
        print("w1 shape :", w1.shape)
        print("b1 shape :", b1.shape)
        print("w2 shape :", w2.shape)
        print("b2 shape :", b2.shape)

    bottom_feat = net.blobs[bottom_layer_name].data   # pool1 (1, 64, 112, 112)
    top_feat  = net.blobs[top_layer_name].data     # conv2_1_v (1,48,112,112)

    pad_btm_feat = np. zeros((bottom_feat.shape[0], bottom_feat.shape[1], \
                            bottom_feat.shape[2]+2*pad_h,\
                            bottom_feat.shape[3]+2*pad_w))
                            # pool1 (1, 64 ,114 , 112)
    if 1: print("padding bottom shape ", pad_btm_feat.shape)
    pad_btm_feat[:,:,1:1+bottom_feat.shape[2],:]=bottom_feat

    c_in = pad_btm_feat.shape[1]  # 64  -> 55
    c_out = top_feat.shape[1]     # 48

    top_feat_dict = np.ndarray(shape=(N, c_out))
    btm_feat_dict = np.ndarray(shape=(N, c_in, k_h, k_w))

    nPicsPerBatch = k_h * k_w  # 3
    for batch in range(nBatch):
        randx = np.random.randint(0, top_feat.shape[2], nPointsPerLayer)
        randy = np.random.randint(0, top_feat.shape[3], nPointsPerLayer)

        point_feat = top_feat[:,:, randx,  randy].reshape(-1, nPointsPerLayer)  # (1,48,1,1)->(48, 10)
        top_feat_dict[batch*nPointsPerLayer:(batch+1)*nPointsPerLayer]=point_feat.T   #(10, 48)

        for p in range(nPointsPerLayer):
            btm_tmp_feat = pad_btm_feat[:,:, randx[p]:randx[p]+k_h , randy[p]:randy[p]+k_w]
            btm_feat_dict[batch*nPointsPerLayer+p] = btm_tmp_feat   # (1, 64 , 3, 1 )

    if 1:
        print('top feat dict shape is ', top_feat_dict.shape)
        print('bottom feat dict shape is ', btm_feat_dict.shape)

    samples = np.random.randint(0, N, 250)

    X = btm_feat_dict.copy()
    #reX = np.rollaxis(btm_feat_dict.reshape((N, c_in, -1))[samples], 1, 0)    # (64, 250, 3*1)
    #reW2 = np.transpose(w2.reshape((c_out, c_in, -1)),[1, 2, 0])

    Y = top_feat_dict -b2.reshape((1,-1))      #(5000, 48)
    #reY = Y[samples].reshape(-1)               #(250*48, )      
    #Z = np.matmul(reX, reW2).reshape((c_in, -1)).T  

    if 1:
        res = rel_error(X.reshape(X.shape[0],-1).dot(w2.reshape((w2.shape[0],-1)).T)  , Y)
        print('rMSE : ', res)
        res_relu = rel_error(relu( X.reshape(X.shape[0],-1).dot(w2.reshape((w2.shape[0],-1)).T) ), Y)
        print('relu- rMSE : ', res_relu)

    return X, w2, b2, Y , c_in, c_out

def fc_kernel(X, Y, copy_X=True, W=None, B=None, ret_reg=False,fit_intercept=True):
    """
    return: n c
    """
    assert copy_X == True
    assert len(X.shape) == 2
    _reg = LinearRegression(n_jobs=-1 , copy_X=copy_X, fit_intercept=fit_intercept)
    _reg.fit(X, Y)
    print('Linear regression score is ',_reg.score(X,Y),end='\t')
    return _reg.coef_, _reg.intercept_

def make_channel_pruning_model(input_file, output_file, bottom_layer, rank):
    with open(input_file, 'r') as fp:
        nett = NetParameter()
        text_format.Parse(fp.read(), nett)

    """ do not anything """
    def _creat_new(name, layer_name):
        new_ = LayerParameter()
        new_.CopyFrom(layer_name)
        new_.name = name
        new_.convolution_param.ClearField('num_output')
        return new_
    
    new_layer=[]
    for layer in nett.layer:
        if layer.name != bottom_layer:
            new_layer.append(layer)
        else:
            newConv = _creat_new(bottom_layer, layer)
            conv_param = newConv.convolution_param
            conv_param.num_output = rank
            new_layer.append(newConv)

    new_net = NetParameter()
    new_net.CopyFrom(nett)
    del(new_net.layer[:])

    new_net.layer.extend(new_layer)
    with open(output_file,'w') as fp:
        fp.write(text_format.MessageToString(new_net))

def make_channel_pruning_weights(prototxt, prune_name, top_layer_name):
    channel_net = caffe.Net(prototxt, caffe.TEST)

    for layer_name, param in channel_net.params.items():
        if layer_name == prune_convs_name[0]:
            orig_w , orig_b = [p.data for p in net.params[layer_name]]
            param[0].data[...] = orig_w[idxs,...].copy()
            param[1].data[...] = orig_b[idxs,...].copy()

        elif layer_name == prune_convs_name[1]:
            param[0].data[...] = newWeight2.copy()
            param[1].data[...] = _b2.reshape(param[1].data.shape).copy()
        else:
            orig_w , orig_b = [p.data for p in net.params[layer_name]]
            new_w , new_b = param[0].data, param[1].data
            new_w[...]=orig_w.copy()
            new_b[...]=orig_b.copy()
           
    channel_net.save(model_weights_out)


