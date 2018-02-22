"""
This is Channel Pruning reproduct，but only for pruning one layer
Author： YQ
data：2018-1-24

"""
# 1. set import libs
import numpy as np
from sklearn.linear_model import *
import sys
sys.path.insert(0, 'D:/Deep_Learning/Win_Caffe/caffe-blvc/python')
import caffe
from google.protobuf import text_format
from argparse import ArgumentParser
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter

device=0   # 0 is cpu  ; 1 is gpu
if device:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

# 2. set original_prototxt, original_weights, pruned_prototxt, pruned_weights
#model_def = 'D:/INstall_files/channel_pruning/train_val.prototxt'
model_def_out = 'D:/INstall_files/channel_pruning/train_val_out.prototxt'

#model_weights = 'D:/INstall_files/channel_pruning/vgg16__iter_10000.caffemodel'
model_weights_out = 'D:/INstall_files/channel_pruning/vgg16__iter_out.caffemodel'

model_def = model_def_out
model_weights =model_weights_out

def rel_error(A,B):
    return np.mean((A-B)**2)**0.5/ np.mean(A**2)**.5
def relu(x):
    return np.maximum(x, 0.0)

"""
we will set some configure infomation to channel pruning:

    1. prune_layer_name - this is we will remove channel from *prune_layer_name.weight*
    2. nPointsPerLayer  - we do not use all feature to Lasso regression, instead of sample
                           number of nPointsPerLayer on feature map
    3. nBatch           - Total of sample is (nBatch * nPointsPerLayer)
    4. rank             - how many channels you will preserve = num_ouput / c_ratio
    5. alpha            - for Lasso Regression params  default is 0.01
"""
prune_layer_name = 'conv2_1'

nPointsPerLayer = 10
nBatch = 500
c_ratio = 1.15
alpha = 1e-3

Debug = 1

def get_layer_params(input_file, layer_name):
    
    with open(input_file,'r') as fp:
        net_file = NetParameter()
        text_format.Parse(fp.read(), net_file)

    pad = 0
    k_w = 0
    k_h = 0

    for layer in net_file.layer:
        if layer.name ==layer_name:
            conv_param = layer.convolution_param
            pad = conv_param.pad
            k_w = conv_param.kernel_size
            k_h = k_w
    return pad, k_w, k_h

def get_layer_weigts(pt, model, layer_name,Debug=Debug):
    net = caffe.Net(pt, model, caffe.TEST)
    net.forward()

    conv_names =[]
    prune_convs_name =[]
    for blob_name in net._blob_names:
        if 'conv' in blob_name:
            conv_names.append(blob_name)
            if blob_name==layer_name:
                prune_convs_name.append(conv_names[-2])
                prune_convs_name.append(conv_names[-1])


    if Debug:print("bottom layer and top layer:",prune_convs_name)
    w1 = net.params[prune_convs_name[0]][0].data
    b1 = net.params[prune_convs_name[0]][1].data
    w2 = net.params[prune_convs_name[1]][0].data
    b2 = net.params[prune_convs_name[1]][1].data

    return net, w1, b1 , w2, b2 ,prune_convs_name


_pad, _k_w, _k_h = get_layer_params(model_def, prune_layer_name)
net, _w1 ,_b1, _w2, _b2 ,prune_convs_name= get_layer_weigts(model_def, model_weights, prune_layer_name,Debug)
_pad = _pad[0]
_k_w = _k_w[0]
_k_h = _k_h[0]

rank = int(_w1.shape[0]/ c_ratio)        #  conv2_1  c_in /c_ratio = 64/1.15 =55 

if Debug:
    print("w1 shape :", _w1.shape)
    print("b1 shape :", _b1.shape)
    print("w2 shape :", _w2.shape)
    print("b2 shape :", _b2.shape)

bottom_layer_name= net.bottom_names[prune_layer_name][0]
bottom_feature = net.blobs[bottom_layer_name].data
top_feature = net.blobs[prune_layer_name].data

pad_bt_feature = np.zeros((bottom_feature.shape[0], bottom_feature.shape[1],\
                            bottom_feature.shape[2]+ _pad*2 , bottom_feature.shape[3]+ _pad*2),\
                            dtype=bottom_feature.dtype)

pad_bt_feature[:,:,1:1+bottom_feature.shape[2], 1:1+bottom_feature.shape[3]]=bottom_feature

c_in = pad_bt_feature.shape[1] # 64   
c_out = top_feature.shape[1]   # 128

point_dict =dict()
top_feats_dict = np.ndarray(shape=(nBatch*nPointsPerLayer, c_out))    # (5000, 128)
btm_feats_dict = np.ndarray(shape=(nBatch*nPointsPerLayer, c_in, _k_w, _k_h))  #(5000, 64, 3, 3)

nPicsPerBatch = _k_w*_k_h   # 9
for batch in range(nBatch):
    
    randx = np.random.randint(0, top_feature.shape[2], nPointsPerLayer)
    randy = np.random.randint(0, top_feature.shape[3], nPointsPerLayer)
    
    point_dict[(prune_layer_name, batch, 'randx')]=randx
    point_dict[(prune_layer_name, batch, 'randy')]=randy
    
    """extract top layer(conv2_1) feature map """
    point_feat = top_feature[:,:, randx, randy].reshape(-1, nPointsPerLayer)     # (128, 10)
    top_feats_dict[batch*nPointsPerLayer:(batch+1)*nPointsPerLayer] = point_feat.T        # (10 , 128)

    """extract bottom layer(pool1) feature map"""
    for p in range(nPointsPerLayer):
       bottom_feat = pad_bt_feature[:,:, randx[p]: randx[p]+_k_w, randy[p]: randy[p]+_k_h]    # (1, 64, 3, 3)
       btm_feats_dict[batch*nPointsPerLayer+p] = bottom_feat
       

if Debug:
    print("top feat dict shape is ", top_feats_dict.shape)
    print("btm feat dict shape is ", btm_feats_dict.shape)

N = nBatch * nPointsPerLayer
samples = np.random.randint(0,N,250)

X = btm_feats_dict.copy()
reX = np.rollaxis(btm_feats_dict.reshape((N,c_in, -1))[samples], 1, 0)          #(64, 250 , 3*3)
reW2 = np.transpose(_w2.reshape((c_out, c_in, -1)),[1, 2, 0])

Y = top_feats_dict - _b2.reshape((1,-1))
reY = Y[samples].reshape(-1)
Z =np.matmul(reX, reW2).reshape((c_in, -1)).T

res = rel_error(X.reshape(X.shape[0], -1).dot(_w2.reshape((_w2.shape[0], -1)).T), Y)
print('rMSE = ', res)

_solver = Lasso(alpha=1e-4, warm_start=True,selection='random' )

def solve(alpha):
    _solver.alpha=alpha
    _solver.fit(Z, reY)
    
    idxs = _solver.coef_ != 0
    tmp = sum(idxs)
    if Debug:print('Lasso score is ',_solver.score(Z,reY),end='\t')
    return idxs, tmp

left = 0
right = alpha
lbound = rank
rbound = rank + 0.1*rank

while True:
    _, tmp =solve(right)

    if tmp<rank:
        break
    else:
        right *=2
        if Debug:print("right is ",right, 'tmp is', tmp)

print('Lasso Regression Done')

while True:
    alpha = (left+right)/2
    idxs, tmp = solve(alpha)

    if Debug:print('tmp=%d, alpha=%f, left=%f, right=%f'%(tmp, alpha, left,right))
    if tmp > rbound:
        left =alpha
    elif tmp < lbound:
        right = alpha
    else:
        break

print("check again: rank is ",tmp)

rank = tmp

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

newW2, newB2 = fc_kernel(X[:, idxs, ...].reshape((N,-1)), Y, W=_w2[:, idxs, ...].reshape(c_out, -1), B=_b2)
newW2= newW2.reshape((c_out, rank, _k_w, _k_h))

_w2[:, ~idxs, ...]= 0
_w2[:,  idxs, ...]=newW2.copy()

newWeight2 = _w2[:, idxs, ...]


res = rel_error(X.reshape(X.shape[0],-1).dot(_w2.reshape(_w2.shape[0],-1).T),Y)
print("rebuild rMSE is ",res)

def make_channel_pruning_model(input_file, output_file, bottom_layer):
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

"""generator  new protobuf """
make_channel_pruning_model(model_def, model_def_out, prune_convs_name[0], prune_convs_name[1])

"""generator new caffemodel"""
channel_net = caffe.Net(model_def_out, caffe.TEST)
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


print(prune_layer_name)
print('Done')
