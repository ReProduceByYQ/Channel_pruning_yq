"""
Step 1:
"""
#%%
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import scipy
import matplotlib.pyplot as plt
# display plots in this notebook
#%matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import sys
sys.path.insert(0, 'D:/Deep_Learning/Win_Caffe/caffe-blvc/python')
#print(sys.path)
import caffe

#caffe.set_device(0)
caffe.set_mode_cpu()

"""
Step 2:
"""
model_def = 'D:/INstall_files/channel_pruning/train_val.prototxt'
model_def_out = 'D:/INstall_files/channel_pruning/train_val_out.prototxt'
model_weights = 'D:/INstall_files/channel_pruning/vgg16__iter_10000.caffemodel'
model_weights_out = 'D:/INstall_files/channel_pruning/vgg16__iter_out.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
outputs = net.forward()

"""
Step 3:
"""
#%%
Debug = True
nBatch = 500         # 500 batch
nPointsPerLayer=10   # 10  point per layer  

feat_map_conv2_1 = net.blobs['conv2_1'].data  # shape (1, 128, 112, 112)  top layer
bottom_feat_map  = net.blobs['pool1'].data    # shape (1 ,64 ,112 ,112)   bottom layer
weights   = net.params['conv2_1'][0].data     # shape (128, 64, 3, 3)
k_w = weights.shape[2]   # 3
k_h = weights.shape[3]   # 3
bottom_bias = net.params['conv2_1'][1].data   # shape (128,)

# pad_feature_map shape is (1, 64, 114, 114)
pad = 1 
pad_feature_map = np.zeros((bottom_feat_map.shape[0], bottom_feat_map.shape[1], \
                            bottom_feat_map.shape[2]+ pad*2, bottom_feat_map.shape[3]+pad*2))
pad_feature_map[:,:,1:1+ feat_map_conv2_1.shape[2], 1:1+feat_map_conv2_1.shape[3]] = bottom_feat_map

c_out = feat_map_conv2_1.shape[1]          # 128 
c_in  = bottom_feat_map.shape[1]           # 64

point_dict = dict()  # store sample points  total= nBatch * nPointsPerLayer
feats_dict = np.ndarray(shape=(nBatch*nPointsPerLayer, c_out))  # store sample  feature at sample point Loc 
# conv2_1 feats_dict  shape is \
#        (nBatch * nPointsPerLayer, channel)
bottom_feat_dict = np.ndarray(shape=(nBatch*nPointsPerLayer, c_in, k_w, k_h))   # (5000, 64,3,3)
bottom_x = np.ndarray(shape=(k_w*k_h*nBatch*nPointsPerLayer, c_in))                             # (3*3*5000, 64)

def rel_error(A,B):
    return np.mean((A-B)**2)**0.5/ np.mean(A**2)**.5
def relu(x):
    return np.maximum(x, 0.0)

batch_idx = 0
nPicsPerBatch = k_w * k_h
for batch in range(nBatch):
    
    randx = np.random.randint(0, feat_map_conv2_1.shape[2], nPointsPerLayer)
    randy = np.random.randint(0, feat_map_conv2_1.shape[3], nPointsPerLayer)
    
    point_dict[('conv2_1', batch, 'randx')]=randx
    point_dict[('conv2_1', batch, 'randy')]=randy
    
    """extract top layer(conv2_1) feature map """
    point_feat = feat_map_conv2_1[:,:, randx, randy].reshape(-1, nPointsPerLayer)     # (128, 10)
    feats_dict[batch*nPointsPerLayer:(batch+1)*nPointsPerLayer] = point_feat.T        # (10 , 128)

    """extract bottom layer(pool1) feature map"""
    for p in range(nPointsPerLayer):
       bottom_feat = pad_feature_map[:,:, randx[p]: randx[p]+k_w, randy[p]: randy[p]+k_h]    # (1, 64, 3, 3)
       bottom_feat_dict[batch*nPointsPerLayer+p] = bottom_feat
       
       i_from = batch_idx + p*nPicsPerBatch
       bottom_x[i_from: (i_from+nPicsPerBatch)] = np.moveaxis(bottom_feat,1,-1).reshape((nPicsPerBatch, -1))

    
    batch_idx += nPicsPerBatch * nPointsPerLayer            # += 90

print('top_layer feature map shape is ',feats_dict.shape)
print('bottom_layer feature map shape is ', bottom_feat_dict.shape)

N = nBatch*nPointsPerLayer
samples = np.random.randint(0, N, 250)      # samples 250 point from nBatch*nPointsPerLayer=5000

X = bottom_feat_dict.copy()                                               # (5000, 64,3, 3)

reX = np.rollaxis(bottom_feat_dict.reshape((N,c_in, -1))[samples], 1, 0)  # ( 64,250, 3*3)
reW2 = np.transpose(weights.reshape((c_out, c_in, -1)),[1, 2, 0])         # (64, 3*3, 128)

B2 = bottom_bias.copy()
Y = feats_dict.copy() - B2.reshape(1,-1)                         #(5000, 128)
reY = feats_dict[samples].reshape(-1)                            # (250, 128) -> (250*128, )

tmpZ = np.matmul(reX,reW2)                                       # (64,250,128) 
tmpB2 = B2.reshape((1,1,c_out))                                  # (1,1, 128)
#tmpZ = relu(tmpZ)
tmpZ2 = tmpZ + tmpB2                                            # (64, 250, 128)
Z2= tmpZ2.reshape((c_in,-1)).T

Z = np.matmul(reX, reW2).reshape((c_in, -1)).T                   # (64, 250*128) - >  (32000, 64)


#%%
if Debug:
    res = rel_error(X.reshape(X.shape[0],-1).dot(weights.reshape(weights.shape[0],-1).T),Y)
    print('rMSE = ',res)
    res = rel_error(relu(X.reshape(X.shape[0],-1)).dot(weights.reshape(weights.shape[0],-1).T),Y)
    print('relu(x)- rMSE = ',res)
    res = rel_error(relu(X.reshape(X.shape[0],-1).dot(weights.reshape(weights.shape[0],-1).T)),Y)
    print('relu - rMSE = ',res)


"""
Step 4:  Lasso regression
"""
from sklearn.linear_model import *
_solver = Lasso(alpha=1e-4, warm_start=True,selection='random' )
rank = 55

def solve(alpha):
    
    _solver.alpha=alpha
    _solver.fit(Z, reY)
    #_solver.fit(Z, reY)
    idxs = _solver.coef_ != 0
    tmp = sum(idxs)
    if Debug:print('Lasso score is ',_solver.score(Z,reY),end='\t')
    return idxs, tmp

left = 0
right = 1e-3
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

print("check again")
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

newW2, newB2 = fc_kernel(X[:, idxs, ...].reshape((N,-1)), Y, W=weights[:, idxs, ...].reshape(c_out, -1), B=B2)
newW2= newW2.reshape((c_out, rank, k_w, k_h))

"""
return idxs, newW2, newB2
"""

weights[:, ~idxs, ...] = 0
weights[:, idxs, ...]=newW2.copy()

bottom_bias = B2.reshape(-1).copy()

newWeight = weights[:,idxs, ...].copy()
print("newWeight shape =",newWeight.shape)

#%%
if Debug:
    print('\n','------Final-----')
    res = rel_error(X.reshape(X.shape[0],-1).dot(weights.reshape(weights.shape[0],-1).T),Y)
    print('rMSE = ',res)
    res = rel_error(relu(X.reshape(X.shape[0],-1)).dot(weights.reshape(weights.shape[0],-1).T),Y)
    print('relu(x) - rMSE = ',res)
    res = rel_error(relu(X.reshape(X.shape[0],-1).dot(weights.reshape(weights.shape[0],-1).T)),Y)
    print('relu - rMSE = ',res)

print('All Done')

       
"""
Step 5: save weight and prototxt
"""
from google.protobuf import text_format
from argparse import ArgumentParser

from caffe.proto.caffe_pb2 import NetParameter, LayerParameter

def make_channel_pruning_model(input_file, output_file):
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
        if layer.name != 'conv1_2':
            new_layer.append(layer)
        else:
            newConv2_1 = _creat_new('conv1_2', layer)
            conv_param = newConv2_1.convolution_param
            conv_param.num_output = rank
            new_layer.append(newConv2_1)

    new_net = NetParameter()
    new_net.CopyFrom(nett)
    del(new_net.layer[:])

    new_net.layer.extend(new_layer)
    with open(output_file,'w') as fp:
        fp.write(text_format.MessageToString(new_net))


"""generator  new protobuf """
make_channel_pruning_model(model_def, model_def_out)

"""generator new caffemodel"""
channel_net = caffe.Net(model_def_out, caffe.TEST)
for layer_name, param in channel_net.params.items():
    print("----------")
    
    if layer_name == 'conv1_2':
        orig_w , orig_b = [p.data for p in net.params[layer_name]]
        param[0].data[...] = orig_w[idxs,...].copy()
        param[1].data[...] = orig_b[idxs,...].copy()

    elif layer_name == 'conv2_1':
        param[0].data[...] = newWeight.copy()
        param[1].data[...] = bottom_bias.reshape(param[1].data.shape).copy()
    
    
    elif layer_name != 'conv2_1':
        print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))
        
        print(layer_name + '\t' + str(net.params[layer_name][0].data.shape), str(net.params[layer_name][1].data.shape))
        orig_w , orig_b = [p.data for p in net.params[layer_name]]
        new_w , new_b = param[0].data, param[1].data
        new_w[...]=orig_w.copy()
        new_b[...]=orig_b.copy()
           
channel_net.save(model_weights_out)


print('generator new caffemodel: ',model_weights_out)


        
