
"""
This is Channel Pruning reproduct，but only for pruning one layer
Author： YQ
data：2018-1-25
Function: 1.  Low rank   2. channel prune

"""
# 1. set import libs
import numpy as np
from sklearn.linear_model import *
import sys, json
sys.path.insert(0, 'D:/Deep_Learning/Win_Caffe/caffe-blvc/python')
import caffe
from google.protobuf import text_format
import google.protobuf as pb
from argparse import ArgumentParser
from caffe.proto.caffe_pb2 import NetParameter, LayerParameter
from util import *    # my function
device=0   # 0 is cpu  ; 1 is gpu
if device:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()
#%%
model_def = 'D:/INstall_files/channel_pruning/train_val.prototxt'
model_weights = 'D:/INstall_files/channel_pruning/vgg16__iter_10000.caffemodel'

f, t =model_def.split(".")
fw, tw = model_weights.split(".")

model_lr_def = f+'_lr.'+t
model_lr_weights = fw+'_lr.'+tw

model_cp_def = f+'_lr_cp.' +t
model_cp_weights = fw+'_lr_cp.'+tw

if 0:
    print('lr_def =',model_lr_def)
    print('lr_weights =',model_lr_weights)
    print('channel def = ', model_cp_def)
    print('channel weights =', model_cp_weights)

conf = load_config('D:/CodeBook/Python/lowrankcnn/imagenet/models_vgg/config.json')
prune_names = [name+'_h' for name in conf if name!='conv5_3']     # don`t remove con5_3

# store cratio every layer
conf_cratio = load_config('D:/CodeBook/Python/lowrankcnn/imagenet/models_vgg/config_cratio.json')
#cratios=[conf_cratio[x[:-2]] for x in prune_names]
# Low Rank Decomposes
if 1:
    make_lowrank_model(model_def, conf, model_lr_def)
    
    approx_lowrank_weights(model_def, model_weights, conf, model_lr_def,
                           model_lr_weights)

print('Low Rank Done')


#bottom_layer_name ="pool1" # (1, 64, 112, 112)  # we will channel prune on this layer ,So we change num_output on this layer
# conv1_2_h - > relu -> pool   so prune channel of conv1_2_h  layer
#top_layer_name = "conv2_1_v"   # (1, 48 ,224, 224)
#prune_layer_name = "conv1_2_h"

cfg=dict()
cfg['nPointsPerLayer'] = 10
cfg['nBatch'] = 500
cfg['c_ratio'] = 0.869
cfg['alpha'] = 1e-3
N = cfg['nBatch'] * cfg['nPointsPerLayer']


for prune_layer_name in prune_names:
    cfg['c_ratio'] = conf_cratio[prune_layer_name[:-2]]   # c_ratio

    lr_net = caffe.Net(model_lr_def , model_lr_weights, caffe.TEST)
    convs_name = lr_net._blob_names
#if prune_layer_name=='conv1_2_h':
    print("-----------------------------------------")
    print(" Start channel prune :", prune_layer_name, 'C_ratio = ',cfg['c_ratio'])
    print('------------------------------------------')

    bottom_layer_name, top_layer_name = extract_layer_name(lr_net, prune_layer_name)

    params = get_layer_params(model_lr_def, top_layer_name)
    print('layer params :', params)
    cfg['params']= params    #(pad_h, pad_w, k_h, k_w)
    pad_h, pad_w, k_h, k_w = params

    samples =np.random.randint(0, N, 250)
    
    X, w2, b2 ,Y, c_in, c_out = extract_feat(prune_layer_name, cfg, lr_net)
    reX = np.rollaxis(X.reshape((N,c_in, -1))[samples], 1, 0)          #(64, 250 , 3*3)
    reW2 = np.transpose(w2.reshape((c_out, c_in, -1)),[1, 2, 0])

    reY = Y[samples].reshape(-1)
    Z =np.matmul(reX, reW2).reshape((c_in, -1)).T
    #Z = relu(Z)

    _solver = Lasso(alpha=1e-4, warm_start=True,selection='random' )
    def solve(alpha):
        _solver.alpha=alpha
        _solver.fit(Z, reY)
    
        idxs = _solver.coef_ != 0
        tmp = sum(idxs)
        if 1:print('Lasso score is ',_solver.score(Z,reY),end='\t')
        return idxs, tmp

    left = 0
    right = cfg['alpha']
    lbound = int(c_in * cfg['c_ratio'])    # 64 * 0.869 ~ 55
    rank = lbound
    rbound = 1.1*lbound

    while True:
        _, tmp =solve(right)

        if tmp<rank:
            break
        else:
            right *=2
            if 1:print("right is ",right, 'tmp is', tmp)
    if 0: print(prune_layer_name+'- Lasso Regression Done.')

    while True:
        alpha = (left+right)/2
        idxs, tmp = solve(alpha)

        if 1:print('tmp=%d, alpha=%f, left=%f, right=%f'%(tmp, alpha, left,right))
        if tmp > rbound:
            left =alpha
        elif tmp < lbound:
            right = alpha
        else:
            break

    if 1:print(prune_layer_name+" - check again: rank is ",tmp)
    rank = tmp

    newW2, newB2 = fc_kernel(X[:, idxs, ...].reshape((N,-1)), Y, W=w2[:, idxs, ...].reshape(c_out, -1), B=b2)
    newW2= newW2.reshape((c_out, rank, k_h, k_w))

    w2[:, ~idxs, ...]= 0
    w2[:,  idxs, ...]=newW2.copy()

    newWeight2 = w2[:, idxs, ...]                      # (48, 55 , 3 ,1)
    if 1: 
        res = rel_error(X.reshape(X.shape[0],-1).dot(w2.reshape(w2.shape[0],-1).T),Y)
        print("After Lasso, rMSE =", res)
        res_relu = rel_error(relu(X.reshape(X.shape[0],-1).dot(w2.reshape(w2.shape[0],-1).T)),Y)
        print('After Lasso, rMSE-relu =', res_relu)
    
    """generator new protobuf"""
    make_channel_pruning_model(model_lr_def, model_cp_def, prune_layer_name, rank)   # (64) -> 55

    """generator new weights"""
    channel_net = caffe.Net(model_cp_def, caffe.TEST)
    for layer_name, param in channel_net.params.items():   
        if layer_name == prune_layer_name:
            orig_w , orig_b = [p.data for p in lr_net.params[layer_name]]
            param[0].data[...] = orig_w[idxs,...].copy()
            param[1].data[...] = orig_b[idxs,...].copy()
        elif layer_name == top_layer_name:
            param[0].data[...] = newWeight2.copy()
            param[1].data[...] = b2.reshape(param[1].data.shape).copy()
        else:
            orig_w , orig_b = [p.data for p in lr_net.params[layer_name]]
            new_w , new_b = param[0].data, param[1].data
            new_w[...]=orig_w.copy()
            new_b[...]=orig_b.copy()
           
    channel_net.save(model_cp_weights)
    print(prune_layer_name)

    model_lr_def = model_cp_def
    model_lr_weights = model_cp_weights

print('Low -Rank and Channel Pruning Done!!')




    
















