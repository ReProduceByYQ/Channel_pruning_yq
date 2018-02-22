# Channel_pruning_yq
This repo is re-produce for Channel_pruning, Framework is [Caffe](https://github.com/BVLC/caffe).

The original source code :[yihui-he](https://github.com/yihui-he/channel-pruning).

### Introduction
Two main function in my implementation  :

* [Low rank decompose](https://github.com/chengtaipu/lowrankcnn/tree/master/imagenet)
* channel prune

### Notice:
 Now, it is only for native ConvNet such as VGG, not work for multi-branch ConvNet such as ResNet. I don`t implementation channel decompose ,because it is unhelpful in my test experiment!!!! So ,I have 2C not 3C.

File  | Intro |
:-------------------------:|:-------------------------
__channel_pruning_one_layer.py__  | This is Channel Pruning reproduct，but only for pruning one layer, you should modify "prune_layer_name"  in Line 49  for youself.
__channel_pruning_reproduce.py__  | Such as pruning layer "conv2_1", This file tell you how to implement step-by-step. Including _Get Feature()_, _Lasso Regression()_, _Linear Regression()_, _Generator New Protobuf()_, _Generator New Weights()_
__low_rank_and_channel_pruning.py__ | All funtion in here, such as __Low Rank__ and __Channel Pruning(layer-by-layer)__, It needs two configure file:__config.json__ and __config_cratio.json__ , you can modify them in Line 45 and Line 49.
__config.json__ |   It is for __Low rank__ configuration, the number obtained from the experiment.
__config_cratio.json__ | It is for __Channel Pruning(layer-by-layer)__ configuration, the number means that how many channels will be keep.

### Cite

    @InProceedings{He_2017_ICCV,
    author = {He, Yihui and Zhang, Xiangyu and Sun, Jian},
    title = {Channel Pruning for Accelerating Very Deep Neural Networks},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2017}
    }

### Todo
- [x] support VGG
- [x] Combination of Conv layer and BN layer
- [ ] support ResNet
- [ ] support Faster RCNN