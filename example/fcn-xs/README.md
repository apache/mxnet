FCN-xs EXAMPLES
---------------
This folder contains the examples of image segmentation in MXNet.

### step1: get the fully convulutional style of vgg16 model  
* dwonload the vgg16 caffe-model from [VGG_ILSVRC_16_layers.caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel),and the corresponding [VGG_ILSVRC_16_layers_deploy.prototxt](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt), the vgg16 model has [license](http://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
* use convert_model.py to convet the caffe model to mxnet model, like(shell):
``` 
    vgg16_deploy=VGG_ILSVRC_16_layers_deploy.prototxt
    vgg16_caffemodel=VGG_ILSVRC_16_layers.caffemodel
    model_prefix=VGG_ILSVRC_16_layers
    cmd=../../tools/caffe_converter/convert_model.py
    python $cmd $vgg16_deploy $vgg16_caffemodel VGG_ILSVRC_16_layers
    mv VGG_ILSVRC_16_layers-0001.params VGG_ILSVRC_16_layers-0074.params
```
* convet conv+fully-connect style to fully convolutional style, like(shell):
```
    python create_vgg16fc_model.py VGG_ILSVRC_16_layers 74 VGG_FC_ILSVRC_16_layers
```
  you can use vgg16fc model now, or you can download it directly from [yun.baidu](http://pan.baidu.com/s/1jGlOvno).  
  
  **`be careful: if you put one (very) large image to the vgg16fc model, you should change the 'workspace_default' value larger(Related to your field) in create_vgg16fc_model.py.`**


### step2: TODO
