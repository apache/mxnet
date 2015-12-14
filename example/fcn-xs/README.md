FCN-xs EXAMPLES
---------------
This folder contains the examples of image segmentation in MXNet.

## Sample results
![fcn-xs pasval_voc result](C:\Users\Administrator\Desktop\fcn\fcn-xs_pascal.jpg)

## How to train fcn-xs in mxnet
#### step1: get the fully convulutional style of vgg16 model  
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
* or you can directly download the ```VGG_FC_ILSVRC_16_layers-symbol.json``` and ```VGG_FC_ILSVRC_16_layers-0074.params``` from [yun.baidu](http://pan.baidu.com/s/1jGlOvno)

#### step2: prepare your training Data
in the example here, the training image list owns the form:  
```index \t image_data_path \t image_label_path```
the labels for one image in image segmentation field is also one image, with the same shape of input image.  
* or you can directly download the ```VOC2012.rar``` from [yun.baidu](http://pan.baidu.com/s/1jGlOvno), and Extract it. the file/folder will be:  
```JPEGImages folder```, ```SegmentationClass folder```, ```train.lst```, ```val.lst```, ```test.lst```

#### step3: begin training fcn-xs
if you want to train the fcn-8s model, it's better for you trained the fcn-32s and fcn-16s model firstly.
when training the fcn-32s model, run in shell ```./run_fcn32s.sh```
* in the fcn_xs.py(e.g. fcn_32s.py, fcn_16s.py, fcn_8s.py), you may need to change the directory ```img_dir```, ```train_lst```,  ```val_lst```, ```fcnxs_model_prefix```
* the output log may like this(when training fcn-8s):
```c++
INFO:root:Start training with gpu(3)
INFO:root:Epoch[0] Batch [50]   Speed: 1.16 samples/sec Train-accuracy=0.894318
INFO:root:Epoch[0] Batch [100]  Speed: 1.11 samples/sec Train-accuracy=0.904681
INFO:root:Epoch[0] Batch [150]  Speed: 1.13 samples/sec Train-accuracy=0.908053
INFO:root:Epoch[0] Batch [200]  Speed: 1.12 samples/sec Train-accuracy=0.912219
INFO:root:Epoch[0] Batch [250]  Speed: 1.13 samples/sec Train-accuracy=0.914238
INFO:root:Epoch[0] Batch [300]  Speed: 1.13 samples/sec Train-accuracy=0.912170
INFO:root:Epoch[0] Batch [350]  Speed: 1.12 samples/sec Train-accuracy=0.912080
```

## TODO
* add the example of using pretrained model
* add the crop_offset function in symbol(both c++ and python side of mxnet)
* make the example more cleaner(the code is some dirty here)
