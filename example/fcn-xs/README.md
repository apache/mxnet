FCN-xs EXAMPLES
---------------
This folder contains the examples of image segmentation in MXNet.

## Sample results
![fcn-xs pasval_voc result](https://github.com/dmlc/web-data/blob/master/mxnet/image/fcnxs-example-result.jpg)

we have trained a simple fcn-xs model, the parameter is below:

| model | lr (fixed) | epoch |
| ---- | ----: | ---------: |
| fcn-32s | 1e-10 | 31 |
| fcn-16s | 1e-12 | 27 |
| fcn-8s | 1e-14 | 19 |
(```when using the newest mxnet, you'd better using larger learning rate, such as 1e-4, 1e-5, 1e-6 instead, because the newest mxnet will do gradient normalization in SoftmaxOutput```)

the training image number is only : 2027, and the Validation image number is: 462  

## How to train fcn-xs in mxnet
#### step1: download the vgg16fc model and experiment data
* vgg16fc model : you can download the ```VGG_FC_ILSVRC_16_layers-symbol.json``` and ```VGG_FC_ILSVRC_16_layers-0074.params```   [baidu yun](http://pan.baidu.com/s/1bgz4PC), [dropbox](https://www.dropbox.com/sh/578n5cxej7ofd6m/AACuSeSYGcKQDi1GoB72R5lya?dl=0).  
this is the fully convolution style of the origin
[VGG_ILSVRC_16_layers.caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel), and the corresponding [VGG_ILSVRC_16_layers_deploy.prototxt](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt), the vgg16 model has [license](http://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
* experiment data : you can download the ```VOC2012.rar```  [robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), and Extract it. the file/folder will be like:  
```JPEGImages folder```, ```SegmentationClass folder```, ```train.lst```, ```val.lst```, ```test.lst```

#### step2: train fcn-xs model
* if you want to train the fcn-8s model, it's better for you trained the fcn-32s and fcn-16s model firstly.
when training the fcn-32s model, run in shell ```./run_fcnxs.sh```, the script in it is:
```shell
python -u fcn_xs.py --model=fcn32s --prefix=VGG_FC_ILSVRC_16_layers --epoch=74 --init-type=vgg16
```
* in the fcn_xs.py, you may need to change the directory ```root_dir```, ```flist_name```, ``fcnxs_model_prefix``` for your own data.
* when you train fcn-16s or fcn-8s model, you should change the code in ```run_fcnxs.sh``` corresponding, such as when train fcn-16s, comment out the fcn32s script, then it will like this:
```shell
 python -u fcn_xs.py --model=fcn16s --prefix=FCN32s_VGG16 --epoch=31 --init-type=fcnxs
```
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

## Using the pre-trained model for image segmentation
* similarly, you should firstly download the pre-trained model from  [yun.baidu](http://pan.baidu.com/s/1bgz4PC), the symbol and model file is ```FCN8s_VGG16-symbol.json```, ```FCN8s_VGG16-0019.params```
* then put the image in your directory for segmentation, and change the ```img = YOUR_IMAGE_NAME``` in ```image_segmentaion.py```
* lastly, use ```image_segmentaion.py``` to segmentation one image by run in shell ```python image_segmentaion.py```, then you will get the segmentation image like the sample result above.

## Tips
* this is the whole image size training, that is to say, we do not need resize/crop the image to the same size, so the batch_size during training is set to 1.
* the fcn-xs model is baed on vgg16 model, with some crop, deconv, element-sum layer added, so the model is some big, moreover, the example is using whole image size training, if the input image is some large(such as 700*500), then it may very memory consumption, so I suggest you using the GPU with 12G memory.
* if you don't have GPU with 12G memory, maybe you shoud change the ```cut_off_size``` to be a small value when you construct your FileIter, like this:  
```python
train_dataiter = FileIter(
      root_dir             = "./VOC2012",
      flist_name           = "train.lst",
      cut_off_size         = 400,
      rgb_mean             = (123.68, 116.779, 103.939),
      )
```
* we are looking forward you to make this example more powerful, thanks.
