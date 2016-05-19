# Fast R-CNN in MXNet

Fast R-CNN treats general object detection as a classification problem and
bounding box prediction as a regression problem. Classifying cropped region
feature maps and predicting bounding box displacements together yields
detection results. Cropping feature maps instead of image input accelerates
computation utilizing shared convolution maps. Bounding box displacements
are simultaneously learned in the training process.

## Getting Started

* MXNet with `ROIPooling` and `smooth_l1` operators are required
* Download data and place them to `data` folder according to `Data Folder Structure`.
  You might want to create a symbolic link to VOCdevkit folder
```
Pascal VOCdevkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
Ross's precomputed object proposals
https://github.com/rbgirshick/fast-rcnn/blob/master/data/scripts/fetch_selective_search_data.sh
```
* Data Folder Structure (suppose root is `data`)
```
demo
selective_search_data
cache (created by imdb)
-- name + source + roidb.pkl (create by imdb)
-- name (created by detection and evaluation)
VOCdevkit
-- VOC + year (JPEG images and annotations)
-- results (created by evaluation)
---- VOC + year
------ main
-------- comp4_det_val_aeroplane.txt
```
* Download VGG16 pretrained model, use `mxnet/tools/caffe_converter` to convert it,
  rename to `vgg16-symbol.json` and `vgg16-0001.params` and place it in `model` folder
* Download 'demo' data and put it in `data/demo` from
```
https://github.com/rbgirshick/fast-rcnn/tree/master/data/demo
```

## Training
* Start training by run `python train.py`. Variable args can be found by run
`python train.py --help`.
* Training can be done in cpu, modify `train.py` accordingly.
```
usage: train.py [-h] [--image_set IMAGE_SET] [--year YEAR]
                [--root_path ROOT_PATH] [--devkit_path DEVKIT_PATH]
                [--pretrained PRETRAINED] [--epoch EPOCH] [--prefix PREFIX]
                [--gpu GPU_ID] [--begin_epoch BEGIN_EPOCH]
                [--end_epoch END_EPOCH] [--frequent FREQUENT]

Train a Fast R-CNN network

optional arguments:
  -h, --help            show this help message and exit
  --image_set IMAGE_SET
                        can be trainval or train
  --year YEAR           can be 2007, 2010, 2012
  --root_path ROOT_PATH
                        output data folder
  --devkit_path DEVKIT_PATH
                        VOCdevkit path
  --pretrained PRETRAINED
                        pretrained model prefix
  --epoch EPOCH         epoch of pretrained model
  --prefix PREFIX       new model prefix
  --gpu GPU_ID          GPU device to train with
  --begin_epoch BEGIN_EPOCH
                        begin epoch of training
  --end_epoch END_EPOCH
                        end epoch of training
  --frequent FREQUENT   frequency of logging
```

## Testing
* Start testing by run `python test.py`. Variable args can be found by run
`python test.py --help`.
* Testing can be done in cpu, modify `test.py` accordingly.
```
usage: test.py [-h] [--image_set IMAGE_SET] [--year YEAR]
               [--root_path ROOT_PATH] [--devkit_path DEVKIT_PATH]
               [--prefix PREFIX] [--epoch EPOCH] [--gpu GPU_ID]

Test a Fast R-CNN network

optional arguments:
  -h, --help            show this help message and exit
  --image_set IMAGE_SET
                        can be test
  --year YEAR           can be 2007, 2010, 2012
  --root_path ROOT_PATH
                        output data folder
  --devkit_path DEVKIT_PATH
                        VOCdevkit path
  --prefix PREFIX       new model prefix
  --epoch EPOCH         epoch of pretrained model
  --gpu GPU_ID          GPU device to test with
```

## Demonstration
* If no training has been done, download reference model from Ross Girshick and use
`mxnet/caffe/caffe_converter` to convert it to MXNet.
```
https://github.com/rbgirshick/fast-rcnn/blob/master/data/scripts/fetch_fast_rcnn_models.sh
```
* Run demo by `demo.py --gpu 0 --prefix path-to-model --epoch 0`, in which
`path-to-model + '%4d' % epoch.params` will be the params file and
`path-to-model + '-symbol.json'` will be the symbol json.
* Demo can be run in cpu, modify `demo.py` accordingly.
```
usage: demo.py [-h] [--prefix PREFIX] [--epoch EPOCH] [--gpu GPU_ID]

Demonstrate a Fast R-CNN network

optional arguments:
  -h, --help       show this help message and exit
  --prefix PREFIX  new model prefix
  --epoch EPOCH    epoch of pretrained model
  --gpu GPU_ID     GPU device to test with
```

## Disclaimer
This repository used code from [MXNet](https://github.com/dmlc/mxnet),
[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn),
[Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn),
[caffe](https://github.com/BVLC/caffe). Training data are from
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/),
[ImageNet](http://image-net.org/). Model comes from
[VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).