# SSD: Single Shot MultiBox Object Detector

SSD is an unified framework for object detection with a single network.

You can use the code to train/evaluate/test for object detection task.

### Disclaimer
This is a re-implementation of original SSD which is based on caffe. The official
repository is available [here](https://github.com/weiliu89/caffe/tree/ssd).
The arXiv paper is available [here](http://arxiv.org/abs/1512.02325).

This example is intended for reproducing the nice detector while fully utilize the
remarkable traits of MXNet.
* The model is fully compatible with caffe version.
* The prediction result is almost identical to the original version. However, due to different non-maximum suppression Implementation, the results might differ slightly.

Due to the permission issue, this example is maintained in this [repository](https://github.com/zhreshold/mxnet-ssd) separately. You can use the link regarding specific per example [issues](https://github.com/zhreshold/mxnet-ssd/issues).

### Demo results
![demo1](https://cloud.githubusercontent.com/assets/3307514/19171057/8e1a0cc4-8be0-11e6-9d8f-088c25353b40.png)
![demo2](https://cloud.githubusercontent.com/assets/3307514/19171063/91ec2792-8be0-11e6-983c-773bd6868fa8.png)
![demo3](https://cloud.githubusercontent.com/assets/3307514/19171086/a9346842-8be0-11e6-8011-c17716b22ad3.png)

### mAP
|        Model          | Training data    | Test data |  mAP |
|:-----------------:|:----------------:|:---------:|:----:|
| VGG16_reduced 300x300 | VOC07+12 trainval| VOC07 test| 71.57|

### Speed
|         Model         |   GPU            | CUDNN | Batch-size | FPS* |
|:---------------------:|:----------------:|:-----:|:----------:|:----:|
| VGG16_reduced 300x300 | TITAN X(Maxwell) | v5.1  |     16     | 95   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) | v5.1  |     8      | 95   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) | v5.1  |     1      | 64   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) |  N/A  |     8      | 36   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) |  N/A  |     1      | 28   |
- *Forward time only, data loading and drawing excluded.*


### Getting started
* You will need python modules: `easydict`, `cv2`, `matplotlib` and `numpy`.
You can install them via pip or package managers, such as `apt-get`:
```
sudo apt-get install python-opencv python-matplotlib python-numpy
sudo pip install easydict
```

* Build MXNet: Follow the official instructions
```
# for Ubuntu/Debian
cp make/config.mk ./config.mk
```
Remember to enable CUDA if you want to be able to train, since CPU training is
insanely slow. Using CUDNN is optional.

### Try the demo
* Download the pretrained model: [`ssd_300.zip`](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.2-alpha/ssd_300_vgg16_reduced_voc0712_trainval.zip), and extract to `model/` directory. (This model is converted from VGG_VOC0712_SSD_300x300_iter_60000.caffemodel provided by paper author).
* Run
```
# cd /path/to/mxnet/example/ssd/
# grab demo images
python data/demo/download_demo_images.py
# run demo.py with defaults
python demo.py
# play with examples:
python demo.py --epoch 0 --images ./data/demo/dog.jpg --thresh 0.5
```
* Check `python demo.py --help` for more options.

### Train the model
This example only covers training on Pascal VOC dataset. Other datasets should
be easily supported by adding subclass derived from class `Imdb` in `dataset/imdb.py`.
See example of `dataset/pascal_voc.py` for details.
* Download the converted pretrained `vgg16_reduced` model [here](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.2-beta/vgg16_reduced.zip), unzip `.param` and `.json` files
into `model/` directory by default.
* Download the PASCAL VOC dataset, skip this step if you already have one.
```
cd /path/to/where_you_store_datasets/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```
* We are goint to use `trainval` set in VOC2007/2012 as a common strategy.
The suggested directory structure is to store `VOC2007` and `VOC2012` directories
in the same `VOCdevkit` folder.
* Then link `VOCdevkit` folder to `data/VOCdevkit` by default:
```
ln -s /path/to/VOCdevkit /path/to/mxnet/example/ssd/data/VOCdevkit
```
Use hard link instead of copy could save us a bit disk space.
* Start training:
```
# cd /path/to/mxnet/example/ssd
python train.py
```
* By default, this example will use `batch-size=32` and `learning_rate=0.001`.
You might need to change the parameters a bit if you have different configurations.
Check `python train.py --help` for more training options. For example, if you have 4 GPUs, use:
```
# note that a perfect training parameter set is yet to be discovered for multi-GPUs
python train.py --gpus 0,1,2,3 --batch-size 128 --lr 0.0005
```
* Memory usage: MXNet is very memory efficient, training on `VGG16_reduced` model with `batch-size` 32 takes around 4684MB without CUDNN.
* Initial lenarning rate: 0.001 is fine for single GPU. 0.0001 should be used for the first couple of epoches then go back to 0.001 via using parameter --resume.

### Evalute trained model
Again, currently we only support evaluation on PASCAL VOC
Use:
```
# cd /path/to/mxnet/example/ssd
python evaluate.py --gpus 0,1 --batch-size 128 --epoch 0
```
### Convert model to deploy mode
This simply removes all loss layers, and attach a layer for merging results and non-maximum suppression.
Useful when loading python symbol is not available.
```
# cd /path/to/mxnet/example/ssd
python deploy.py --num-class 20
```
