<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# SSD: Single Shot MultiBox Object Detector

SSD is an unified framework for object detection with a single network.

You can use the code to train/evaluate/test for object detection task.

-------------------

## Gluon Implementation

You can find a Gluon implementation on [gluon-cv](https://gluon-cv.mxnet.io/build/examples_detection/train_ssd_voc.html).

-------------------

### Disclaimer
This is a re-implementation of original SSD which is based on caffe. The official
repository is available [here](https://github.com/weiliu89/caffe/tree/ssd).
The arXiv paper is available [here](http://arxiv.org/abs/1512.02325).

This example is intended for reproducing the nice detector while fully utilize the
remarkable traits of MXNet.
* Model [converter](#convert-caffemodel) from caffe is available now!
* The result is almost identical to the original version. However, due to different implementation details, the results might differ slightly.

Due to the permission issue, this example is maintained in this [repository](https://github.com/zhreshold/mxnet-ssd) separately. You can use the link regarding specific per example [issues](https://github.com/zhreshold/mxnet-ssd/issues).

### What's new
* Support training and inference on COCO dataset. Int8 inference achieves 0.253 mAP on CPU with MKL-DNN backend, which is a comparable accuracy to FP32 (0.2552 mAP).
* Support uint8 inference on CPU with MKL-DNN backend. Uint8 inference achieves 0.8364 mAP, which is a comparable accuracy to FP32 (0.8366 mAP).
* Added live camera capture and detection display (run with --camera flag). Example:
    `./demo.py --camera --cpu --frame-resize 0.5`
* Added multiple trained models.
* Added a much simpler way to compose network from mainstream classification networks (resnet, inception...) and [Guide](symbol/README.md).
* Update to the latest version according to caffe version, with 5% mAP increase.
* Use C++ record iterator based on back-end multi-thread engine to achieve huge speed up on multi-gpu environments.
* Monitor validation mAP during training.
* More network symbols under development and test.
* Extra operators are now in `mxnet/src/operator/contrib`.
* Old models are incompatible, use [e06c55d](https://github.com/apache/incubator-mxnet/commits/e06c55d6466a0c98c7def8f118a48060fb868901) or [e4f73f1](https://github.com/apache/incubator-mxnet/commits/e4f73f1f4e76397992c4b0a33c139d52b4b7af0e) for backward compatibility. Or, you can modify the json file to update the symbols if you are familiar with it, because only names have changed while weights and bias should still be good.

### Demo results
![demo1](https://cloud.githubusercontent.com/assets/3307514/19171057/8e1a0cc4-8be0-11e6-9d8f-088c25353b40.png)
![demo2](https://cloud.githubusercontent.com/assets/3307514/19171063/91ec2792-8be0-11e6-983c-773bd6868fa8.png)
![demo3](https://cloud.githubusercontent.com/assets/3307514/19171086/a9346842-8be0-11e6-8011-c17716b22ad3.png)

### mAP
|        Model          | Training data    | Test data |  mAP | Note |
|:-----------------:|:----------------:|:---------:|:----:|:-----|
| [VGG16_reduced 300x300](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.5-beta/vgg16_ssd_300_voc0712_trainval.zip) | VOC07+12 trainval| VOC07 test| 77.8| fast |
| [VGG16_reduced 512x512](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.5-beta/vgg16_ssd_512_voc0712_trainval.zip) | VOC07+12 trainval | VOC07 test| 79.9| slow |
| [Inception-v3 512x512](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/inceptionv3_ssd_512_voc0712_trainval.zip) | VOC07+12 trainval| VOC07 test| 78.9 | fastest |
| [Resnet-50 512x512](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/resnet50_ssd_512_voc0712_trainval.zip) | VOC07+12 trainval| VOC07 test| 78.9 | fast |

### Speed
|         Model         |   GPU            | CUDNN | Batch-size | FPS* |
|:---------------------:|:----------------:|:-----:|:----------:|:----:|
| VGG16_reduced 300x300 | TITAN X(Maxwell) | v5.1  |     16     | 95   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) | v5.1  |     8      | 95   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) | v5.1  |     1      | 64   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) |  N/A  |     8      | 36   |
| VGG16_reduced 300x300 | TITAN X(Maxwell) |  N/A  |     1      | 28   |
*Forward time only, data loading and drawing excluded.*


### Getting started
* You will need python modules: `cv2`, `matplotlib` and `numpy`.
If you use mxnet-python api, you probably have already got them.
You can install them via pip or package managers, such as `apt-get`:
```
sudo apt-get install python-opencv python-matplotlib python-numpy
```

* Build MXNet: Follow the official instructions
```
# for Ubuntu/Debian
cp make/config.mk ./config.mk
# enable cuda, cudnn if applicable
```
Remember to enable CUDA if you want to be able to train, since CPU training is
insanely slow. Using CUDNN is optional, but highly recommended.

### Try the demo
* Download the pretrained model: [`ssd_resnet50_0712.zip`](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/resnet50_ssd_512_voc0712_trainval.zip), and extract to `model/` directory.

* Run
```
# cd /path/to/incubator-mxnet/example/ssd
# download the test images
python data/demo/download_demo_images.py
# run the demo
python demo.py --gpu 0
# play with examples:
python demo.py --epoch 0 --images ./data/demo/dog.jpg --thresh 0.5
python demo.py --cpu --network resnet50 --data-shape 512
# wait for library to load for the first time
```
* Check `python demo.py --help` for more options.

### Live Camera detection

Use `init.sh` to download the trained model.
You can use `./demo.py --camera` to use a video capture device with opencv such as a webcam. This
will open a window that will display the camera output together with the detections. You can play
with the detection threshold to get more or less detections.

### Train the model on VOC
* Note that we recommend to use gluon-cv to train the model, please refer to [gluon-cv ssd](https://gluon-cv.mxnet.io/build/examples_detection/train_ssd_voc.html).
This example only covers training on Pascal VOC or MS COCO dataset. Other datasets should
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
* We are going to use `trainval` set in VOC2007/2012 as a common strategy.
The suggested directory structure is to store `VOC2007` and `VOC2012` directories
in the same `VOCdevkit` folder.
* Then link `VOCdevkit` folder to `data/VOCdevkit` by default:
```
ln -s /path/to/VOCdevkit /path/to/incubator-mxnet/example/ssd/data/VOCdevkit
```
Use hard link instead of copy could save us a bit disk space.
* Create packed binary file for faster training:
```
# cd /path/to/incubator-mxnet/example/ssd
bash tools/prepare_pascal.sh
# or if you are using windows
python tools/prepare_dataset.py --dataset pascal --year 2007,2012 --set trainval --target ./data/train.lst
python tools/prepare_dataset.py --dataset pascal --year 2007 --set test --target ./data/val.lst --no-shuffle
```
* Start training:
```
# cd /path/to/incubator-mxnet/example/ssd
python train.py
```
* By default, this example will use `batch-size=32` and `learning_rate=0.002`.
You might need to change the parameters a bit if you have different configurations.
Check `python train.py --help` for more training options. For example, if you have 4 GPUs, use:
```
# note that a perfect training parameter set is yet to be discovered for multi-GPUs
python train.py --gpus 0,1,2,3 --batch-size 32
```

### Train the model on COCO
* Download the COCO2014 dataset, skip this step if you already have one.
```
cd /path/to/where_you_store_datasets/
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
# Extract the data.
unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip
```
* We are going to use `train2014,valminusminival2014` set in COCO2014 for training and `minival2014` for evaluation as a common strategy.
* Then link `COCO2014` folder to `data/coco` by default:
```
ln -s /path/to/COCO2014 /path/to/incubator-mxnet/example/ssd/data/coco
```
Use hard link instead of copy could save us a bit disk space.
* Create packed binary file for faster training:
```
# cd /path/to/incubator-mxnet/example/ssd
bash tools/prepare_coco.sh
# or if you are using windows
python tools/prepare_dataset.py --dataset coco --set train2014,valminusminival2014 --target ./data/train.lst --root ./data/coco
python tools/prepare_dataset.py --dataset coco --set minival2014 --target ./data/val.lst --root ./data/coco --no-shuffle
```
* Start training:
```
# cd /path/to/incubator-mxnet/example/ssd
python train.py --label-width=560 --num-class=80 --class-names=./dataset/names/coco_label --pretrained="" --num-example=117265 --batch-size=64
```

### Evalute trained model
Make sure you have val.rec as validation dataset. It's the same one as used in training. Use:
```
# cd /path/to/incubator-mxnet/example/ssd
python evaluate.py --gpus 0,1 --batch-size 128 --epoch 0

# Evaluate on COCO dataset
python evaluate.py --gpus 0,1 --batch-size 128 --epoch 0 --num-class=80 --class-names=./dataset/names/mscoco.names
```

### Quantize model

To quantize a model on VOC dataset, follow the [Train instructions](https://github.com/apache/incubator-mxnet/tree/master/example/ssd#train-the-model-on-VOC) to train a FP32 `SSD-VGG16_reduced_300x300` model based on Pascal VOC dataset. You can also download our [SSD-VGG16 pre-trained model](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/ssd_vgg16_reduced_300-dd479559.zip) and [packed binary data](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/ssd-val-fc19a535.zip). Create `model` and `data` directories if they're not exist, extract the zip files, then rename the uncompressed files as follows (eg, rename `ssd-val-fc19a535.idx` to `val.idx`, `ssd-val-fc19a535.lst` to `val.lst`, `ssd-val-fc19a535.rec` to `val.rec`, `ssd_vgg16_reduced_300-dd479559.params` to `ssd_vgg16_reduced_300-0000.params`, `ssd_vgg16_reduced_300-symbol-dd479559.json` to `ssd_vgg16_reduced_300-symbol.json`.)

To quantize a model on COCO dataset, follow the [Train instructions](https://github.com/apache/incubator-mxnet/tree/master/example/ssd#train-the-model-on-COCO) to train a FP32 `SSD-VGG16_reduced_300x300` model based on COCO dataset. You can also download our [SSD-VGG16 pre-trained model](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/ssd_vgg16_reduced_300-7fedd4ad.zip) and [packed binary data](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/ssd_coco-val-e91096e8.zip). Create `model` and `data` directories if they're not exist, extract the zip files, then rename the uncompressed files as follows (eg, rename `ssd_coco-val-e91096e8.idx` to `val.idx`, `ssd_coco-val-e91096e8.lst` to `val.lst`, `ssd_coco-val-e91096e8.rec` to `val.rec`, `ssd_vgg16_reduced_300-7fedd4ad.params` to `ssd_vgg16_reduced_300-0000.params`, `ssd_vgg16_reduced_300-symbol-7fedd4ad.json` to `ssd_vgg16_reduced_300-symbol.json`.)

```
data/
|---val.rec
|---val.lxt
|---val.idx
model/
|---ssd_vgg16_reduced_300-0000.params
|---ssd_vgg16_reduced_300-symbol.json
```

Then, use the following command for quantization. By default, this script uses 5 batches (32 samples per batch) for naive calibration:

```
python quantization.py
```

After quantization, INT8 models will be saved in `model/` dictionary.  Use the following command to launch inference.

```

# Launch FP32 Inference on VOC dataset
python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/ssd_

# Launch INT8 Inference on VOC dataset
python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/cqssd_

# Launch FP32 Inference on COCO dataset

python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/ssd_ --num-class=80 --class-names=./dataset/names/mscoco.names

# Launch INT8 Inference on COCO dataset

python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/cqssd_ --num-class=80 --class-names=./dataset/names/mscoco.names

# Launch dummy data Inference
python benchmark_score.py --deploy --prefix=./model/ssd_
python benchmark_score.py --deploy --prefix=./model/cqssd_
```
### Convert model to deploy mode
This simply removes all loss layers, and attach a layer for merging results and non-maximum suppression.
Useful when loading python symbol is not available.
```
# cd /path/to/incubator-mxnet/example/ssd
python deploy.py --num-class 20
```

### Convert caffe model
Converter from caffe is available at `/path/to/incubator-mxnet/example/ssd/tools/caffe_converter`

This is specifically modified to handle custom layer in caffe-ssd. Usage:
```
cd /path/to/incubator-mxnet/example/ssd/tools/caffe_converter
make
python convert_model.py deploy.prototxt name_of_pretrained_caffe_model.caffemodel ssd_converted
# you will use this model in deploy mode without loading from python symbol(layer names inconsistent)
python demo.py --prefix ssd_converted --epoch 1 --deploy
```
There is no guarantee that conversion will always work, but at least it's good for now.

### Legacy models
Since the new interface for composing network is introduced, the old models have inconsistent names for weights.
You can still load the previous model by rename the symbol to `legacy_xxx.py`
and call with `python train/demo.py --network legacy_xxx `
For example:
```
python demo.py --network 'legacy_vgg16_ssd_300.py' --prefix model/ssd_300 --epoch 0
```
