# SegNet

SegNet is a deep encoder-decoder architecture for multi-class pixelwise segmentation researched and developed by members of the [Computer Vision and Robotics Group](http://mi.eng.cam.ac.uk/Main/CVR) at the University of Cambridge, UK.

This mxnet version reference a caffe version https://github.com/alexgkendall/SegNet-Tutorial .

The segnet_basic and segnet networks were included in this mxnet version.

## Requirement

Need python package Pillow.
```
pip install Pillow
```
Build MXNet with new pooling and upsampling operators.
```
# copy new operators to src/operator/
cp segnet/op/* incubator-mxnet/src/operator/
# rebuild MXNet from source
cd incubator-mxnet/
make
cd python/
python setup.py install
```

## Dataset

The model can be trained for road scene understanding using the [CamVid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/). The Cambridge-driving Labeled Video Database (CamVid) is the first collection of videos with object class semantic labels, complete with metadata. The database provides ground truth labels that associate each pixel with one of [32 semantic classes](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/#ClassLabels).

This dataset is small, consisting of 367 training and 233 testing RGB images (day and dusk scenes) at
360*480 resolution. The challenge is to segment 11 classes such
as road, building, cars, pedestrians, signs, poles, side-walk etc.

## Train

- Getting Started

  Install python package `Pillow` (required by `image_segment.py`).

```
[sudo] pip install Pillow
```

- train model
```
python train_segnet.py \
--gpus=0,1,2,3 \
--lr=0.005 \
--wd=0.0005 \
--network=segnet_basic \
--batch-size=12
```

​	The output log may look like this
```
Epoch[132] Batch [5]    Speed: 14.15 samples/sec        accuracy=0.903786
Epoch[132] Batch [10]   Speed: 14.80 samples/sec        accuracy=0.913140
Epoch[132] Batch [15]   Speed: 13.62 samples/sec        accuracy=0.909260
Epoch[132] Batch [20]   Speed: 14.09 samples/sec        accuracy=0.893901
Epoch[132] Batch [25]   Speed: 14.74 samples/sec        accuracy=0.911765
Epoch[132] Train-accuracy=0.906672
Epoch[132] Time cost=25.005
Epoch[132] Validation-accuracy=0.819186

```
- Using the pre-trained model for image segmentation

  It load parameters from model_prefix and load_epoch as normal.

  We can also load convolution parameters from vgg16 model by use model_prefix_for_vgg16 and load_epoch_for_vgg16.
- The accuracy can reach 0.84 with segnet_basic. When train segnet with vgg pre-trained parameters, the accuracy will reach about 0.89 .  

## Score

- score the model

```
python score.py --batch-size=1 --load_epoch=270 --score_file=test.txt
```

The output log may look like this. Each line shows the global segment accuracy in one picture. The segment result can be saved in a package named "res_pic".

```
INFO:root:('accuracy', 0.80057751076185835)
INFO:root:('accuracy', 0.80030805232087987)
INFO:root:('accuracy', 0.80030805232087987)
INFO:root:('accuracy', 0.80014792169887283)
INFO:root:('accuracy', 0.80014792169887283)
INFO:root:('accuracy', 0.80037548100048095)
INFO:root:('accuracy', 0.80037548100048095)

```

## Segment one pic

```
python segment_one_pic.py \
--image_name=0016E5_04530.png \
--label_name=0016E5_04530.png \
--load_epoch=270 \
```

The program will output segment result and label in a package named "res_pic".

The left one is a label picture with 11 classes. The right one below is a segment result.

![label2](https://user-images.githubusercontent.com/13029886/32312590-9120270e-bfd9-11e7-9fdb-de29aece2422.png) ![res2](https://user-images.githubusercontent.com/13029886/32312591-9159ea7a-bfd9-11e7-8658-e3fa1ce90bf2.png)
