Finetune Example
================

Data Preparation
-------------------

Download Caltech101 dataset from <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>, and organize them as the following structure:

```
<root>/images/<cls>/*.jpg
```

Convert images to `.rec` format:

```
python make_list.py images caltech101 --train_ratio=0.8 --recursive=True
python im2rec.py caltech101_train images --resize=256
python im2rec.py caltech101_val images --resize=256
```

Rename and organize `caltech101_train.rec` and `caltech101_val.rec` to:

```
caltech101/train.rec
          /val.rec
```

For this split, there are 7315 training images, and 1829 validation images.

Pre-trained Model Preparation
-----------------------------

Download VGG16 pre-trained model:

```
wget -c https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt
wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
```

Convert caffe model to mxnet model:

```
python convert_model.py VGG_ILSVRC_16_layers_deploy.prototxt VGG_ILSVRC_16_layers.caffemodel vgg16
```

Organize then generates as:

```
vgg16/vgg16-0001.params
     /vgg16-symbol.json
```

Finetune Pre-trained Model on New Dataset
-----------------------------------------

Train Caltech101 with VGG16 from scratch:

```
python train_imagenet.py --data-dir=caltech101 --dataset=caltech101 --num-classes=102 --num-examples=7315 --model-prefix=train_caltech101_vgg16 --network=vgg16 --batch-size=20 --lr=0.0005 --gpus=0 
```

Finetune Caltech101 from pre-trained VGG16 model:

```
python train_imagenet.py --data-dir=caltech101 --dataset=caltech101 --num-classes=102 --num-examples=7315 --model-prefix=finetune_caltech101_vgg16 --finetune-from=vgg16/vgg16-0001 --network=vgg16 --batch-size=20 --lr=0.0005 --gpus=0 
```

Append ` 2>&1 | tee finetune_caltech101_vgg16.log.txt` to keep the log file. The following results are trained on single Tesla K20Xm GPU (Windows).

```
python parse_log.py finetune_caltech101_vgg16.log.txt
| epoch | train-accuracy | valid-accuracy | time |
| --- | --- | --- | --- |
|  1 | 0.543579 | 0.749457 | 552.3 |
|  2 | 0.785929 | 0.857692 | 550.8 |
|  3 | 0.857787 | 0.850543 | 550.3 |
|  4 | 0.895902 | 0.879670 | 550.6 |
|  5 | 0.911339 | 0.878804 | 550.5 |
|  6 | 0.926776 | 0.879670 | 550.1 |
|  7 | 0.943306 | 0.884783 | 550.0 |
|  8 | 0.947951 | 0.904945 | 550.0 |
|  9 | 0.957514 | 0.883152 | 550.3 |
| 10 | 0.955738 | 0.890659 | 550.0 |
| 11 | 0.965027 | 0.892308 | 550.1 |
| 12 | 0.969536 | 0.892391 | 550.0 |
| 13 | 0.969262 | 0.888462 | 550.0 |
| 14 | 0.979781 | 0.902174 | 550.0 |
| 15 | 0.976776 | 0.900000 | 550.3 |
| 16 | 0.971858 | 0.880435 | 550.5 |
| 17 | 0.973087 | 0.896154 | 550.4 |
| 18 | 0.980191 | 0.895652 | 551.0 |
| 19 | 0.984016 | 0.907143 | 551.2 |
| 20 | 0.986202 | 0.895604 | 551.3 |
```



