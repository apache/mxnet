# Single Shot Multi Object Detection using Scala Inference API

### Disclaimer
This Documentation used the setup guide from [MMS Docs](https://github.com/awslabs/mxnet-model-server/blob/master/examples/ssd/README.md).

## Introduction
In this example, you will learn how to use Scala Inference API to import pre-trained Single Shot Multi Object Detection (SSD) MXNet model.

The pre-trained model is trained on the [Pascal VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html). The network is a SSD model built on Resnet50 as base network to extract image features. The model is trained to detect the following entities (classes): ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']. For more details about the model, you can refer [here](https://github.com/apache/incubator-mxnet/tree/master/example/ssd).

## Setup guide

### Step 1 - Download the pre-trained SSD Model
You can download the files using the script (get_ssd_data.sh). It will download and place the files in the 'data' folder under current directory.

```bash
$ bash get_ssd_data.sh
```

Alternatively use the following links to download the Symbol and Params files via your browser:
- [resnet50_ssd_model-symbol.json](https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json)
- [resnet50_ssd_model-0000.params](https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params)
- [synset.txt](https://github.com/awslabs/mxnet-model-server/blob/master/examples/ssd/synset.txt)

In the pre-trained model, the `input_name` is `data` and shape is `(1, 3, 512, 512)`.
This shape translates to: a batch of `1` image, the image has color and uses `3` channels (RGB), and the image has the dimensions of `512` pixels in height by `512` pixels in width.

The signature also specifies `image/jpeg` as the expected input type, since this example's image pre-processor only supports the handling of binary JPEG images.

The signature specifies the output shape is `(1, 6132, 6)`. As with the input, the `1` is the number of images. `6132` is the number of prediction results, and `6` is for the size of each prediction. Each prediction contains the following components:
- `Class`
- `Accuracy`
- `Xmin`
- `Ymin`
- `Xmax`
- `Ymax`

### Step 2 - Setup Datapath and Parameters

The code `Line 31: val baseDir = System.getProperty("user.dir")` in the example will automatically searches the work directory you have defined. Please put the files in your [work directory](https://stackoverflow.com/questions/16239130/java-user-dir-property-what-exactly-does-it-mean). <!-- how do you define the work directory? -->

Alternatively, if you would like to use your own path, please change line 31 into your own path
```scala
val baseDir = <Your Own Path>
```

The followings is the parameters defined for this example, you can find more information in the `class SSDClassifierExample`.

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `model-dir`                   | Model Folder path |
| `model-prefix`                | prefix to the model(including json, params any synset file).
| `input-image`                 | The input image to run inference on. |
| `input-dir`                   | The directory having input images to run inference on. |

### Step 3 - Run the code
After the previous steps, you should be able to run the code. 
```
$ bash run_ssd_example.sh
```

The example should give expected output as shown below:
```
Class: car
Probabilties: 0.99847263
(Coord:,312.21335,72.0291,456.01443,150.66176)
Class: bicycle
Probabilties: 0.90473825
(Coord:,155.95807,149.96362,383.8369,418.94513)
Class: dog
Probabilties: 0.8226818
(Coord:,83.82353,179.13998,206.63783,476.7875)
```
The first output came from the image `dog.jpg`, with top3 prediction picked.

## Infer APIs used
This example uses ObjectDetector class provided by MXNet's scala package Infer APIs. It provides methods to load the images, create NDArray out of BufferedImage and run prediction using Classifier and Predictor APIs.