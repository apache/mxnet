# Single Shot Multi Object Detection using Scala Inference API

In this example, you will learn how to use Scala Inference API to run Inference on pre-trained Single Shot Multi Object Detection (SSD) MXNet model.

The model is trained on the [Pascal VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html). The network is a SSD model built on Resnet50 as base network to extract image features. The model is trained to detect the following entities (classes): ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']. For more details about the model, you can refer to the [MXNet SSD example](https://github.com/apache/incubator-mxnet/tree/master/example/ssd).


## Contents

1. [Prerequisites](#prerequisites)
2. [Download artifacts](#download-artifacts)
3. [Setup datapath and parameters](#setup-datapath-and-parameters)
4. [Run the image inference example](#run-the-image-inference-example)
5. [Infer APIs](#infer-api-details)
6. [Next steps](#next-steps)


## Prerequisites

1. MXNet
2. MXNet Scala Package
3. [IntelliJ IDE (or alternative IDE) project setup](http://mxnet.incubator.apache.org/tutorials/scala/mxnet_scala_on_intellij.html) with the MXNet Scala Package
4. wget


## Setup Guide

### Download Artifacts
#### Step 1
You can download the files using the script `get_ssd_data.sh`. It will download and place the model files in a `model` folder and the test image files in a `image` folder in the current directory.
From the `scala-package/examples/scripts/infer/objectdetector/` folder run:

```bash
./get_ssd_data.sh
```

**Note**: You may need to run `chmod +x get_ssd_data.sh` before running this script.

In the pre-trained model, the `input_name` is `data` and shape is `(1, 3, 512, 512)`.
This shape translates to: a batch of `1` image, the image has color and uses `3` channels (RGB), and the image has the dimensions of `512` pixels in height by `512` pixels in width.

`image/jpeg` is the expected input type, since this example's image pre-processor only supports the handling of binary JPEG images.

The output shape is `(1, 6132, 6)`. As with the input, the `1` is the number of images. `6132` is the number of prediction results, and `6` is for the size of each prediction. Each prediction contains the following components:
- `Class`
- `Accuracy`
- `Xmin`
- `Ymin`
- `Xmax`
- `Ymax`


### Setup Datapath and Parameters
#### Step 2
The followings is the parameters defined for this example, you can find more information in the `class SSDClassifierExample`.

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `model-path-prefix`                   | Folder path with prefix to the model (including json, params, and any synset file). |
| `input-image`                 | The image to run inference on. |
| `input-dir`                   | The directory of images to run inference on. |


## How to Run Inference
After the previous steps, you should be able to run the code using the following script that will pass all of the required parameters to the Infer API.

From the `scala-package/examples/scripts/inferexample/objectdetector/` folder run:

```bash
./run_ssd_example.sh ../models/resnet50_ssd/resnet50_ssd_model ../images/dog.jpg ../images
```

**Notes**:
* These are relative paths to this script.
* You may need to run `chmod +x run_ssd_example.sh` before running this script.

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
the outputs come from the the input image, with top3 predictions picked.


## Infer API Details
This example uses ObjectDetector class provided by MXNet's scala package Infer APIs. It provides methods to load the images, create NDArray out of Java BufferedImage and run prediction using Classifier and Predictor APIs.


## References
This documentation used the model and inference setup guide from the [MXNet Model Server SSD example](https://github.com/awslabs/mxnet-model-server/blob/master/examples/ssd/README.md).


## Next Steps

Check out the following related tutorials and examples for the Infer API:

* [Image Classification with the MXNet Scala Infer API](../imageclassifier/README.md)
