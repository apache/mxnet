# Image Classification

This folder contains example for image classification. The goal of image
classification is to identify the objects contained in images. The following
example shows recognized object classes with corresponding probabilities using a pre-trained
model. This example uses []new Infer APIs](https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer)
provided by MXNet Scala package.

## Contents

1. [Prepare Dataset](#basic-usages)
2. [How to run inference](#prepare-datasets)
3. [A List of pre-trained models](#pre-trained-models)
4. [Infer APIs](#infer-apis-used)


## Prepare Dataset

For this tutorial, you can get the model and sample input image by running following bash file.

  ```bash
  cd incubator-mxnet/scala-package/examples/scripts/inferexample/imageclassifier/
  bash get_resnet_data.sh
  ```

## How to run

  ```bash
  cd incubator-mxnet/scala-package/examples/scripts/inferexample/imageclassifier/
  bash run_classifier_example.sh resnet/resnet-152 images/Cat-hd-wallpapers.jpg images/
  ```

There are few options which you can provide to run the example, one can list them by passing `--help`.
They are also listed as following:

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `model-dir`                   | Folder path with prefix to the model(including json, params any synset file). |
| `input-image`                 | The input image to run inference on. |
| `input-dir`                   | The directory having input images to run inference on. |


## Pre-trained Models

We provide multiple pre-trained models on various datasets. Use
[Python modelzoo.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/common/modelzoo.py)
to download these models. These models can be also be downloaded from [here](http://data.mxnet.io/models/imagenet/).

## Infer APIs used

This example uses [ImageClassifier](https://github.com/apache/incubator-mxnet/blob/master/scala-package/infer/src/main/scala/ml/dmlc/mxnet/infer/ImageClassifier.scala)
class provided by MXNet's scala package Infer APIs.
It provides methods to load the images, create NDArray out of BufferedImage and run prediction
using [Classifier](https://github.com/apache/incubator-mxnet/blob/master/scala-package/infer/src/main/scala/ml/dmlc/mxnet/infer/Classifier.scala)
and [Predictor](https://github.com/apache/incubator-mxnet/blob/master/scala-package/infer/src/main/scala/ml/dmlc/mxnet/infer/Predictor.scala) APIs.
