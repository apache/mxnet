<!-- This file should be in scala-package/examples/scripts/inferexample or another readme should be there that points to this file. -->
<!-- The SSD example is going into a subfolder objectdetector, so let's be consistent otherwise this readme speaks for all of the inferexample subfolders...-->

# Image Classification

This folder contains an example for image classification with the [MXNet Scala Infer API](https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer).
The goal of image classification is to identify the objects contained in images.
The following example shows recognized object classes with corresponding probabilities using a pre-trained model.


## Contents

1. [Prerequisites](#prerequisites)
2. [Download artifacts](#download-artifacts)
3. [Run the image inference example](#run-the-image-inference-example)
4. [Pre-trained models](#pretrained-models)
5. [Infer APIs](#infer-api-details)
6. [Next steps](#next-steps)

## Prerequisites

1. MXNet
2. MXNet Scala Package
3. [IntelliJ IDE (or alternative IDE) project setup](http://mxnet.incubator.apache.org/tutorials/scala/mxnet_scala_on_intellij.html) with the MXNet Scala Package
4. wget

## Download Artifacts

For this tutorial, you can get the model and sample input image by running following bash file. This script will use `wget` to download these artifacts from AWS S3.

From the `scala-package/examples/scripts/inferexample/imageclassifier/` folder run:

```bash
./get_resnet_data.sh
```

**Note**: you may need to run `chmod +x <name of script file>` before running these scripts.

## Run the Image Inference Example

Now that you have the model files and the test kitten image, you can run the following script to pass the necessary parameters to the JDK to run this inference example.

```bash
./run_predictor_example.sh \
/resnet/resnet-152  /images/kitten.jpg  /images/
```
<!-- it seems weird to use these absolute paths -->

There are few options which you can provide to run the example. Use the `--help` argument to list them.

```bash
./run_predictor_example.sh --help
```

The available arguments are as follows:

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `model-path-prefix`           | Folder path with prefix to the model (including json, params any synset file). |
| `input-image`                 | The image to run inference on. |
| `input-dir`                   | The directory of images to run inference on. |

* You must use `model-dir`.
* You must use **either** `input-image` **or** `input-dir`.
<!-- can you use both? why would you? -->
<!-- does it work with only jpg or do other image formats work? -->
<!-- what are the implications of picking other image formats? where do you fix that? -->

## Pretrained Models

The MXNet project repository provides several [pre-trained models on various datasets](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification#pre-trained-models) and examples on how to train them. You may use the [modelzoo.py](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/common/modelzoo.py) helper script to download these models. Many ImageNet models may be also be downloaded directly from [http://data.mxnet.io/models/imagenet/](http://data.mxnet.io/models/imagenet/).

## Infer API Details

This example uses the [ImageClassifier](https://github.com/apache/incubator-mxnet/blob/master/scala-package/infer/src/main/scala/ml/dmlc/mxnet/infer/ImageClassifier.scala)
class provided by the [MXNet Scala Infer API](https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer).
It provides methods to load the images, create a NDArray out of a `BufferedImage`, and run prediction using the following Infer APIs:
* [Classifier](https://github.com/apache/incubator-mxnet/blob/master/scala-package/infer/src/main/scala/ml/dmlc/mxnet/infer/Classifier.scala)
* [Predictor](https://github.com/apache/incubator-mxnet/blob/master/scala-package/infer/src/main/scala/ml/dmlc/mxnet/infer/Predictor.scala)

## Next Steps

Check out the following related tutorials and examples for the Infer API:

* [Single Shot Detector with the MXNet Scala Infer API](../objectdetector/README.md)
