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

# Image Classification

This folder contains an example for image classification with the [MXNet Scala Infer API](https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer).
The goal of image classification is to identify the objects contained in images.
The following example shows recognized object classes with corresponding probabilities using a pre-trained model.


## Contents

1. [Prerequisites](#prerequisites)
2. [Download artifacts](#download-artifacts)
3. [Run the image inference example](#run-the-image-inference-example)
4. [Pretrained models](#pretrained-models)
5. [Infer APIs](#infer-api-details)
6. [Next steps](#next-steps)


## Prerequisites

1. MXNet
2. MXNet Scala Package
3. [IntelliJ IDE (or alternative IDE) project setup](http://mxnet.incubator.apache.org/tutorials/scala/mxnet_scala_on_intellij.html) with the MXNet Scala Package
4. wget


## Download Artifacts

For this tutorial, you can get the model and sample input image by running following bash file. This script will use `wget` to download these artifacts from AWS S3.

From the `scala-package/examples/scripts/infer/imageclassifier/` folder run:

```bash
./get_resnet_data.sh
```

**Note**: You may need to run `chmod +x get_resnet_data.sh` before running this script.


## Run the Image Inference Example

Now that you have the model files and the test kitten image, you can run the following script to pass the necessary parameters to the JDK to run this inference example.

```bash
./run_classifier_example.sh \
../resnet/resnet-152  ../images/kitten.jpg  ../images/
```

**Notes**:
* These are relative paths to this script.
* You may need to run `chmod +x run_predictor_example.sh` before running this script.

There are few options which you can provide to run the example. Use the `--help` argument to list them.

```bash
./run_predictor_example.sh --help
```

The available arguments are as follows:

| Argument                      | Comments                                 |
| ----------------------------- | ---------------------------------------- |
| `model-dir`                   | Folder path with prefix to the model (including json, params, and any synset file). |
| `input-image`                 | The image to run inference on. |
| `input-dir`                   | The directory of images to run inference on. |

* You must use `model-dir`.
* You must use `input-image` and `input-dir` as this example shows single image inference as well as batch inference together.


## Pretrained Models

The MXNet project repository provides several [pre-trained models on various datasets](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification#pre-trained-models) and examples on how to train them. You may use the [modelzoo.py](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/common/modelzoo.py) helper script to download these models. Many ImageNet models may be also be downloaded directly from [http://data.mxnet.io/models/imagenet/](http://data.mxnet.io/models/imagenet/).


## Infer API Details

This example uses the [ImageClassifier](https://github.com/apache/incubator-mxnet/blob/master/scala-package/infer/src/main/scala/org/apache/mxnet/infer/ImageClassifier.scala)
class provided by the [MXNet Scala Infer API](https://github.com/apache/incubator-mxnet/tree/master/scala-package/infer).
It provides methods to load the images, create a NDArray out of a `BufferedImage`, and run prediction using the following Infer APIs:
* [Classifier](https://github.com/apache/incubator-mxnet/blob/master/scala-package/infer/src/main/scala/org/apache/mxnet/infer/Classifier.scala)
* [Predictor](https://github.com/apache/incubator-mxnet/blob/master/scala-package/infer/src/main/scala/org/apache/mxnet/infer/Predictor.scala)


## Next Steps

Check out the following related tutorials and examples for the Infer API:

* [Single Shot Detector with the MXNet Scala Infer API](../objectdetector/README.md)
