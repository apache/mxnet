---
layout: page_api
title: Infer API
is_tutorial: true
tag: scala
permalink: /api/scala/docs/tutorials/infer
---
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

# Infer API
The MXNet Scala Infer API provides you with model loading and inference functionality using the MXNet Scala package.


## Prerequisites
To use the Infer API you must first install the MXNet Scala package. Instructions for this are provided in the following variations:
* [Tutorial for setting up a project in the IntelliJ IDE](mxnet_scala_on_intellij)
* [Installing the MXNet Scala Package for macOS]({{'get_started/ubuntu_setup.html#install-the-mxnet-package-for-scala'|relative_url}})
* [Installing the MXNet Scala for Linux]({{'get_started/ubuntu_setup.html#install-the-mxnet-package-for-scala'|relative_url}})

## Inference
The Scala Infer API includes both single image and batch modes. Here is an example of running inference on a single image by using the `ImageClassifier` class. A complete [image classification example](https://github.com/apache/mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier/ImageClassifierExample.scala) using ResNet-152 is provided in the [Scala package's example folder](https://github.com/apache/mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples). This example also demonstrates inference with batches of images.

```scala
def runInferenceOnSingleImage(modelPathPrefix: String, inputImagePath: String,
                              context: Array[Context]):
IndexedSeq[IndexedSeq[(String, Float)]] = {
  val dType = DType.Float32
  val inputShape = Shape(1, 3, 224, 224)

  val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

  // Create object of ImageClassifier class
  val imgClassifier: ImageClassifier = new
      ImageClassifier(modelPathPrefix, inputDescriptor, context)

  // Loading single image from file and getting BufferedImage
  val img = ImageClassifier.loadImageFromFile(inputImagePath)

  // Running inference on single image
  val output = imgClassifier.classifyImage(img, Some(5))

  output
}
```


## Related Resources
* [Infer API Scaladocs]({{'/api/scala/docs/api/#org.apache.mxnet.infer.package'|relative_url}})
* [Single Shot Detector Inference Example](https://github.com/apache/mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/objectdetector)
* [Image Classification Example](https://github.com/apache/mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier)
