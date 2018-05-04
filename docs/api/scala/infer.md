# Infer API
The MXNet Scala Infer API provides you with model loading and inference functionality using the MXNet Scala package.


## Prerequisites
To use the Infer API you must first install the MXNet Scala package. Instructions for this are provided in the following variations:
* [Tutorial for setting up a project in the IntelliJ IDE](../../tutorials/scala/mxnet_scala_on_intellij.html)
* [Installing the MXNet Scala Package for macOS](../../install/ubuntu_setup.html#install-the-mxnet-package-for-scala)
* [Installing the MXNet Scala for Linux](../../install/ubuntu_setup.html#install-the-mxnet-package-for-scala)


## Inference
The Scala Infer API includes both single image and batch modes. Here is an example of running inference on a single image by using the `ImageClassifier` class. A complete [image classification example](https://github.com/apache/incubator-mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier/ImageClassifierExample.scala) using ResNet-152 is provided in the [Scala package's example folder](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples). This example also demonstrates inference with batches of images.

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
* [Infer API Scaladocs](docs/index.html#org.apache.mxnet.infer.package)
* [Single Shot Detector Inference Example](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/objectdetector)
* [Image Classification Example](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier)
