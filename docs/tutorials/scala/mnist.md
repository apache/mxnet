# Handwritten Digit Recognition

This Scala tutorial guides you through a classic computer vision application: identifying hand written digits.

Let's train a 3-layer network (i.e multilayer perceptron network) on the MNIST dataset to classify handwritten digits.

## Prerequisites
To complete this tutorial, we need:

- to compile the latest MXNet version. See the MXNet installation instructions for your operating system in [Setup and Installation](http://mxnet.io/install/index.html).
- to compile the Scala API. See Scala API build instructions in [Build](https://github.com/dmlc/mxnet/tree/master/scala-package).

## Define the Network

First, define the neural network's architecture using the Symbol API:

```scala
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer.SGD

// model definition
val data = Symbol.Variable("data")
val fc1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> 128))
val act1 = Symbol.Activation(name = "relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
val fc2 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> act1, "num_hidden" -> 64))
val act2 = Symbol.Activation(name = "relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
val fc3 = Symbol.FullyConnected(name = "fc3")()(Map("data" -> act2, "num_hidden" -> 10))
val mlp = Symbol.SoftmaxOutput(name = "sm")()(Map("data" -> fc3))
```

## Load the Data

Then, load the training and validation data using DataIterators.

You can download the MNIST data using the [get_mnist_data script](https://github.com/dmlc/mxnet/blob/master/scala-package/core/scripts/get_mnist_data.sh). We've already written a DataIterator for the MNIST dataset:

```scala
// load MNIST dataset
val trainDataIter = IO.MNISTIter(Map(
  "image" -> "data/train-images-idx3-ubyte",
  "label" -> "data/train-labels-idx1-ubyte",
  "data_shape" -> "(1, 28, 28)",
  "label_name" -> "sm_label",
  "batch_size" -> "50",
  "shuffle" -> "1",
  "flat" -> "0",
  "silent" -> "0",
  "seed" -> "10"))

val valDataIter = IO.MNISTIter(Map(
  "image" -> "data/t10k-images-idx3-ubyte",
  "label" -> "data/t10k-labels-idx1-ubyte",
  "data_shape" -> "(1, 28, 28)",
  "label_name" -> "sm_label",
  "batch_size" -> "50",
  "shuffle" -> "1",
  "flat" -> "0", "silent" -> "0"))
```

## Train the model

We can use the FeedForward builder to train our network:

```scala
// setup model and fit the training data
val model = FeedForward.newBuilder(mlp)
      .setContext(Context.cpu())
      .setNumEpoch(10)
      .setOptimizer(new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
      .setTrainData(trainDataIter)
      .setEvalData(valDataIter)
      .build()
```

## Make predictions

Finally, let's make predictions against the validation dataset and compare the predicted labels with the real labels.

```scala
val probArrays = model.predict(valDataIter)
// in this case, we do not have multiple outputs
require(probArrays.length == 1)
val prob = probArrays(0)

// get real labels
import scala.collection.mutable.ListBuffer
valDataIter.reset()
val labels = ListBuffer.empty[NDArray]
while (valDataIter.hasNext) {
  val evalData = valDataIter.next()
  labels += evalData.label(0).copy()
}
val y = NDArray.concatenate(labels)

// get predicted labels
val predictedY = NDArray.argmax_channel(prob)
require(y.shape == predictedY.shape)

// calculate accuracy
var numCorrect = 0
var numTotal = 0
for ((labelElem, predElem) <- y.toArray zip predictedY.toArray) {
  if (labelElem == predElem) {
    numCorrect += 1
  }
  numTotal += 1
}
val acc = numCorrect.toFloat / numTotal
println(s"Final accuracy = $acc")
```

Check out more MXNet Scala examples below.

## Next Steps
* [Scala API](http://mxnet.io/api/scala/)
* [More Scala Examples](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/)
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
