<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnet2.png width=135/> Deep Learning for Scala/Java
=====

[![Build Status](https://travis-ci.org/dmlc/mxnet.svg?branch=master)](https://travis-ci.org/dmlc/mxnet)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

Here you find the MXNet Scala Package!
It brings flexible and efficient GPU/CPU computing and state-of-art deep learning to JVM.

- It enables you to write seamless tensor/matrix computation with multiple GPUs
  in Scala, Java and other languages built on JVM.
- It also enables you to construct and customize the state-of-art deep learning models in JVM languages,
  and apply them to tasks such as image classification and data science challenges.

Build
------------

Checkout the [Installation Guide](http://mxnet.readthedocs.org/en/latest/build.html) contains instructions to install mxnet.
Then you can compile the Scala Package by

```bash
make scalapkg
```

Run unit/integration tests by

```bash
make scalatest
```

If everything goes well, you will find a jar file named like `mxnet_2.10-osx-x86_64-0.1-SNAPSHOT-full.jar` under `assembly/target`. Then you can use this jar in your own project.

Also `scalapkg` target will build jars for `core` and `example` modules. If you've already downloaded and unpacked MNIST dataset to `./data/`, you can run the training example by

```bash
java -Xmx4m -cp scala-package/assembly/target/*:scala-package/examples/target/mxnet-scala-examples_2.10-0.1-SNAPSHOT.jar:scala-package/examples/target/classes/lib/args4j-2.0.29.jar ml.dmlc.mxnet.examples.imclassification.TrainMnist --data-dir=./data/ --num-epochs=10 --network=mlp --cpus=0,1,2,3
```

Change the arguments and have fun!

Usage
-------
Here is a Scala example of how training a simple 3-layer MLP on MNIST looks like:

```scala
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer.SGD

// model definition
val data = Symbol.Variable("data")
val fc1 = Symbol.FullyConnected(name = "fc1")(Map("data" -> data, "num_hidden" -> 128))
val act1 = Symbol.Activation(name = "relu1")(Map("data" -> fc1, "act_type" -> "relu"))
val fc2 = Symbol.FullyConnected(name = "fc2")(Map("data" -> act1, "num_hidden" -> 64))
val act2 = Symbol.Activation(name = "relu2")(Map("data" -> fc2, "act_type" -> "relu"))
val fc3 = Symbol.FullyConnected(name = "fc3")(Map("data" -> act2, "num_hidden" -> 10))
val mlp = Symbol.SoftmaxOutput(name = "sm")(Map("data" -> fc3))

// load MNIST dataset
val trainDataIter = IO.MNISTIter(Map(
  "image" -> "data/train-images-idx3-ubyte",
  "label" -> "data/train-labels-idx1-ubyte",
  "data_shape" -> "(1, 28, 28)",
  "label_name" -> "sm_label",
  "batch_size" -> batchSize.toString,
  "shuffle" -> "1",
  "flat" -> "0",
  "silent" -> "0",
  "seed" -> "10"))

val valDataIter = IO.MNISTIter(Map(
  "image" -> "data/t10k-images-idx3-ubyte",
  "label" -> "data/t10k-labels-idx1-ubyte",
  "data_shape" -> "(1, 28, 28)",
  "label_name" -> "sm_label",
  "batch_size" -> batchSize.toString,
  "shuffle" -> "1",
  "flat" -> "0", "silent" -> "0"))

// setup model
val model = new FeedForward(mlp, Context.cpu(), numEpoch = 10,
	optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
model.fit(trainDataIter, valDataIter)
```

Predict using the model in the following way:

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
val py = NDArray.argmaxChannel(prob)
require(y.shape == py.shape)

// calculate accuracy
var numCorrect = 0
var numInst = 0
for ((labelElem, predElem) <- y.toArray zip py.toArray) {
  if (labelElem == predElem) {
    numCorrect += 1
  }
  numInst += 1
}
val acc = numCorrect.toFloat / numInst
println(s"Final accuracy = $acc")
```

You can refer to [MXNet Scala Package Examples](https://github.com/javelinjs/mxnet-scala-example)
for more information about how to integrate MXNet Scala Package into your own project.
Currently you have to put the Jars into your project's build classpath manully.
We will provide pre-built binary package on [Maven Repository](http://mvnrepository.com) soon.

License
-------
MXNet Scala Package is licensed under [Apache-2](https://github.com/dmlc/mxnet/blob/master/scala-package/LICENSE) license.
