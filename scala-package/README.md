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
  
Install
------------
 
Technically, all you need is the `mxnet-full_2.10-{arch}-{xpu}-0.1.1.jar` in your classpath.
It will automatically extract the native library to a tempfile and load it.

Currently we provide `linux-x86_64-gpu`, `linux-x86_64-cpu` and `osx-x86_64-cpu`. Support for Windows will come soon.
Use the following dependency in maven, change the artifactId according to your own architecture, e.g., `mxnet-full_2.10-osx-x86_64-cpu` for OSX (and cpu-only).

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-full_2.10-linux-x86_64-gpu</artifactId>
  <version>0.1.1</version>
</dependency>
```

You can also use `mxnet-core_2.10-0.1.1.jar` and put the compiled native library somewhere in your load path.

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-core_2.10</artifactId>
  <version>0.1.1</version>
</dependency>
```

If you have some native libraries conflict with the ones in the provided 'full' jar (e.g., you use openblas instead of atlas), this is a recommended way.
Refer to the next section for how to build it from the very source.

Build
------------

Checkout the [Installation Guide](http://mxnet.io/get_started/setup.html) contains instructions to install mxnet.
Then you can compile the Scala Package by

```bash
make scalapkg
```

(Optional) run unit/integration tests by

```bash
make scalatest
```

Or run a subset of unit tests by, e.g.,

```bash
make SCALA_TEST_ARGS=-Dsuites=ml.dmlc.mxnet.NDArraySuite scalatest
```

If everything goes well, you will find jars for `assembly`, `core` and `example` modules.
Also it produces the native library in `native/{your-architecture}/target`, which you can use to cooperate with the `core` module.

Once you've downloaded and unpacked MNIST dataset to `./data/`, run the training example by

```bash
java -Xmx4G -cp \
  scala-package/assembly/{your-architecture}/target/*:scala-package/examples/target/*:scala-package/examples/target/classes/lib/* \
  ml.dmlc.mxnet.examples.imclassification.TrainMnist \
  --data-dir=./data/ \
  --num-epochs=10 \
  --network=mlp \
  --cpus=0,1,2,3
```

If you've compiled with `USE_DIST_KVSTORE` enabled, the python tools in `mxnet/tracker` can be used to launch distributed training.
The following command runs the above example using 2 worker nodes (and 2 server nodes) in local. Refer to [Distributed Training](http://mxnet.io/how_to/multi_devices.html) for more details.

```bash
tracker/dmlc_local.py -n 2 -s 2 \
  java -Xmx4G -cp \
  scala-package/assembly/{your-architecture}/target/*:scala-package/examples/target/*:scala-package/examples/target/classes/lib/* \
  ml.dmlc.mxnet.examples.imclassification.TrainMnist \
  --data-dir=./data/ \
  --num-epochs=10 \
  --network=mlp \
  --cpus=0 \
  --kv-store=dist_sync
```

Change the arguments and have fun!

Usage
-------
Here is a Scala example of what training a simple 3-layer multilayer perceptron on MNIST looks like. You can download the MNIST dataset using [get_mnist_data script](https://github.com/dmlc/mxnet/blob/master/scala-package/core/scripts/get_mnist_data.sh).

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

// setup model and fit the training data
val model = FeedForward.newBuilder(mlp)
      .setContext(Context.cpu())
      .setNumEpoch(10)
      .setOptimizer(new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
      .setTrainData(trainDataIter)
      .setEvalData(valDataIter)
      .build()
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
val py = NDArray.argmax_channel(prob)
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

Release
-------
- Version 0.1.1, March 24, 2016.
  - Bug fix for MAE & MSE metrics.
- Version 0.1.0, March 22, 2016.

License
-------
MXNet Scala Package is licensed under [Apache-2](https://github.com/dmlc/mxnet/blob/master/scala-package/LICENSE) license.
