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

MXNet Package for Scala/Java

[![Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet/job/master/badge/icon)](http://jenkins.mxnet-ci.amazon-ml.com/job/incubator-mxnet/job/master/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

Here you find the MXNet Scala Package!
It brings flexible and efficient GPU/CPU computing and state-of-art deep learning to JVM.

- It enables you to write seamless tensor/matrix computation with multiple GPUs
  in Scala, Java and other languages built on JVM.
- It also enables you to construct and customize the state-of-art deep learning models in JVM languages,
  and apply them to tasks such as image classification and data science challenges.
- The Scala/Java Inferece APIs provides an easy out of the box solution for loading pre-trained MXNet models and running inference on them.
  
Pre-Built Maven Packages
------------------------

### Stable ###

The MXNet Scala/Java packages can be easily included in your Maven managed project.
The stable jar files for the packages are available on the [MXNet Maven Package Repository](https://search.maven.org/search?q=g:org.apache.mxnet)
Currently we provide packages for Linux (Ubuntu 16.04) (CPU and GPU) and macOS (CPU only). Stable packages for Windows and CentOS will come soon. For now, if you have a CentOS machine, follow the ```Build From Source``` section below. 

To add MXNet Scala/Java package to your project, add the dependency as shown below corresponding to your platform, under the ```dependencies``` tag in your project's ```pom.xml``` :

**Linux GPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux gpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
  <version>[1.3.1,)</version>
</dependency>
```

**Linux CPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-linux cpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
  <version>[1.3.1,)</version>
</dependency>
```

**macOS CPU**

<a href="https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu"><img src="https://img.shields.io/badge/org.apache.mxnet-macOS cpu-green.svg" alt="maven badge"/></a>

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>[1.3.1,)</version>
</dependency>
```

**Note:** ```<version>[1.3.1,)<\version>``` indicates that we will fetch packages with version 1.3.1 or higher. This will always ensure that the pom.xml is able to fetch the latest and greatest jar files from Maven.  

### Nightly ###

Apart from these, the nightly builds representing the bleeding edge development  on Scala/Java packages are also available on the [MXNet Maven Nexus Package Repository](https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~~~). 
Currently we provide nightly packages for Linux (CPU and GPU) and MacOS (CPU only). The Linux nightly jar files also work on CentOS. Nightly packages for Windows will come soon.

Add the following ```repository``` to your project's ```pom.xml``` file : 

Install
------------

Technically, all you need is the `mxnet-full_2.11-{arch}-{xpu}-{version}.jar` in your classpath.
It will automatically extract the native library to a tempfile and load it.
You can find the pre-built jar file in [here](https://search.maven.org/search?q=g:org.apache.mxnet)
 and also our nightly build package [here](https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~)

Currently we provide `linux-x86_64-gpu`, `linux-x86_64-cpu` and `osx-x86_64-cpu`. Support for Windows will come soon.
Use the following dependency in maven, change the artifactId according to your own architecture, e.g., `mxnet-full_2.11-osx-x86_64-cpu` for OSX (and cpu-only).

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.10-linux-x86_64-gpu</artifactId>
  <version>0.1.1</version>
</dependency>
```

You can also use `mxnet-core_2.10-0.1.1.jar` and put the compiled native library somewhere in your load path.

```HTML
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-core_2.10</artifactId>
  <version>0.1.1</version>
</dependency>
```

If you have some native libraries conflict with the ones in the provided 'full' jar (e.g., you use openblas instead of atlas), this is a recommended way.
Refer to the next section for how to build it from the very source.

Build From Source
-----------------

Checkout the [Installation Guide](http://mxnet.incubator.apache.org/install/index.html) contains instructions to install mxnet package and build it from source. Scala maven build assume you already have a ``lib/libmxnet.so`` file.
If you have built MXNet from source and are looking to setup Scala from that point, you may simply run the following from the MXNet source root, Scala build will detect your platform (OSX/Linux) and libmxnet.so flavor (CPU/GPU):

```bash
cd scala-package
mvn install
```

(Optional) run unit/integration tests by

```bash
cd scala-package
mvn integration-test -DskipTests=false
```

Or run a subset of unit tests by, e.g.,

```bash
cd scala-package
mvn -Dsuites=org.apache.mxnet.NDArraySuite integration-test
```

If everything goes well, you will find jars for `assembly`, `core` and `example` modules.
Also it produces the native library in `native/target`, which you can use to cooperate with the `core` module.

Deploy to repository
--------------------

By default, `maven deploy` will deploy artifacts to local file system, you can file then in: ``scala-package/deploy/target/repo`` folder.

For nightly build in CI, a snapshot build will be uploaded to apache repository with follow command:

```bash
cd scala-package
mvn deploy -Pnightly
```

Use following command to deploy release build (push artifacts to apache staging repository):

```bash
cd scala-package
mvn deploy -Pstaging
```

Once you've downloaded and unpacked MNIST dataset to `./data/`, run the training example by

```bash
java -Xmx4G -cp \
  scala-package/assembly/{your-architecture}/target/*:scala-package/examples/target/*:scala-package/examples/target/classes/lib/* \
  org.apache.mxnet.examples.imclassification.TrainMnist \
  --data-dir=./data/ \
  --num-epochs=10 \
  --network=mlp \
  --cpus=0,1,2,3
```

If you've compiled with `USE_DIST_KVSTORE` enabled, the python tools in `mxnet/tracker` can be used to launch distributed training.
The following command runs the above example using 2 worker nodes (and 2 server nodes) in local. Refer to [Distributed Training](http://mxnet.incubator.apache.org/how_to/multi_devices.html) for more details.

```bash
tracker/dmlc_local.py -n 2 -s 2 \
  java -Xmx4G -cp \
  scala-package/assembly/{your-architecture}/target/*:scala-package/examples/target/*:scala-package/examples/target/classes/lib/* \
  org.apache.mxnet.examples.imclassification.TrainMnist \
  --data-dir=./data/ \
  --num-epochs=10 \
  --network=mlp \
  --cpus=0 \
  --kv-store=dist_sync
```

Change the arguments and have fun!

Usage
-------
Here is a Scala example of what training a simple 3-layer multilayer perceptron on MNIST looks like. You can download the MNIST dataset using [get_mnist_data script](https://github.com/apache/incubator-mxnet/blob/master/scala-package/core/scripts/get_mnist_data.sh).

```scala
import org.apache.mxnet._
import org.apache.mxnet.optimizer.SGD

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

Release
-------
- Version 0.1.1, March 24, 2016.
  - Bug fix for MAE & MSE metrics.
- Version 0.1.0, March 22, 2016.

License
-------
MXNet Scala Package is licensed under [Apache-2](https://github.com/apache/incubator-mxnet/blob/master/scala-package/LICENSE) license.

MXNet uses some 3rd party softwares. Following 3rd party license files are bundled inside Scala jar file:
* cub/LICENSE.TXT
* mkldnn/external/mklml_mac_2019.0.1.20180928/license.txt
