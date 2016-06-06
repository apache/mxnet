<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/mxnet2.png width=135/> Deep Learning on Spark
=====

Here comes the MXNet on [Spark](http://spark.apache.org/).
It is built on the MXNet Scala Package and brings deep learning to Spark. 

Now you have an end-to-end solution for large-scale deep models, which means you can take advantage of both the flexible parallel training approaches and GPU support with MXNet, and the fast data processing flow with Spark, to build the full pipeline from raw data to efficient deep learning.

The MXNet on Spark is still in *experimental stage*. Any suggestion or contribution will be highly appreciated.
  
Build
------------

Checkout the [Installation Guide](http://mxnet.readthedocs.org/en/latest/how_to/build.html) contains instructions to install mxnet. Remember to enable the distributed training, i.e., set `USE_DIST_KVSTORE = 1`.

Compile the Scala Package by

```bash
make scalapkg
```

This will automatically build the `spark` submodule. Now you can submit Spark job with these built jars.

You can find a piece of submit script in the `bin` directory of the `spark` module. Remember to set variables and versions according to your own environment.

Usage
------------
Here is a Spark job example of how training a deep network looks like.

First define the parameters for the training procedure,

```scala
val conf = new SparkConf().setAppName("MXNet")
val sc = new SparkContext(conf)

val mxnet = new MXNet()
  .setBatchSize(128)
  .setLabelName("softmax_label")
  .setContext(Context.cpu()) // or GPU if you like
  .setDimension(Shape(784))
  .setNetwork(network) // e.g. MLP model
  .setNumEpoch(10)
  .setNumServer(2)
  .setNumWorker(4)
  // These jars are required by the KVStores at runtime.
  // They will be uploaded and distributed to each node automatically
  .setExecutorJars(cmdLine.jars)
```

Now load data and do distributed training,

```scala
val trainData = parseRawData(sc, cmdLine.input)
val model = mxnet.fit(trainData)
```

In this example, it will start PS scheduler on driver and launch 2 servers. The input data will be split into 4 pieces and train with  `dist_async` mode.

To save the output model, simply call `save` method,

```scala
model.save(sc, cmdLine.output + "/model")
```

Predicting is straightforward,

```scala
val valData = parseRawData(sc, cmdLine.inputVal)
val brModel = sc.broadcast(model)
val res = valData.mapPartitions { data =>
  val probArrays = brModel.value.predict(points.toIterator)
  require(probArrays.length == 1)
  val prob = probArrays(0)
  val py = NDArray.argmaxChannel(prob.get)
  val labels = py.toArray.mkString(",")
  py.dispose()
  prob.get.dispose()
  labels
}
res.saveAsTextFile(cmdLine.output + "/data")
```

Pitfalls
------------

- Sometime you have to specify the `java` argument, to help MXNet find the right java binary on worker nodes.
- MXNet and [ps-lite](https://github.com/dmlc/ps-lite) currently do NOT support multiple instances in one process, (we will fix this issue in the future, but with lower priority.) thus you must run Spark job in cluster mode (standalone, yarn-client, yarn-cluster). Local mode is NOT supported because it runs tasks in multiples threads with one process, which will block the initialization of KVStore.
(Hint: If you only have one physical node and want to test the Spark package, you can start N workers on one node by setting `export SPARK_WORKER_INSTANCES=N` in `spark-env.sh`.)
Also, remember to set `--executor-cores 1` to ensure there's only one task run in one Spark executor.
- Fault tolerance is not fully supported. If some of your tasks fail, please restart the whole application. We will solve it soon.
