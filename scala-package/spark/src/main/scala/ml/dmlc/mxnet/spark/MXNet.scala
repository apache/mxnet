package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.Callback.Speedometer
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.spark.io.LabeledPointIter
import ml.dmlc.mxnet._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
 * MXNet Training On Spark
 * @author Yizhi Liu
 */
object MXNet {
  def train(data: RDD[LabeledPoint]): Unit = {
    data.foreachPartition { partition =>
    }
  }

  def main(args: Array[String]): Unit = {
    val numWorker = args(1).toInt
    println(s"numWorker: $numWorker")
    val classpath = args(2)

    val conf = new SparkConf().setAppName("MXNet")
    val sc = new SparkContext(conf)

    val trainRaw = sc.textFile(args(0))
    val trainData = trainRaw.map { s =>
      val parts = s.split(' ')
      val label = java.lang.Double.parseDouble(parts(0))
      val features = Vectors.dense(parts(1).trim().split(',').map(java.lang.Double.parseDouble))
      LabeledPoint(label, features)
    }.repartition(numWorker)

    require(trainData.getNumPartitions == numWorker)

    val batchSize = 128
    val dimension = trainData.first().features.size
    println(s"Dimension: $dimension")
    val numExamples = trainData.count().toInt
    println(s"numExamples: $numExamples")

    println("Starting scheduler ...")
    val tmp = "/mnt/hgfs/lewis/Workspace/source-codes/forks/mxnet/scala-package/assembly/linux-x86_64-cpu/target/*" +
      ":/mnt/hgfs/lewis/Workspace/source-codes/forks/mxnet/scala-package/spark/target/classes/lib/*" +
      ":/mnt/hgfs/lewis/Workspace/source-codes/forks/mxnet/scala-package/spark/target/*"
    //PSLauncher.launch("scheduler", numWorker = numWorker, spawn = true, classpath)
    val scheduler = new PSScheduler(tmp, "127.0.0.1", 9293, numServer = 1, numWorker = numWorker)
    require(scheduler.startProcess(), "Failed to start ps scheduler process")

    sc.parallelize(1 to 1, 1).foreachPartition { p =>
      println("PSLauncher launching server ...")
      PSLauncher.launch("server", numWorker = numWorker, spawn = true, classpath)
    }

    val job = trainData.mapPartitions { partition =>
      val dataIter = new LabeledPointIter(
        partition, dimension, batchSize, labelName = "softmax_label")

      println("PSLauncher launching worker ...")
      //PSLauncher.launch("worker", spawn = false)
      val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
      envs.put("DMLC_ROLE", "worker")
      envs.put("DMLC_PS_ROOT_URI", "127.0.0.1")
      envs.put("DMLC_PS_ROOT_PORT", "9293")
      envs.put("DMLC_NUM_WORKER", numWorker.toString)
      envs.put("DMLC_NUM_SERVER", "1")
      KVStoreServer.init(envs.toMap)

      val kv = KVStore.create("dist_async")
      val optimizer: Optimizer = new SGD(learningRate = 0.01f,
        momentum = 0.9f, wd = 0.00001f)
      //println("Set optimizer")
      //kv.setOptimizer(optimizer)

      println("Define model")
      val model = new FeedForward(ctx = Context.cpu(),
        symbol = getMlp,
        numEpoch = 10,
        optimizer = optimizer,
        initializer = new Xavier(factorType = "in", magnitude = 2.34f),
        argParams = null,
        auxParams = null,
        beginEpoch = 0,
        epochSize = numExamples / batchSize / kv.numWorkers)
      println("Start fit")
      model.fit(trainData = dataIter,
        evalData = null,
        evalMetric = new Accuracy(),
        kvStore = kv)

      /*
      while (dataIter.hasNext) {
        val dataBatch = dataIter.next()
        println(s"Data: ${dataBatch.label.head.toArray.mkString(",")}, " +
          s"dim = ${dataBatch.data.head.shape}")
      }
      */

      println("PSWorker finished")
      //kv.dispose()
      Iterator(new MXNetModel(model))
    }.cache()

    job.foreachPartition(() => _)
    //Thread.sleep(60000) // one minute

    sc.stop()
  }

  def getMlp: Symbol = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")(Map("data" -> data, "num_hidden" -> 128))
    val act1 = Symbol.Activation(name = "relu1")(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")(Map("data" -> act1, "num_hidden" -> 64))
    val act2 = Symbol.Activation(name = "relu2")(Map("data" -> fc2, "act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected(name = "fc3")(Map("data" -> act2, "num_hidden" -> 10))
    val mlp = Symbol.SoftmaxOutput(name = "softmax")(Map("data" -> fc3))
    mlp
  }
}
