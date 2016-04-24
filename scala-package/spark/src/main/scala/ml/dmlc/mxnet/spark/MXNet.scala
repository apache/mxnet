package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.spark.io.LabeledPointIter
import ml.dmlc.mxnet.{Optimizer, KVStore, KVStoreServer}
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
    val conf = new SparkConf().setAppName("MXNet")
    val sc = new SparkContext(conf)

    println("Starting scheduler ...")
    PSLauncher.launch("scheduler", spawn = true)

    sc.parallelize(1 to 1, 1).foreachPartition { p =>
      println("PSLauncher launching server ...")
      PSLauncher.launch("server", spawn = true)
    }

    val trainRaw = sc.textFile("/Users/lewis/Workspace/source-codes/forks/mxnet/data/spark/val.txt")
    val trainData = trainRaw.map { s =>
      val parts = s.split(' ')
      val label = java.lang.Double.parseDouble(parts(0))
      val features = Vectors.dense(parts(1).trim().split(',').map(java.lang.Double.parseDouble))
      LabeledPoint(label, features)
    }.repartition(1)
    val dimension = trainData.first().features.size
    println(s"Dimension: $dimension")

    println("Partition #: " + trainData.partitions.length)
    trainData.foreachPartition { partition =>
      val dataIter = new LabeledPointIter(partition, dimension, 100)

      println("PSLauncher launching worker ...")
      //PSLauncher.launch("worker", spawn = false)
      val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
      envs.put("DMLC_ROLE", "worker")
      envs.put("DMLC_PS_ROOT_URI", "127.0.0.1")
      envs.put("DMLC_PS_ROOT_PORT", "9293")
      envs.put("DMLC_NUM_WORKER", "1")
      envs.put("DMLC_NUM_SERVER", "1")
      KVStoreServer.init(envs.toMap)

      val kv = KVStore.create("dist_sync")
      val optimizer: Optimizer = new SGD(learningRate = 0.01f,
        momentum = 0.9f, wd = 0.00001f)
      println("Set optimizer")
      kv.setOptimizer(optimizer)

      while (dataIter.hasNext) {
        val dataBatch = dataIter.next()
        println(s"Data: ${dataBatch.label.head.toArray.mkString(",")}, " +
          s"dim = ${dataBatch.data.head.shape}")
      }

      println("PSWorker finished")
      kv.dispose()
    }
    sc.stop()
  }
}
