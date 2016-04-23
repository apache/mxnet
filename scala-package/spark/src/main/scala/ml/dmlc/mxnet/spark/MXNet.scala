package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.{Optimizer, KVStore, KVStoreServer}
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

    val schedulerThread = new Thread(new Runnable {
      override def run(): Unit = {
        println("Starting scheduler ...")
        PSLauncher.launch("scheduler", spawn = true)
        println("Scheduler started")
      }
    })
    schedulerThread.setDaemon(true)
    schedulerThread.start()


    val input = sc.textFile("/Users/lewis/Workspace/source-codes/forks/mxnet/config.mk")
    println("Partition #: " + input.partitions.length)
    val serverThread = new Thread(new Runnable {
      override def run(): Unit = {
        sc.parallelize(1 to 1, 1).foreachPartition { p =>
          println("PSLauncher launching server ...")
          PSLauncher.launch("server", spawn = true)
        }
      }
    })
    serverThread.start()
    println("server launch thread done")
    val workerThread = new Thread(new Runnable {
      override def run(): Unit = {
        input.foreachPartition { partition =>
          new Thread(new Runnable {
            override def run(): Unit = {
              println("PSLauncher launching worker ...")
              PSLauncher.launch("worker", spawn = false)
            }
          }).start()
          /*
          val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
          envs.put("DMLC_ROLE", "worker")
          envs.put("DMLC_PS_ROOT_URI", "127.0.0.1")
          envs.put("DMLC_PS_ROOT_PORT", "9291")
          envs.put("DMLC_NUM_WORKER", "2")
          envs.put("DMLC_NUM_SERVER", "1")
          KVStoreServer.init(envs.toMap)

          val kv = KVStore.create("dist_sync")
          val optimizer: Optimizer = new SGD(learningRate = 0.01f,
            momentum = 0.9f, wd = 0.00001f)
          println("Set optimizer")
          kv.setOptimizer(optimizer)
          println("PSWorker finished")
          kv.dispose()
          */
        }
      }
    })
    workerThread.start()
    println("worker launch thread done")
    /*
    val sparkJobThread = new Thread() {
      override def run() {
        // force the job
        boosters.foreachPartition(() => _)
      }
    }
    sparkJobThread.start()
    */

    schedulerThread.join()
    serverThread.join()
    workerThread.join()
    sc.stop()
  }
}
