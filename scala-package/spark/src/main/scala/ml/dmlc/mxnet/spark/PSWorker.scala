package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.{KVStore, Optimizer, KVStoreServer}

import scala.collection.mutable

object PSWorker {
  def main(args: Array[String]): Unit = {
    println("Start worker in object PSWorker")
    val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
    envs.put("DMLC_ROLE", "worker")
    envs.put("DMLC_PS_ROOT_URI", "127.0.0.1")
    envs.put("DMLC_PS_ROOT_PORT", "9293")
    envs.put("DMLC_NUM_WORKER", "1")
    envs.put("DMLC_NUM_SERVER", "1")
    KVStoreServer.init(envs.toMap)

    val kv = KVStore.create("dist_async")
    val optimizer: Optimizer = new SGD(learningRate = 0.01f,
      momentum = 0.9f, wd = 0.00001f)
    println("Set optimizer")
    kv.setOptimizer(optimizer)
    Thread.sleep(5000)
    println("PSWorker finished")
    kv.dispose()
  }
}

class PSWorker
