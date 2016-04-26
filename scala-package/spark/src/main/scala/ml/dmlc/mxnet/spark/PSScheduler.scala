package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.KVStoreServer

import scala.collection.mutable

/**
 * Start parameter scheduler on spark driver
 * @author Yizhi Liu
 */
object PSScheduler {
  def main(args: Array[String]): Unit = {
    println("Start scheduler in object PSScheduler")
    val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
    envs.put("DMLC_ROLE", "scheduler")
    envs.put("DMLC_PS_ROOT_URI", "127.0.0.1")
    envs.put("DMLC_PS_ROOT_PORT", "9293")
    envs.put("DMLC_NUM_WORKER", args(0))
    envs.put("DMLC_NUM_SERVER", "1")
    println(s"PSScheduler env: $envs")
    KVStoreServer.init(envs.toMap)
    KVStoreServer.start()
    println("Scheduler stop...")
  }
}
class PSScheduler
