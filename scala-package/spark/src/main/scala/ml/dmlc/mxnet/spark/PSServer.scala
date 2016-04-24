package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.KVStoreServer

import scala.collection.mutable

object PSServer {
  def main(args: Array[String]): Unit = {
    println("Start server in object PSServer")
    val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
    envs.put("DMLC_ROLE", "server")
    envs.put("DMLC_PS_ROOT_URI", "127.0.0.1")
    envs.put("DMLC_PS_ROOT_PORT", "9293")
    envs.put("DMLC_NUM_WORKER", "1")
    envs.put("DMLC_NUM_SERVER", "1")
    KVStoreServer.init(envs.toMap)
    KVStoreServer.start()
    println("Server stop...")
  }
}

class PSServer
