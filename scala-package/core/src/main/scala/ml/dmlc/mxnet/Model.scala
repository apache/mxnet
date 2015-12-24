package ml.dmlc.mxnet

import org.slf4j.LoggerFactory

object Model {
  private val logger = LoggerFactory.getLogger(classOf[Model])
  /**
   * Create kvstore
   * This function select and create a proper kvstore given the kvstore type
   * @param kvStore KVStore type
   * @param numDevice The number of devices
   * @param maxSize max size of the kvstore
   * @return Option of created [[KVStore]] and whether or not update weight on it
   */
  private def createKVStore(kvStore: String, numDevice: Int, maxSize: Int): (Option[KVStore], Boolean) = {
    if (numDevice == 1 && !kvStore.contains("dist")) {
      // no need to use kv for single device and single machine
      (None, false)
    } else {
      var kvType = kvStore
      if (kvType == "local") {
        //automatically select a proper local
        kvType =
          if (maxSize < 1024 * 1024 * 16) {
            "local_update_cpu"
          } else {
            "local_allreduce_cpu"
          }
        logger.info(s"Auto - select kvstore type = $kvType")
      }
      (Option(KVStore.create(kvType)), !kvType.contains("local_allreduce"))
    }
  }

  /**
   * Create a kvstore (wrap it with Option, None if given kvStore == null)
   * @param kvStore
   * @return Option of created [[KVStore]] and whether or not update weight on it
   */
  private def createKVStore(kvStore: KVStore): (Option[KVStore], Boolean) = {
    (Option(kvStore), kvStore != null && !kvStore.`type`.contains("local_allreduce"))
  }

  // Initialize kvstore
  private def initializeKVStore(kvStore: KVStore,
                                paramArrays: Array[NDArray],
                                argParams: Map[String, NDArray],
                                paramNames: Array[String],
                                updateOnKVStore: Boolean): Unit = {
    require(paramArrays.length == paramNames.length)
    for (idx <- 0 until paramArrays.length) {
      val paramOnDevs = paramArrays(idx)
      kvStore.init(idx, argParams(paramNames(idx)))
      if (updateOnKVStore) {
        kvStore.pull(idx, paramOnDevs, -idx)
      }
    }
  }
}

class Model {
}
