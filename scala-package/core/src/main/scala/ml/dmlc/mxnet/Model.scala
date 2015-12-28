package ml.dmlc.mxnet

import org.slf4j.LoggerFactory

/**
 * Describe the model flow
 * @author Yizhi Liu
 */
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
                                paramArrays: Array[Array[NDArray]],
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

  // Perform update of param_arrays from grad_arrays on kvstore
  private def updateParamsOnKVStore(paramArrays: Array[Array[NDArray]],
                                    gradArrays: Array[Array[NDArray]],
                                    kvStore: KVStore): Unit = {
    (paramArrays zip gradArrays).zipWithIndex.foreach { case ((argList, gradList), index) =>
      if (gradList != null) {
        // push gradient, priority is negative index
        kvStore.push(index, gradList, -index)
        // pull back the weights
        kvStore.pull(index, argList, -index)
      }
    }
  }

  // Perform update of param_arrays from grad_arrays not on kvstore
  private def updateParams(paramArrays: Array[Array[NDArray]],
                           gradArrays: Array[Array[NDArray]],
                           updater: MXKVStoreUpdater,
                           numDevice: Int,
                           kvStore: Option[KVStore] = None) {
    (paramArrays zip gradArrays).zipWithIndex.foreach { case ((argList, gradList), index) =>
      if (gradList != null) {
        kvStore.foreach(kv => {
          // push gradient, priority is negative index
          kv.push(index, gradList, -index)
          // pull back the sum gradients, to the same locations.
          kv.pull(index, gradList, -index)
        })
        (argList zip gradList).zipWithIndex.foreach { case ((w: NDArray, g: NDArray), k: Int) =>
          // faked an index here, to make optimizer create diff
          // state for the same index but on diff devs,
          // (copy from python package) TODO(mli) use a better solution latter
          updater.update(index * numDevice + k, g, w)
        }
      }
    }
  }
}

class Model {
}
