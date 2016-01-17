package ml.dmlc.mxnet

import org.slf4j.{Logger, LoggerFactory}

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
  private def createKVStore(kvStore: String,
                            numDevice: Int,
                            maxSize: Int): (Option[KVStore], Boolean) = {
    if (numDevice == 1 && !kvStore.contains("dist")) {
      // no need to use kv for single device and single machine
      (None, false)
    } else {
      var kvType = kvStore
      if (kvType == "local") {
        // automatically select a proper local
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

  /**
   * Internal training function on multiple devices.
   * This function will also work for single device as well.
   * @param symbol The network configuration
   * @param ctx The training devices.
   * @param argNames Name of all arguments of the network.
   * @param paramNames Name of all trainable parameters of the network.
   * @param auxNames Name of all auxiliary states of the network.
   * @param argParams Model parameter, dict of name to NDArray of net's weights.
   * @param auxParams Model parameter, dict of name to NDArray of net's auxiliary states.
   * @param beginEpoch The begining training epoch.
   * @param endEpoch The end training epoch.
   * @param epochSize Number of batches in a epoch.
   *                  In default, it is set to ceil(num_train_examples / batch_size)
   * @param optimizer The optimization algorithm
   * @param kvstore The KVStore
   * @param updateOnKVStore whether or not perform weight updating on kvstore
   * @param trainData Training data iterator.
   * @param evalData Validation data iterator.
   * @param evalMetric A evaluation function.
   * @param epochEndCallback A callback that is invoked at end of each epoch.
   *                         This can be used to checkpoint model each epoch.
   * @param batchEndCallback A callback that is invoked at end of each batch.
   *                         This can be used to measure speed,
   *                         get result from evaluation metric. etc.
   * @param logger When not specified, default logger will be used.
   * @param workLoadList The list of work load for different devices, in the same order as ctx
   * @param monitor Monitor outputs, weights, and gradients for debugging
   * @note This function will inplace update the NDArrays in argParams and auxStates.
   */
  // scalastyle:off parameterNum
  private def trainMultiDevice(symbol: Symbol, ctx: Seq[Context],
                               argNames: Seq[String], paramNames: Seq[String],
                               auxNames: Seq[String], argParams: Map[String, NDArray],
                               auxParams: Map[String, NDArray],
                               beginEpoch: Int, endEpoch: Int, epochSize: Int,
                               optimizer: Optimizer,
                               kvstore: KVStore, updateOnKVStore: Boolean,
                               trainData: DataIter = null, evalData: DataIter = null,
                               evalMetric: EvalMetric = null,
                               epochEndCallback: EpochEndCallback = null,
                               batchEndCallback: BatchEndCallback = null,
                               logger: Logger = logger,
                               workLoadList: Seq[Float] = Nil,
                               monitor: Monitor = null): Unit = {
    val executorManager = new DataParallelExecutorManager(
        symbol = symbol,
        ctx = ctx,
        trainData = trainData,
        paramNames = paramNames,
        argNames = argNames,
        auxNames = auxNames,
        workLoadList = workLoadList,
        logger = logger)
  }
  // scalastyle:on parameterNum
}

trait EpochEndCallback {
  def invoke(epoch: Int, symbol: Symbol,
             argParams: Map[String, NDArray],
             auxStates: Map[String, NDArray]): Unit
}

trait BatchEndCallback {
  def invoke(epoch: Int, nBatch: Int, evalMetric: EvalMetric)
}

class Model {
}
