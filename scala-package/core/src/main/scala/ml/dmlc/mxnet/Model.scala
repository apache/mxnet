/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.mxnet

import java.nio.ByteBuffer

import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
 * Describe the model flow
 */
class Model
object Model {
  private val logger = LoggerFactory.getLogger(classOf[Model])

  /**
   * Checkpoint the model data into file.
   * @param prefix Prefix of model name.
   * @param epoch The epoch number of the model.
   * @param symbol The input symbol
   * @param argParams Model parameter, dict of name to NDArray of net's weights.
   * @param auxParams Model parameter, dict of name to NDArray of net's auxiliary states.
   * @note
   * - ``prefix-symbol.json`` will be saved for symbol.
   * - ``prefix-epoch.params`` will be saved for parameters.
   */
  def saveCheckpoint(prefix: String, epoch: Int, symbol: Symbol,
                     argParams: Map[String, NDArray], auxParams: Map[String, NDArray]): Unit = {
    symbol.save(s"$prefix-symbol.json")
    val saveDict = argParams.map { case (k, v) => s"arg:$k" -> v } ++
      auxParams.map { case (k, v) => s"aux:$k" -> v }
    val paramName = "%s-%04d.params".format(prefix, epoch)
    NDArray.save(paramName, saveDict)
    logger.info(s"Saved checkpoint to $paramName")
  }

  /**
   * Load model checkpoint from file.
   *
   * @param prefix Prefix of model name.
   * @param epoch Epoch number of model we would like to load.
   *
   * @return
   * symbol : The symbol configuration of computation network.
   * argParams : Model parameter, dict of name to NDArray of net's weights.
   * auxParams : Model parameter, dict of name to NDArray of net's auxiliary states.
   * @note
   * - symbol will be loaded from ``prefix-symbol.json``.
   * - parameters will be loaded from ``prefix-epoch.params``.
   */
  def loadCheckpoint(prefix: String, epoch: Int):
    (Symbol, Map[String, NDArray], Map[String, NDArray]) = {
    val symbol = Symbol.load(s"$prefix-symbol.json")
    val saveDict = NDArray.load("%s-%04d.params".format(prefix, epoch))
    val argParams = mutable.HashMap[String, NDArray]()
    val auxParams = mutable.HashMap[String, NDArray]()
    for ((k, v) <- saveDict._1 zip saveDict._2) {
      val splitted = k.split(":", 2)
      val tp = splitted(0)
      val name = splitted(1)
      if (tp == "arg") {
        argParams(name) = v
      } else if (tp == "aux") {
        auxParams(name) = v
      }
    }
    (symbol, argParams.toMap, auxParams.toMap)
  }

  // a helper class for serializing model
  class SerializedModel private[mxnet] (
    val symbol: String,
    val argParams: Map[String, Array[Byte]],
    val auxParams: Map[String, Array[Byte]]) extends Serializable

  private[mxnet] def serialize(symbol: Symbol,
                               argParams: Map[String, NDArray],
                               auxParams: Map[String, NDArray]): Array[Byte] = {
    val serializedModel = new SerializedModel(
      symbol.toJson,
      argParams.map { case (k, v) => (k, v.serialize()) },
      auxParams.map { case (k, v) => (k, v.serialize()) }
    )
    Serializer.getSerializer.serialize(serializedModel).array()
  }

  private[mxnet] def deserialize(bytes: Array[Byte]):
    (Symbol, Map[String, NDArray], Map[String, NDArray]) = {
    val model = Serializer.getSerializer.deserialize[SerializedModel](ByteBuffer.wrap(bytes))
    val symbol = Symbol.loadJson(model.symbol)
    val argParams = model.argParams.map { case (k, v) =>
      (k, NDArray.deserialize(v))
    }
    val auxParams = model.auxParams.map { case (k, v) =>
      (k, NDArray.deserialize(v))
    }
    (symbol, argParams, auxParams)
  }

  /**
   * Create kvstore
   * This function select and create a proper kvstore given the kvstore type
   * @param kvStore KVStore type
   * @param numDevice The number of devices
   * @param argParams Model parameter, dict of name to NDArray of net's weights.
   * @return Option of created [[KVStore]] and whether or not update weight on it
   */
  private[mxnet] def createKVStore(kvStore: String,
                                   numDevice: Int,
                                   argParams: Map[String, NDArray]): (Option[KVStore], Boolean) = {
    if (numDevice == 1 && !kvStore.contains("dist")) {
      // no need to use kv for single device and single machine
      (None, false)
    } else {
      var kvType = kvStore
      if (kvType == "local") {
        // automatically select a proper local
        val maxSize = argParams.values.map(_.shape.product).max
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
   * Create a kvStore (wrap it with Option, None if given kvStore == null)
   * @param kvStore KVStore
   * @return Option of created [[KVStore]] and whether or not update weight on it
   */
  private[mxnet] def createKVStore(kvStore: KVStore): (Option[KVStore], Boolean) = {
    (Option(kvStore), kvStore != null && !kvStore.`type`.contains("local_allreduce"))
  }

  // Initialize kvstore
  private[mxnet] def initializeKVStore(kvStore: KVStore,
                                       paramArrays: IndexedSeq[Array[NDArray]],
                                       argParams: Map[String, NDArray],
                                       paramNames: IndexedSeq[String],
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
  private[mxnet] def updateParamsOnKVStore(paramArrays: IndexedSeq[Array[NDArray]],
                                           gradArrays: IndexedSeq[Array[NDArray]],
                                           kvStore: Option[KVStore]): Unit = {
    (paramArrays zip gradArrays).zipWithIndex.foreach { case ((argList, gradList), index) =>
      if (gradList != null) {
        // push gradient, priority is negative index
        kvStore.foreach(_.push(index, gradList, -index))
        // pull back the weights
        kvStore.foreach(_.pull(index, argList, -index))
      }
    }
  }

  // Perform update of param_arrays from grad_arrays not on kvstore
  private[mxnet] def updateParams(paramArrays: IndexedSeq[Array[NDArray]],
                                  gradArrays: IndexedSeq[Array[NDArray]],
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
   * @param kvStore The KVStore
   * @param updateOnKVStore whether or not perform weight updating on kvstore
   * @param trainData Training data iterator.
   * @param evalData Validation data iterator.
   * @param evalMetric A evaluation function.
   * @param epochEndCallback A callback that is invoked at end of each epoch.
   *                         This can be used to checkpoint model each epoch.
   * @param batchEndCallback A callback that is invoked at end of each batch.
   *                         This can be used to measure speed,
   *                         get result from evaluation metric. etc.
   * @param workLoadList The list of work load for different devices, in the same order as ctx
   * @param monitor Monitor outputs, weights, and gradients for debugging
   * @note This function will inplace update the NDArrays in argParams and auxStates.
   */
  // scalastyle:off parameterNum
  private[mxnet] def trainMultiDevice(symbol: Symbol, ctx: Array[Context],
                                      argNames: IndexedSeq[String], paramNames: IndexedSeq[String],
                                      auxNames: IndexedSeq[String], argParams: Map[String, NDArray],
                                      auxParams: Map[String, NDArray],
                                      beginEpoch: Int, endEpoch: Int, epochSize: Int,
                                      optimizer: Optimizer,
                                      kvStore: Option[KVStore], updateOnKVStore: Boolean,
                                      trainData: DataIter,
                                      evalData: Option[DataIter] = None,
                                      evalMetric: EvalMetric,
                                      epochEndCallback: Option[EpochEndCallback] = None,
                                      batchEndCallback: Option[BatchEndCallback] = None,
                                      workLoadList: Seq[Float] = Nil,
                                      monitor: Option[Monitor] = None,
                                      symGen: SymbolGenerator = null): Unit = {
    val executorManager = new DataParallelExecutorManager(
        symbol = symbol,
        symGen = symGen,
        ctx = ctx,
        trainData = trainData,
        paramNames = paramNames,
        argNames = argNames,
        auxNames = auxNames,
        workLoadList = workLoadList)

    monitor.foreach(executorManager.installMonitor)
    executorManager.setParams(argParams, auxParams)

    // updater for updateOnKVStore = false
    val updaterLocal = Optimizer.getUpdater(optimizer)

    kvStore.foreach(initializeKVStore(_, executorManager.paramArrays,
      argParams, executorManager.paramNames, updateOnKVStore))
    if (updateOnKVStore) {
      kvStore.foreach(_.setOptimizer(optimizer))
    }

    // Now start training
    for (epoch <- beginEpoch until endEpoch) {
      // Training phase
      val tic = System.currentTimeMillis
      evalMetric.reset()
      var nBatch = 0
      var epochDone = false
      // Iterate over training data.
      trainData.reset()
      while (!epochDone) {
        var doReset = true
        while (doReset && trainData.hasNext) {
          val dataBatch = trainData.next()
          executorManager.loadDataBatch(dataBatch)
          monitor.foreach(_.tic())
          executorManager.forward(isTrain = true)
          executorManager.backward()
          if (updateOnKVStore) {
            updateParamsOnKVStore(executorManager.paramArrays,
              executorManager.gradArrays,
              kvStore)
          } else {
            updateParams(executorManager.paramArrays,
              executorManager.gradArrays,
              updaterLocal, ctx.length,
              kvStore)
          }
          monitor.foreach(_.tocPrint())
          // evaluate at end, so out_cpu_array can lazy copy
          executorManager.updateMetric(evalMetric, dataBatch.label)

          nBatch += 1
          batchEndCallback.foreach(_.invoke(epoch, nBatch, evalMetric))

          // this epoch is done possibly earlier
          if (epochSize != -1 && nBatch >= epochSize) {
            doReset = false
          }
        }
        if (doReset) {
          trainData.reset()
        }

        // this epoch is done
        epochDone = (epochSize == -1 || nBatch >= epochSize)
      }

      val (name, value) = evalMetric.get
      name.zip(value).foreach { case (n, v) =>
        logger.info(s"Epoch[$epoch] Train-$n=$v")
      }
      val toc = System.currentTimeMillis
      logger.info(s"Epoch[$epoch] Time cost=${toc - tic}")

      evalData.foreach { evalDataIter =>
        evalMetric.reset()
        evalDataIter.reset()
        // TODO: make DataIter implement Iterator
        while (evalDataIter.hasNext) {
          val evalBatch = evalDataIter.next()
          executorManager.loadDataBatch(evalBatch)
          executorManager.forward(isTrain = false)
          executorManager.updateMetric(evalMetric, evalBatch.label)
        }

        val (name, value) = evalMetric.get
        name.zip(value).foreach { case (n, v) =>
          logger.info(s"Epoch[$epoch] Train-$n=$v")
        }
      }

      if (epochEndCallback.isDefined || epoch + 1 == endEpoch) {
        executorManager.copyTo(argParams, auxParams)
      }
      epochEndCallback.foreach(_.invoke(epoch, symbol, argParams, auxParams))
    }

    updaterLocal.dispose()
    executorManager.dispose()
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
