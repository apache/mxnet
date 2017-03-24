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

package ml.dmlc.mxnet.module

import java.io.{FileInputStream, BufferedInputStream, BufferedOutputStream, FileOutputStream}
import ml.dmlc.mxnet.DType.DType
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.module.DataParallelExecutorGroup.Builder
import ml.dmlc.mxnet.optimizer.SGD
import org.slf4j.LoggerFactory

/**
 * Module is a basic module that wrap a `Symbol`. It is functionally the same
 * as the `FeedForward` model, except under the module API.
 * @param symbolVar : Symbol definition.
 * @param dataNames Input data names.
 * @param labelNames Input label names
 * @param contexts Default is cpu().
 * @param workLoadList  Default `None`, indicating uniform workload.
 * @param fixedParamNames Default `None`, indicating no network parameters are fixed.
 */
class Module(symbolVar: Symbol,
             val dataNames: IndexedSeq[String] = IndexedSeq("data"),
             labelNames: IndexedSeq[String] = IndexedSeq("softmax_label"),
             contexts: Array[Context] = Context.cpu(),
             workLoadList: Option[IndexedSeq[Float]] = None,
             fixedParamNames: Option[Set[String]] = None) extends BaseModule {
  private val logger = LoggerFactory.getLogger(classOf[Module])

  require(symbolVar != null)
  this.symbol = symbolVar

  private val workLoads = workLoadList.getOrElse(contexts.map(_ => 1f).toIndexedSeq)
  require(workLoads.size == contexts.length)

  private val labelNameList = if (labelNames == null) IndexedSeq.empty[String] else labelNames

  private val argNames = symbol.listArguments()
  private val inputNames = dataNames ++ labelNameList
  private val paramNames = argNames.filterNot(inputNames.toSet)
  private val auxNames = symbol.listAuxiliaryStates()
  private val outputNamesVar = symbol.listOutputs()

  private var paramsDirty = false

  private var optimizer: Optimizer = null
  private var kvstore: Option[KVStore] = None
  private var updateOnKVStore: Boolean = false
  private var updater: Option[MXKVStoreUpdater] = None
  private var preloadOptStates: Option[String] = None

  private var dataShapesVar: IndexedSeq[DataDesc] = null
  private var labelShapesVar: Option[IndexedSeq[DataDesc]] = None

  override def dataShapes: IndexedSeq[DataDesc] = {
    require(binded)
    dataShapesVar
  }

  override def labelShapes: IndexedSeq[DataDesc] = {
    require(binded)
    labelShapesVar.orNull
  }

  override def outputShapes: IndexedSeq[(String, Shape)] = {
    require(binded)
    execGroup.getOutputShapes
  }

  def outputNames: IndexedSeq[String] = outputNamesVar

  /**
   * Get current parameters.
   * `(arg_params, aux_params)`, each a dictionary of name to parameters (in
   * `NDArray`) mapping.
   */
  override def getParams: (Map[String, NDArray], Map[String, NDArray]) = {
    require(binded && paramsInitialized)
    if (paramsDirty) {
      syncParamsFromDevices()
    }
    (argParams, auxParams)
  }

  /**
   * Initialize the parameters and auxiliary states.
   * @param initializer Called to initialize parameters if needed.
   * @param argParams If not None, should be a dictionary of existing arg_params.
   *                  Initialization will be copied from that.
   * @param auxParams If not None, should be a dictionary of existing aux_params.
   *                  Initialization will be copied from that.
   * @param allowMissing If true, params could contain missing values,
   *                     and the initializer will be called to fill those missing params.
   * @param forceInit If true, will force re-initialize even if already initialized.
   */
  override def initParams(initializer: Initializer = new Uniform(0.01f),
                          argParams: Map[String, NDArray] = null,
                          auxParams: Map[String, NDArray] = null,
                          allowMissing: Boolean = false, forceInit: Boolean = false): Unit = {
    if (paramsInitialized && !forceInit) {
      return
    }
    require(binded, "call bind before initializing the parameters")

    if (this.argParams == null) {
      val paramArrays =
        execGroup.paramArrays.map(nds => NDArray.zeros(nds(0).shape, dtype = nds(0).dtype))
      this.argParams = this.paramNames.zip(paramArrays).toMap
    }

    if (this.auxParams == null) {
      val auxArrays =
        execGroup.auxArrays.map(nds => NDArray.zeros(nds(0).shape, dtype = nds(0).dtype))
      this.auxParams = this.auxNames.zip(auxArrays).toMap
    }

    this.argParams.foreach { case (name, arr) =>
      impl(name, arr, allowMissing, Option(initializer), argParams)
    }

    this.auxParams.foreach { case (name, arr) =>
      impl(name, arr, allowMissing, Option(initializer), auxParams)
    }

    this.paramsInitialized = true
    this.paramsDirty = false

    // copy the initialized parameters to devices
    this.execGroup.setParams(this.argParams, this.auxParams)
  }

  // Internal helper for parameter initialization
  private def impl(name: String, arr: NDArray, allowMissing: Boolean,
                   initializer: Option[Initializer] = None,
                   cache: Map[String, NDArray] = null): Unit = {
    if (cache != null) {
      if (cache.contains(name)) {
        val cacheArr = cache(name) // just in case the cached array is just the target itself
        if (cacheArr ne arr) {
          cacheArr.copyTo(arr)
        }
      } else {
        require(allowMissing, s"$name is not presented")
        initializer.foreach(inst => inst(name, arr))
      }
    } else {
      initializer.foreach(inst => inst(name, arr))
    }
  }

  // Internal function to reset binded state.
  private def resetBind(): Unit = {
    binded = false
    execGroup = null
    dataShapesVar = null
    labelShapesVar = None
  }

  /**
   * Bind the symbols to construct executors. This is necessary before one
   * can perform computation with the module.
   * @param dataShapes Typically is `dataIter.provideData`.
   * @param labelShapes Typically is `data_iter.provide_label`.
   * @param forTraining Default is `true`. Whether the executors should be bind for training.
   * @param inputsNeedGrad Default is `false`.
   *                       Whether the gradients to the input data need to be computed.
   *                       Typically this is not needed.
   *                       But this might be needed when implementing composition of modules.
   * @param forceRebind Default is `false`.
   *                    This function does nothing if the executors are already binded.
   *                    But with this `true`, the executors will be forced to rebind.
   * @param sharedModule Default is `None`. This is used in bucketing.
   *                     When not `None`, the shared module essentially corresponds to
   *                     a different bucket -- a module with different symbol
   *                     but with the same sets of parameters
   *                     (e.g. unrolled RNNs with different lengths).
   */
  override def bind(dataShapes: IndexedSeq[DataDesc],
                    labelShapes: Option[IndexedSeq[DataDesc]] = None,
                    forTraining: Boolean = true, inputsNeedGrad: Boolean = false,
                    forceRebind: Boolean = false, sharedModule: Option[BaseModule] = None,
                    gradReq: String = "write"): Unit = {
    // force rebinding is typically used when one want to switch from training to prediction phase.
    if (forceRebind) {
      resetBind()
    }

    if (binded) {
      logger.warn("Already binded, ignoring bind()")
      return
    }

    this.forTraining = forTraining
    this.inputsNeedGrad = inputsNeedGrad
    this.binded = true

    if (!forTraining) {
      require(!inputsNeedGrad)
    } else {
      // this is not True, as some module might not contains a loss function
      // that consumes the labels
      // require(labelShapes != None)
    }

    this.dataShapesVar = dataShapes
    this.labelShapesVar = labelShapes

    val sharedGroup =
      sharedModule.map(sharedModuleInst => {
        require(sharedModuleInst.binded && sharedModuleInst.paramsInitialized)
        sharedModuleInst.execGroup
      })

    val inputTypes = this.dataShapesVar.map(dataDesc => (dataDesc.name, dataDesc.dtype)).toMap ++
      labelShapes.map(shapes => shapes.map(dataDesc => (dataDesc.name, dataDesc.dtype)).toMap)
                 .getOrElse(Map.empty[String, DType])

    execGroup = new Builder(symbol, contexts, paramNames)
      .setWorkLoadList(workLoads)
      .setDataShapes(dataShapes)
      .setLabelShapes(labelShapes.orNull)
      .setForTraining(forTraining)
      .setInputsNeedGrad(inputsNeedGrad)
      .setSharedGroup(sharedGroup.orNull)
      .setFixedParamNames(fixedParamNames.orNull)
      .setGradReq(gradReq)
      .setInputTypes(inputTypes)
      .build()

    if (sharedModule.isDefined) {
      paramsInitialized = true
      argParams = sharedModule.get.argParams
      auxParams = sharedModule.get.auxParams
    } else if (paramsInitialized) {
      // if the parameters are already initialized, we are re-binding
      // so automatically copy the already initialized params
      execGroup.setParams(argParams, auxParams)
    }

    sharedModule.foreach {
      case sharedModuleInst: Module =>
        if (sharedModuleInst.optimizerInitialized) {
          borrowOptimizer(sharedModuleInst)
        }
      case _ =>
    }
  }

  /**
   * Install and initialize optimizers.
   * @param kvstore
   * @param optimizer
   * @param resetOptimizer Default `True`, indicating whether we should set `rescaleGrad`
   *                       & `idx2name` for optimizer according to executorGroup
   * @param forceInit Default `False`, indicating whether we should force re-initializing
   *                  the optimizer in the case an optimizer is already installed.
   */
  def initOptimizer(kvstore: String = "local", optimizer: Optimizer = new SGD(),
                    resetOptimizer: Boolean = true, forceInit: Boolean = false): Unit = {
    require(binded && paramsInitialized)
    if (optimizerInitialized && !forceInit) {
      logger.warn("optimizer already initialized, ignoring ...")
    } else {
      val (kvstoreInst, updateOnKVStore) = Model.createKVStore(kvstore, contexts.length, argParams)
      val batchSize = execGroup.getBatchSize * (
        if (kvstoreInst != None && kvstoreInst.get.`type` == "dist_sync") {
          kvstoreInst.get.numWorkers
        } else {
          1
        })
      if (resetOptimizer) {
        val idx2name =
          if (updateOnKVStore) {
            execGroup.paramNames.zipWithIndex.map { case (name, i) => (i, name) }.toMap
          } else {
            (0 until contexts.length).flatMap(k =>
              execGroup.paramNames.zipWithIndex.map { case (name, i) =>
                (i * contexts.length + k, name)
              }
            ).toMap
          }
        optimizer.setIdx2Name(idx2name)
        optimizer.setRescaleGrad(1f / batchSize)
      }

      this.optimizer = optimizer
      this.kvstore = kvstoreInst
      this.updateOnKVStore = updateOnKVStore

      kvstoreInst.foreach(kv =>
        // copy initialized local parameters to kvstore
        Model.initializeKVStore(kv, execGroup.paramArrays,
          argParams, paramNames, updateOnKVStore)
      )
      updater =
        if (updateOnKVStore) {
          kvstoreInst.foreach(_.setOptimizer(this.optimizer))
          None
        } else {
          Some(Optimizer.getUpdater(optimizer))
        }

      optimizerInitialized = true
      preloadOptStates.foreach { optStates =>
        loadOptimizerStates(optStates)
      }
      preloadOptStates = None
    }
  }

  /**
   * Borrow optimizer from a shared module. Used in bucketing, where exactly the same
   * optimizer (esp. kvstore) is used.
   * @param sharedModule
   */
  def borrowOptimizer(sharedModule: Module): Unit = {
    require(sharedModule.optimizerInitialized)
    optimizer = sharedModule.optimizer
    kvstore = sharedModule.kvstore
    updateOnKVStore = sharedModule.updateOnKVStore
    updater = sharedModule.updater
    optimizerInitialized = true
  }

  /**
   * Forward computation.
   * @param dataBatch input data
   * @param isTrain Default is `None`, which means `is_train` takes the value of `for_training`.
   */
  def forward(dataBatch: DataBatch, isTrain: Option[Boolean] = None): Unit = {
    require(binded && paramsInitialized)
    execGroup.forward(dataBatch, isTrain)
  }

  /**
   * Backward computation.
   * @param outGrads Gradient on the outputs to be propagated back.
   *                 This parameter is only needed when bind is called
   *                 on outputs that are not a loss function.
   */
  def backward(outGrads: Array[NDArray] = null): Unit = {
    require(binded && paramsInitialized)
    execGroup.backward(outGrads)
  }

  // Update parameters according to the installed optimizer and the gradients computed
  // in the previous forward-backward batch.
  def update(): Unit = {
    require(binded && paramsInitialized && optimizerInitialized)
    paramsDirty = true
    if (updateOnKVStore) {
      Model.updateParamsOnKVStore(execGroup.paramArrays,
        execGroup.gradArrays, kvstore)
    } else {
      require(updater != None)
      Model.updateParams(execGroup.paramArrays,
        execGroup.gradArrays, updater.orNull, contexts.length, kvstore)
    }
  }

  /**
   * Get outputs of the previous forward computation.
   * @return In the case when data-parallelism is used,
   *         the outputs will be collected from multiple devices.
   *         The results will look like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`,
   *         those `NDArray` might live on different devices.
   */
  def getOutputs(): IndexedSeq[IndexedSeq[NDArray]] = {
    require(binded && paramsInitialized)
    execGroup.getOutputs()
  }

  /**
   * Get outputs of the previous forward computation.
   * @return In the case when data-parallelism is used,
   *         the outputs will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[out1, out2]`
   */
  def getOutputsMerged(): IndexedSeq[NDArray] = {
    require(binded && paramsInitialized)
    execGroup.getOutputsMerged()
  }

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be collected from multiple devices.
   *         The results will look like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`,
   *         those `NDArray` might live on different devices.
   */
  def getInputGrads(): IndexedSeq[IndexedSeq[NDArray]] = {
    require(binded && paramsInitialized && inputsNeedGrad)
    execGroup.getInputGrads()
  }

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[grad1, grad2]`
   */
  def getInputGradsMerged(): IndexedSeq[NDArray] = {
    require(binded && paramsInitialized && inputsNeedGrad)
    execGroup.getInputGradsMerged()
  }

  /**
   * Evaluate and accumulate evaluation metric on outputs of the last forward computation.
   * @param evalMetric
   * @param labels
   */
  def updateMetric(evalMetric: EvalMetric, labels: IndexedSeq[NDArray]): Unit = {
    execGroup.updateMetric(evalMetric, labels)
  }

  // Synchronize parameters from devices to CPU. This function should be called after
  // calling `update` that updates the parameters on the devices, before one can read the
  // latest parameters from `self._arg_params` and `self._aux_params`.
  private def syncParamsFromDevices(): Unit = {
    execGroup.getParams(argParams, auxParams)
  }

  // Install monitor on all executors
  def installMonitor(monitor: Monitor): Unit = {
    require(binded)
    execGroup.installMonitor(monitor)
  }

  /**
   * Save optimizer (updater) state to file
   * @param fname Path to output states file.
   */
  def saveOptimizerStates(fname: String): Unit = {
    require(optimizerInitialized, "Optimizer should be initialized before saving.")
    if (updateOnKVStore) {
      kvstore.foreach(_.saveOptimizerStates(fname))
    } else {
      updater.foreach {
        case cachedStates: MXKVStoreCachedStates =>
          val target = new BufferedOutputStream(new FileOutputStream(fname))
          try {
            target.write(cachedStates.serializeState())
          } finally {
            target.close()
          }
        case _ =>
          logger.warn("Updater does not have states, skip saving to {}", fname)
      }
    }
  }

  /**
   * Load optimizer (updater) state from file
   * @param fname Path to input states file.
   */
  def loadOptimizerStates(fname: String): Unit = {
    require(optimizerInitialized, "Optimizer should be initialized before loading.")
    if (updateOnKVStore) {
      kvstore.foreach(_.loadOptimizerStates(fname))
    } else {
      updater.foreach {
        case cachedStates: MXKVStoreCachedStates =>
          val bis = new BufferedInputStream(new FileInputStream(fname))
          try {
            val bArray = Stream.continually(bis.read).takeWhile(-1 !=).map(_.toByte).toArray
            cachedStates.deserializeState(bArray)
          } finally {
            bis.close()
          }
        case _ =>
          logger.warn("Updater does not have states, skip loading from {}", fname)
      }
    }
  }

  /**
   * Save current progress to checkpoint.
   * Use mx.callback.module_checkpoint as epoch_end_callback to save during training.
   * @param prefix The file prefix to checkpoint to
   * @param epoch The current epoch number
   * @param saveOptStates Whether to save optimizer states for continue training
   */
  def saveCheckpoint(prefix: String, epoch: Int, saveOptStates: Boolean = false): Unit = {
    symbol.save(s"$prefix-symbol.json")
    val paramName = "%s-%04d.params".format(prefix, epoch)
    saveParams(paramName)
    logger.info("Saved checkpoint to {}", paramName)
    if (saveOptStates) {
      val stateName = "%s-%04d.states".format(prefix, epoch)
      saveOptimizerStates(stateName)
      logger.info("Saved optimizer state to {}", stateName)
    }
  }
}

object Module {
  /**
   * Create a model from previously saved checkpoint.
   * @param prefix Path prefix of saved model files. You should have "prefix-symbol.json",
   *               "prefix-xxxx.params", and optionally "prefix-xxxx.states",
   *               where xxxx is the epoch number.
   * @param epoch Epoch to load.
   * @param loadOptimizerStates Whether to load optimizer states.
   *                            Checkpoint needs to have been made with saveOptimizerStates=True
   * @param dataNames Input data names.
   * @param labelNames Input label names
   * @param contexts Default is cpu().
   * @param workLoadList  Default `None`, indicating uniform workload.
   * @param fixedParamNames Default `None`, indicating no network parameters are fixed.
   */
  def loadCheckpoint(prefix: String, epoch: Int, loadOptimizerStates: Boolean = false,
                     dataNames: IndexedSeq[String] = IndexedSeq("data"),
                     labelNames: IndexedSeq[String] = IndexedSeq("softmax_label"),
                     contexts: Array[Context] = Context.cpu(),
                     workLoadList: Option[IndexedSeq[Float]] = None,
                     fixedParamNames: Option[Set[String]] = None): Module = {
    val (sym, args, auxs) = Model.loadCheckpoint(prefix, epoch)
    val mod = new Module(symbolVar = sym,
      dataNames, labelNames, contexts, workLoadList, fixedParamNames)
    mod.argParams = args
    mod.auxParams = auxs
    mod.paramsInitialized = true
    if (loadOptimizerStates) {
      mod.preloadOptStates = Some("%s-%04d.states".format(prefix, epoch))
    }
    mod
  }
}
