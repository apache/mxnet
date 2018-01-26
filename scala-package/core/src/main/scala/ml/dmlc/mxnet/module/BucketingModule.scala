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

import ml.dmlc.mxnet._
import org.slf4j.LoggerFactory
import org.slf4j.Logger
import scala.collection.mutable.ArrayBuffer
import ml.dmlc.mxnet.optimizer.SGD
import scala.collection.immutable.ListMap
import ml.dmlc.mxnet.module.BaseModule._

/**
 * This module helps to deal efficiently with varying-length inputs.
 * @param symGen A function when called with a bucket key, returns a triple
 *              ``(symbol, dataNames, labelNames)``.
 * @param defaultBucketKey The key for the default bucket.
 * @param contexts Default is cpu().
 * @param workLoadList Default `None`, indicating uniform workload.
 * @param fixedParamNames Default `None`, indicating no network parameters are fixed.
 */
class BucketingModule(symGen: AnyRef => (Symbol, IndexedSeq[String], IndexedSeq[String]),
                     defaultBucketKey: AnyRef, contexts: Array[Context] = Context.cpu(),
                     workLoadList: Option[IndexedSeq[Float]] = None,
                     fixedParamNames: Option[Set[String]] = None) extends BaseModule {
  private val logger = LoggerFactory.getLogger(classOf[BucketingModule])

  {
    val (sym, dNames, lNames) = symGen(defaultBucketKey)
    val dataNameList = if (dNames == null) IndexedSeq.empty[String] else dNames
    val labelNameList = if (lNames == null) IndexedSeq.empty[String] else lNames
    val fixedParamNameList = fixedParamNames.getOrElse(IndexedSeq.empty[String]).toIndexedSeq

    _checkInputNames(sym, dataNameList, "data", true, logger)
    _checkInputNames(sym, labelNameList, "label", false, logger)
    _checkInputNames(sym, fixedParamNameList, "fixed_param", true, logger)
  }

  private val workLoads = workLoadList.getOrElse(contexts.map(_ => 1f).toIndexedSeq)
  require(workLoads.size == contexts.length)

  private val _buckets = scala.collection.mutable.Map[AnyRef, Module]()
  private var _currModule: Module = null
  private var _currBucketKey = defaultBucketKey

  private var paramsDirty = false

  // Internal function to reset binded state.
  private def resetBind(): Unit = {
    this.binded = false
    this._buckets.clear()
    this._currModule = null
    this._currBucketKey = defaultBucketKey
  }

  // Symbol information
  // A list of names for data required by this module.
  override def dataNames: IndexedSeq[String] = {
    if (this.binded) this._currModule.dataNames
    else this.symGen(this.defaultBucketKey)._2
  }

  // A list of names for the outputs of this module.
  override def outputNames: IndexedSeq[String] = {
    if (this.binded) this._currModule.outputNames
    else this.symGen(this.defaultBucketKey)._1.listOutputs()
  }

  // Input/Output information
  // A list of (name, shape) pairs specifying the data inputs to this module.
  override def dataShapes: IndexedSeq[DataDesc] = {
    require(this.binded)
    this._currModule.dataShapes
  }

  /**
   * A list of (name, shape) pairs specifying the label inputs to this module.
   * If this module does not accept labels -- either it is a module without loss
   * function, or it is not binded for training, then this should return an empty
   * list `[]`.
   */
  override def labelShapes: IndexedSeq[DataDesc] = {
    require(this.binded)
    this._currModule.labelShapes
  }

  // A list of (name, shape) pairs specifying the outputs of this module.
  override def outputShapes: IndexedSeq[(String, Shape)] = {
    require(this.binded)
    this._currModule.outputShapes
  }

  /**
   * Get current parameters.
   * `(arg_params, aux_params)`, each a dictionary of name to parameters (in
   * `NDArray`) mapping.
   */
  override def getParams: (Map[String, NDArray], Map[String, NDArray]) = {
    require(binded && paramsInitialized)
    this._currModule.paramsDirty = this.paramsDirty
    val params = this._currModule.getParams
    this.paramsDirty = false
    params
  }

  /**
   * Assign parameter and aux state values.
   * @param argParams Dictionary of name to value (`NDArray`) mapping.
   * @param auxParams Dictionary of name to value (`NDArray`) mapping.
   * @param allowMissing
   *         If true, params could contain missing values, and the initializer will be
   *         called to fill those missing params.
   * @param forceInit
   *         If true, will force re-initialize even if already initialized.
   * @param allowExtra
   *         Whether allow extra parameters that are not needed by symbol.
   *         If this is True, no error will be thrown when argParams or auxParams
   *         contain extra parameters that is not needed by the executor.
   */
  override def setParams(argParams: Map[String, NDArray],
                auxParams: Map[String, NDArray],
                allowMissing: Boolean = false,
                forceInit: Boolean = true,
                allowExtra: Boolean = false): Unit = {
    if (!allowMissing) {
      this.initParams(null, argParams, auxParams, allowMissing, forceInit, allowExtra)
    } else if (this.paramsInitialized && !forceInit) {
      logger.warn("Parameters already initialized and forceInit=false. " +
        "setParams call ignored.")
    } else {
      this._currModule.setParams(
        argParams, auxParams, allowMissing, forceInit, allowExtra)

      // because we didn't update self._arg_params, they are dirty now.
      this.paramsDirty = true
      this.paramsInitialized = true
    }
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
   * @param allowExtra Whether allow extra parameters that are not needed by symbol.
   *         If this is True, no error will be thrown when argParams or auxParams
   *         contain extra parameters that is not needed by the executor.
   */
  override def initParams(initializer: Initializer = new Uniform(0.01f),
                          argParams: Map[String, NDArray] = null,
                          auxParams: Map[String, NDArray] = null,
                          allowMissing: Boolean = false,
                          forceInit: Boolean = false,
                          allowExtra: Boolean = false): Unit = {
    if (paramsInitialized && !forceInit) {
      return
    }
    require(binded, "call bind before initializing the parameters")
    this._currModule.initParams(initializer, argParams, auxParams,
      allowMissing, forceInit, allowExtra)
    this.paramsDirty = false
    this.paramsInitialized = true
  }

  /**
   * Bind the symbols to construct executors. This is necessary before one
   * can perform computation with the module.
   * @param dataShapes Typically is `dataIter.provideData`.
   * @param labelShapes Typically is `dataIter.provideLabel`.
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
    // in case we already initialized params, keep it
    val (argParams, auxParams) =
      if (this.paramsInitialized) this.getParams
      else (null, null)

    // force rebinding is typically used when one want to switch from
    // training to prediction phase.
    if (forceRebind) this.resetBind()

    if (this.binded) {
      logger.warn("Already bound, ignoring bind()")
      return
    }

    require(sharedModule == None,
      "shared_module for BucketingModule is not supported")

    this.forTraining = forTraining
    this.inputsNeedGrad = inputsNeedGrad
    this.binded = true

    val (sym, dNames, lNames) = this.symGen(this.defaultBucketKey)
    val module = new Module(sym, dNames, lNames, this.contexts,
      this.workLoadList, this.fixedParamNames)
    module.bind(dataShapes, labelShapes, forTraining, inputsNeedGrad,
      forceRebind = false, sharedModule = None, gradReq)
    this._currModule = module
    this._currBucketKey = this.defaultBucketKey
    this._buckets(this.defaultBucketKey) = module

    // copy back saved params, if already initialized
    if (this.paramsInitialized) {
      this.setParams(argParams, auxParams)
    }
  }

  /**
   * Switches to a different bucket. This will change ``this._currModule``.
   * @param bucketKey The key of the target bucket.
   * @param dataShapes Typically is `dataIter.provideData`.
   * @param labelShapes Typically is `dataIter.provideLabel`.
   */
  def switchBucket(bucketKey: AnyRef, dataShapes: IndexedSeq[DataDesc],
    labelShapes: Option[IndexedSeq[DataDesc]] = None): Unit = {
    require(this.binded, "call bind before switching bucket")
    if (!this._buckets.contains(bucketKey)) {
      val (sym, dNames, lNames) = this.symGen(bucketKey)
      val module = new Module(sym, dNames, lNames, this.contexts,
        this.workLoadList, this.fixedParamNames)
      module.bind(dataShapes, labelShapes, this._currModule.forTraining,
        this._currModule.inputsNeedGrad, forceRebind = false,
        sharedModule = Option(this._buckets(this.defaultBucketKey)))
      this._buckets(bucketKey) = module
    }

    this._currModule = this._buckets(bucketKey)
    this._currBucketKey = bucketKey
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
  override def initOptimizer(kvstore: String = "local", optimizer: Optimizer = new SGD(),
                    resetOptimizer: Boolean = true, forceInit: Boolean = false): Unit = {
    require(binded && paramsInitialized)
    if (optimizerInitialized && !forceInit) {
      logger.warn("optimizer already initialized, ignoring ...")
    } else {
      this._currModule.initOptimizer(kvstore, optimizer, resetOptimizer, forceInit)
      for (mod <- this._buckets.values) {
        if (mod != this._currModule) mod.borrowOptimizer(this._currModule)
      }
      this.optimizerInitialized = true
    }
  }

  /**
   * Prepares a data batch for forward.
   * @param dataBatch input data
   */
  def prepare(dataBatch: DataBatch): Unit = {
    // perform bind if haven't done so
    require(this.binded && this.paramsInitialized)
    val bucketKey = dataBatch.bucketKey
    val originalBucketKey = this._currBucketKey
    this.switchBucket(bucketKey, dataBatch.provideData, Option(dataBatch.provideLabel))
    // switch back
    this.switchBucket(originalBucketKey, null, None)
  }

  /**
   * Forward computation.
   * @param dataBatch input data
   * @param isTrain Default is `None`, which means `is_train` takes the value of `for_training`.
   */
  override def forward(dataBatch: DataBatch, isTrain: Option[Boolean] = None): Unit = {
    require(binded && paramsInitialized)
    this.switchBucket(dataBatch.bucketKey, dataBatch.provideData,
      Option(dataBatch.provideLabel))
    this._currModule.forward(dataBatch, isTrain)
  }

  /**
   * Backward computation.
   * @param outGrads Gradient on the outputs to be propagated back.
   *                 This parameter is only needed when bind is called
   *                 on outputs that are not a loss function.
   */
  override def backward(outGrads: Array[NDArray] = null): Unit = {
    require(binded && paramsInitialized)
    this._currModule.backward(outGrads)
  }

  // Update parameters according to the installed optimizer and the gradients computed
  // in the previous forward-backward cycle.
  override def update(): Unit = {
    require(binded && paramsInitialized && optimizerInitialized)
    this.paramsDirty = true
    this._currModule.update()
  }

  /**
   * Get outputs of the previous forward computation.
   * @return In the case when data-parallelism is used,
   *         the outputs will be collected from multiple devices.
   *         The results will look like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`,
   *         those `NDArray` might live on different devices.
   */
  override def getOutputs(): IndexedSeq[IndexedSeq[NDArray]] = {
    require(binded && paramsInitialized)
    this._currModule.getOutputs()
  }

  /**
   * Get outputs of the previous forward computation.
   * @return In the case when data-parallelism is used,
   *         the outputs will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[out1, out2]`
   */
  override def getOutputsMerged(): IndexedSeq[NDArray] = {
    require(binded && paramsInitialized)
    this._currModule.getOutputsMerged()
  }

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be collected from multiple devices.
   *         The results will look like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`,
   *         those `NDArray` might live on different devices.
   */
  override def getInputGrads(): IndexedSeq[IndexedSeq[NDArray]] = {
    require(binded && paramsInitialized && inputsNeedGrad)
    this._currModule.getInputGrads()
  }

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[grad1, grad2]`
   */
  override def getInputGradsMerged(): IndexedSeq[NDArray] = {
    require(binded && paramsInitialized && inputsNeedGrad)
    this._currModule.getInputGradsMerged()
  }

  /**
   * Evaluate and accumulate evaluation metric on outputs of the last forward computation.
   * @param evalMetric
   * @param labels
   */
  override def updateMetric(evalMetric: EvalMetric, labels: IndexedSeq[NDArray]): Unit = {
    require(binded && paramsInitialized)
    this._currModule.updateMetric(evalMetric, labels)
  }

  override def getSymbol: Symbol = {
    require(binded)
    this._currModule.symbol
  }

  // Install monitor on all executors
  override def installMonitor(monitor: Monitor): Unit = {
    require(binded)
    for (mod <- this._buckets.values) mod.installMonitor(monitor)
  }
}
