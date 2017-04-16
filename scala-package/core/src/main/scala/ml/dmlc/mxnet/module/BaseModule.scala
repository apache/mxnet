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

import java.io.IOException

import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet._
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

/**
 * The base class of a modules. A module represents a computation component. The design
 * purpose of a module is that it abstract a computation "machine", that one can run forward,
 * backward, update parameters, etc. We aim to make the APIs easy to use, especially in the
 * case when we need to use imperative API to work with multiple modules (e.g. stochastic
 * depth network).
 *
 * A module has several states:
 *
 * - Initial state. Memory is not allocated yet, not ready for computation yet.
 * - Binded. Shapes for inputs, outputs, and parameters are all known, memory allocated,
 *   ready for computation.
 * - Parameter initialized. For modules with parameters, doing computation before initializing
 *   the parameters might result in undefined outputs.
 * - Optimizer installed. An optimizer can be installed to a module. After this, the parameters
 *   of the module can be updated according to the optimizer after gradients are computed
 *   (forward-backward).
 *
 *  In order for a module to interactive with others, a module should be able to report the
 *  following information in its raw stage (before binded)
 *
 *  - `data_names`: list of string indicating the names of required data.
 *  - `output_names`: list of string indicating the names of required outputs.
 *
 *  And also the following richer information after binded:
 *
 *  - state information
 *    - `binded`: `bool`, indicating whether the memory buffers needed for computation
 *      has been allocated.
 *    - `forTraining`: whether the module is binded for training (if binded).
 *    - `paramsInitialized`: `bool`, indicating whether the parameters of this modules
 *      has been initialized.
 *    - `optimizerInitialized`: `bool`, indicating whether an optimizer is defined
 *      and initialized.
 *    - `inputsNeedGrad`: `bool`, indicating whether gradients with respect to the
 *      input data is needed. Might be useful when implementing composition of modules.
 *
 *  - input/output information
 *    - `dataShapes`: a list of `(name, shape)`. In theory, since the memory is allocated,
 *      we could directly provide the data arrays. But in the case of data parallelization,
 *      the data arrays might not be of the same shape as viewed from the external world.
 *    - `labelShapes`: a list of `(name, shape)`. This might be `[]` if the module does
 *      not need labels (e.g. it does not contains a loss function at the top), or a module
 *      is not binded for training.
 *    - `outputShapes`: a list of `(name, shape)` for outputs of the module.
 *
 *  - parameters (for modules with parameters)
 *    - `getParams()`: return a tuple `(argParams, auxParams)`. Each of those
 *      is a dictionary of name to `NDArray` mapping. Those `NDArray` always lives on
 *      CPU. The actual parameters used for computing might live on other devices (GPUs),
 *      this function will retrieve (a copy of) the latest parameters. Therefore, modifying
 *    - `setParams(argParams, auxParams)`: assign parameters to the devices
 *      doing the computation.
 *    - `initParams(...)`: a more flexible interface to assign or initialize the parameters.
 *
 *  - setup
 *    - `bind()`: prepare environment for computation.
 *    - `initOptimizer()`: install optimizer for parameter updating.
 *
 *  - computation
 *    - `forward(dataBatch)`: forward operation.
 *    - `backward(outGrads=None)`: backward operation.
 *    - `update()`: update parameters according to installed optimizer.
 *    - `getOutputs()`: get outputs of the previous forward operation.
 *    - `getInputGrads()`: get the gradients with respect to the inputs computed
 *      in the previous backward operation.
 *    - `updateMetric(metric, labels)`: update performance metric for the previous forward
 *      computed results.
 *
 *  - other properties (mostly for backward compatibility)
 *    - `symbol`: the underlying symbolic graph for this module (if any)
 *      This property is not necessarily constant. For example, for `BucketingModule`,
 *      this property is simply the *current* symbol being used. For other modules,
 *      this value might not be well defined.
 *
 *  When those intermediate-level API are implemented properly, the following
 *  high-level API will be automatically available for a module:
 *
 *  - `fit`: train the module parameters on a data set
 *  - `predict`: run prediction on a data set and collect outputs
 *  - `score`: run prediction on a data set and evaluate performance
 */
abstract class BaseModule {
  private val logger = LoggerFactory.getLogger(classOf[BaseModule])

  private[module] var binded: Boolean = false
  private[module] var forTraining: Boolean = false
  private[module] var inputsNeedGrad: Boolean = false
  private[module] var paramsInitialized: Boolean = false
  private[module] var optimizerInitialized: Boolean = false
  private[module] var symbol: Symbol = null
  private[module] var execGroup: DataParallelExecutorGroup = null
  private[module] var argParams: Map[String, NDArray] = null
  private[module] var auxParams: Map[String, NDArray] = null

  // High Level API

  // A convenient function that calls both `forward` and `backward`.
  def forwardBackward(dataBatch: DataBatch): Unit = {
    forward(dataBatch, isTrain = Option(true))
    backward()
  }

  /**
   * Run prediction on `eval_data` and evaluate the performance according to `eval_metric`.
   * @param evalData : DataIter
   * @param evalMetric : EvalMetric
   * @param numBatch Number of batches to run. Default is `Integer.MAX_VALUE`,
   *                 indicating run until the `DataIter` finishes.
   * @param batchEndCallback Could also be a list of functions.
   * @param reset Default `True`,
   *              indicating whether we should reset `eval_data` before starting evaluating.
   * @param epoch Default 0. For compatibility, this will be passed to callbacks (if any).
   *              During training, this will correspond to the training epoch number.
   */
  def score(evalData: DataIter, evalMetric: EvalMetric,
            numBatch: Int = Integer.MAX_VALUE,
            batchEndCallback: Option[BatchEndCallback] = None,
            scoreEndCallback: Option[BatchEndCallback] = None,
            reset: Boolean = true, epoch: Int = 0): EvalMetric = {
    require(evalData != null && evalMetric != null)
    require(binded && paramsInitialized)

    if (reset) {
      evalData.reset()
    }

    evalMetric.reset()

    var nBatch = 0
    while (evalData.hasNext && nBatch < numBatch) {
      val evalBatch = evalData.next()

      forward(evalBatch, isTrain = Option(false))
      updateMetric(evalMetric, evalBatch.label)

      batchEndCallback.foreach(callback => {
        callback.invoke(epoch, nBatch, evalMetric)
      })
      nBatch += 1
    }

    scoreEndCallback.foreach(callback => {
      callback.invoke(epoch, nBatch, evalMetric)
    })

    evalMetric
  }

  /**
   * Run prediction and collect the outputs.
   * @param evalData
   * @param numBatch Default is -1, indicating running all the batches in the data iterator.
   * @param reset Default is `True`, indicating whether we should reset the data iter before start
   *              doing prediction.
   * @return The return value will be a nested list like
   *         `[[out1_batch1, out2_batch1, ...], [out1_batch2, out2_batch2, ...]]`
   *         This mode is useful because in some cases (e.g. bucketing),
   *         the module does not necessarily produce the same number of outputs.
   */
  def predictEveryBatch(evalData: DataIter, numBatch: Int = -1, reset: Boolean = true)
    : IndexedSeq[IndexedSeq[NDArray]] = {
    require(binded && paramsInitialized)
    if (reset) {
      evalData.reset()
    }
    val outputList = ArrayBuffer.empty[IndexedSeq[NDArray]]

    var nBatch = 0
    while (evalData.hasNext && nBatch != numBatch) {
      val evalBatch = evalData.next()
      outputList.append(predict(evalBatch))
      nBatch += 1
    }

    outputList
  }

  def predict(batch: DataBatch): IndexedSeq[NDArray] = {
    require(binded && paramsInitialized)
    forward(batch, isTrain = Option(false))
    val pad = batch.pad
    getOutputsMerged().map(out =>
      out.slice(0, out.shape(0)-pad).copy()
    )
  }

  /**
   * Run prediction and collect the outputs.
   * @param evalData
   * @param numBatch Default is -1, indicating running all the batches in the data iterator.
   * @param reset Default is `True`, indicating whether we should reset the data iter before start
   *              doing prediction.
   * @return The return value will be a list `[out1, out2, out3]`.
   *         Where each element is concatenation of the outputs for all the mini-batches.
   */
  def predict(evalData: DataIter, numBatch: Int = -1, reset: Boolean = true)
    : IndexedSeq[NDArray] = {
    val outputBatches = predictEveryBatch(evalData, numBatch, reset)
    val numOutputs = outputBatches.head.size
    outputBatches.foreach(out =>
      require(out.size == numOutputs,
      "Cannot merge batches, as num of outputs is not the same in mini-batches." +
      "Maybe bucketing is used?")
    )
    outputBatches.map(out => NDArray.concatenate(out))
  }

  // Symbol information
  // A list of names for data required by this module.
  def dataNames: IndexedSeq[String]

  // A list of names for the outputs of this module.
  def outputNames: IndexedSeq[String]

  // Input/Output information
  // A list of (name, shape) pairs specifying the data inputs to this module.
  def dataShapes: IndexedSeq[DataDesc]

  /**
   * A list of (name, shape) pairs specifying the label inputs to this module.
   * If this module does not accept labels -- either it is a module without loss
   * function, or it is not binded for training, then this should return an empty
   * list `[]`.
   */
  def labelShapes: IndexedSeq[DataDesc]

  // A list of (name, shape) pairs specifying the outputs of this module.
  def outputShapes: IndexedSeq[(String, Shape)]

  // Parameters of a module
  /**
   * Get parameters, those are potentially copies of the the actual parameters used
   * to do computation on the device.
   * @return `(arg_params, aux_params)`, a pair of dictionary of name to value mapping.
   */
  def getParams: (Map[String, NDArray], Map[String, NDArray])

  /**
   * Initialize the parameters and auxiliary states.
   * @param initializer : Initializer
   *         Called to initialize parameters if needed.
   *     arg_params : dict
   *         If not None, should be a dictionary of existing arg_params. Initialization
   *         will be copied from that.
   *     aux_params : dict
   *         If not None, should be a dictionary of existing aux_params. Initialization
   *         will be copied from that.
   *     allow_missing : bool
   *         If true, params could contain missing values, and the initializer will be
   *         called to fill those missing params.
   *     force_init : bool
   *         If true, will force re-initialize even if already initialized.
   */
  def initParams(initializer: Initializer = new Uniform(0.01f),
                 argParams: Map[String, NDArray] = null,
                 auxParams: Map[String, NDArray] = null,
                 allowMissing: Boolean = false, forceInit: Boolean = false): Unit

  /**
   * Assign parameter and aux state values.
   *     arg_params : dict
   *         Dictionary of name to value (`NDArray`) mapping.
   *     aux_params : dict
   *         Dictionary of name to value (`NDArray`) mapping.
   *     allow_missing : bool
   *         If true, params could contain missing values, and the initializer will be
   *         called to fill those missing params.
   *     force_init : bool
   *         If true, will force re-initialize even if already initialized.
   */
  def setParams(argParams: Map[String, NDArray],
                auxParams: Map[String, NDArray],
                allowMissing: Boolean = false,
                forceInit: Boolean = true): Unit = {
    initParams(initializer = null, argParams = argParams, auxParams = auxParams,
      allowMissing = allowMissing, forceInit = forceInit)
  }

  /**
   * Save model parameters to file.
   * @param fname Path to output param file.
   *
   */
  def saveParams(fname: String): Unit = {
    val (argParams, auxParams) = getParams
    val saveDict = (
      argParams.map { case (k, v) => (s"arg:$k", v.asInContext(Context.cpu())) }
      ++ auxParams.map { case (k, v) => (s"aux:$k", v.asInContext(Context.cpu())) }
    )
    NDArray.save(fname, saveDict)
  }

  /**
   * Load model parameters from file.
   * @param fname Path to input param file.
   * @throws IOException if param file is invalid
   */
  @throws(classOf[IOException])
  def loadParams(fname: String): Unit = {
    val saveDict = NDArray.load(fname)
    val argParams = scala.collection.mutable.HashMap.empty[String, NDArray]
    val auxParams = scala.collection.mutable.HashMap.empty[String, NDArray]
    (saveDict._1 zip saveDict._2) foreach { case (key, value) =>
      key.split(":", 2) match {
        case Array(argType, name) if argType == "arg" => argParams.put(name, value)
        case Array(argType, name) if argType == "aux" => auxParams.put(name, value)
        case _ => throw new IOException("Invalid param file " + fname)
      }
    }
    setParams(argParams.toMap, auxParams.toMap)
  }

  /**
   *
   * Train the module parameters.
   * @param trainData
   * @param evalData If not `None`, will be used as validation set and evaluate
   *                 the performance after each epoch.
   * @param numEpoch Number of epochs to run training.
   * @param fitParams Extra parameters for training.
   */
  def fit(trainData: DataIter, evalData: Option[DataIter] = None, numEpoch: Int = 1,
          fitParams: FitParams = new FitParams): Unit = {
    require(fitParams != null)
    require(numEpoch > 0, "please specify number of epochs")
    import ml.dmlc.mxnet.DataDesc._
    bind(dataShapes = trainData.provideData, labelShapes = Option(trainData.provideLabel),
         forTraining = true, forceRebind = fitParams.forceRebind)
    fitParams.monitor.foreach(installMonitor)
    initParams(fitParams.initializer, argParams, auxParams,
      fitParams.allowMissing, fitParams.forceInit)
    initOptimizer(fitParams.kvstore, fitParams.optimizer)

    val valMetric = fitParams.validationMetric.getOrElse(fitParams.evalMetric)

    // training loop
    for (epoch <- fitParams.beginEpoch until numEpoch) {
      val tic = System.currentTimeMillis
      fitParams.evalMetric.reset()

      var nBatch = 0
      while (trainData.hasNext) {
        val dataBatch = trainData.next()

        fitParams.monitor.foreach(_.tic())
        forwardBackward(dataBatch)
        update()
        updateMetric(fitParams.evalMetric, dataBatch.label)
        fitParams.monitor.foreach(_.tocPrint())

        fitParams.batchEndCallback.foreach(callback =>
          callback.invoke(epoch, nBatch, fitParams.evalMetric)
        )

        nBatch += 1
      }

      // one epoch of training is finished
      val (name, value) = fitParams.evalMetric.get
      logger.info(s"Epoch[$epoch] Train-$name=$value")
      val toc = System.currentTimeMillis
      logger.info(s"Epoch[$epoch] Time cost=${toc - tic}")

      // sync aux params across devices
      val (argParamsSync, auxParamsSync) = getParams
      setParams(argParamsSync, auxParamsSync)

      fitParams.epochEndCallback.foreach(callback =>
        callback.invoke(epoch, symbol, argParamsSync, auxParamsSync)
      )

      // evaluation on validation set
      evalData.foreach(data => {
        val res = score(data, valMetric,
          scoreEndCallback = fitParams.evalEndCallback,
          batchEndCallback = fitParams.evalBatchEndCallback, epoch = epoch)
        val (name, value) = res.get
        logger.info(s"Epoch[$epoch] Validation-$name=$value")
      })

      // end of 1 epoch, reset the data-iter for another epoch
      trainData.reset()
    }
  }

  // Install monitor on all executors
  def installMonitor(monitor: Monitor): Unit

  // Computations
  /**
   * Forward computation.
   * @param dataBatch Could be anything with similar API implemented.
   * @param isTrain Default is `None`, which means `isTrain` takes the value of `this.forTraining`.
   */
  def forward(dataBatch: DataBatch, isTrain: Option[Boolean] = None): Unit

  /**
   * Backward computation.
   * @param outGrads Gradient on the outputs to be propagated back.
   *                 This parameter is only needed when bind is called
   *                 on outputs that are not a loss function.
   */
  def backward(outGrads: Array[NDArray] = null): Unit

  /**
   * Get outputs of the previous forward computation.
   * @return In the case when data-parallelism is used,
   *         the outputs will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[out1, out2]`
   */
  def getOutputsMerged(): IndexedSeq[NDArray]

  /**
   * Get outputs of the previous forward computation.
   * @return In the case when data-parallelism is used,
   *         the outputs will be collected from multiple devices.
   *         The results will look like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`,
   *         those `NDArray` might live on different devices.
   */
  def getOutputs(): IndexedSeq[IndexedSeq[NDArray]]

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[grad1, grad2]`
   */
  def getInputGradsMerged(): IndexedSeq[NDArray]

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be collected from multiple devices.
   *         The results will look like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`,
   *         those `NDArray` might live on different devices.
   */
  def getInputGrads(): IndexedSeq[IndexedSeq[NDArray]]

  // Update parameters according to the installed optimizer and the gradients computed
  // in the previous forward-backward batch.
  def update(): Unit

  /**
   * Evaluate and accumulate evaluation metric on outputs of the last forward computation.
   * @param evalMetric
   * @param labels Typically `DataBatch.label`.
   */
  def updateMetric(evalMetric: EvalMetric, labels: IndexedSeq[NDArray]): Unit

  // module setup
  /**
   * Bind the symbols to construct executors.
   * This is necessary before one can perform computation with the module.
   * @param dataShapes Typically is `DataIter.provideData`.
   * @param labelShapes Typically is `DataIter.provideLabel`.
   * @param forTraining Default is `True`. Whether the executors should be bind for training.
   * @param inputsNeedGrad  Default is `False`.
   *                        Whether the gradients to the input data need to be computed.
   *                        Typically this is not needed.
   *                        But this might be needed when implementing composition of modules.
   * @param forceRebind Default is `False`. This function does nothing
   *                    if the executors are already binded. But with this `True`,
   *                    the executors will be forced to rebind.
   * @param sharedModule  Default is `None`. This is used in bucketing. When not `None`,
   *                      the shared module essentially corresponds to a different bucket
   *                      -- a module with different symbol but with the same sets of parameters
   *                      (e.g. unrolled RNNs with different lengths).
   * @param gradReq Requirement for gradient accumulation (globally).
   *                Can be 'write', 'add', or 'null' (default to 'write').
   */
  def bind(dataShapes: IndexedSeq[DataDesc], labelShapes: Option[IndexedSeq[DataDesc]] = None,
           forTraining: Boolean = true, inputsNeedGrad: Boolean = false,
           forceRebind: Boolean = false, sharedModule: Option[BaseModule] = None,
           gradReq: String = "write"): Unit

  // Install and initialize optimizers.
  def initOptimizer(kvstore: String = "local", optimizer: Optimizer = new SGD(),
                    resetOptimizer: Boolean = true, forceInit: Boolean = false): Unit
}

class FitParams {
  private[module] var evalMetric: EvalMetric = new Accuracy()
  private[module] var epochEndCallback: Option[EpochEndCallback] = None
  private[module] var batchEndCallback: Option[BatchEndCallback] = None
  private[module] var kvstore: String = "local"
  private[module] var optimizer: Optimizer = new SGD()
  private[module] var evalEndCallback: Option[BatchEndCallback] = None
  private[module] var evalBatchEndCallback: Option[BatchEndCallback] = None
  private[module] var initializer: Initializer = new Uniform(0.01f)
  private[module] var argParams: Map[String, NDArray] = null
  private[module] var auxParams: Map[String, NDArray] = null
  private[module] var allowMissing: Boolean = false
  private[module] var forceRebind: Boolean = false
  private[module] var forceInit: Boolean = false
  private[module] var beginEpoch: Int = 0
  private[module] var validationMetric: Option[EvalMetric] = None
  private[module] var monitor: Option[Monitor] = None

  // The performance measure used to display during training.
  def setEvalMetric(evalMetric: EvalMetric): FitParams = {
    require(evalMetric != null)
    this.evalMetric = evalMetric
    this
  }

  // Each callback will be called with the current
  // `epoch`, `symbol`, `arg_params` and `aux_params`.
  def setEpochEndCallback(epochEndCallback: EpochEndCallback): FitParams = {
    this.epochEndCallback = Option(epochEndCallback)
    this
  }

  // Each callback will be called with a `BatchEndParam`.
  def setBatchEndCallback(batchEndCallback: BatchEndCallback): FitParams = {
    this.batchEndCallback = Option(batchEndCallback)
    this
  }

  def setKVStore(kvStore: String): FitParams = {
    require(kvStore != null)
    this.kvstore = kvstore
    this
  }

  def setOptimizer(optimizer: Optimizer): FitParams = {
    require(optimizer != null)
    this.optimizer = optimizer
    this
  }

  // These will be called at the end of each full evaluation,
  // with the metrics over the entire evaluation set.
  def setEvalEndCallback(evalEndCallback: BatchEndCallback): FitParams = {
    this.evalEndCallback = Option(evalEndCallback)
    this
  }

  // These will be called at the end of each minibatch during evaluation.
  def setEvalBatchEndCallback(evalBatchEndCallback: BatchEndCallback): FitParams = {
    this.evalBatchEndCallback = Option(evalBatchEndCallback)
    this
  }

  // Will be called to initialize the module parameters if not already initialized.
  def setInitializer(initializer: Initializer): FitParams = {
    require(initializer != null)
    this.initializer = initializer
    this
  }

  // Default `None`, if not `None`, should be existing parameters from a trained
  // model or loaded from a checkpoint (previously saved model). In this case,
  // the value here will be used to initialize the module parameters,
  // unless they are already initialized by the user
  // via a call to `init_params` or `fit`.
  // `argParams` has higher priority to `initializer`.
  def setArgParams(argParams: Map[String, NDArray]): FitParams = {
    this.argParams = argParams
    this
  }

  // Default `None`. Similar to `argParams`, except for auxiliary states.
  def setAuxParams(auxParams: Map[String, NDArray]): FitParams = {
    this.auxParams = auxParams
    this
  }

  // Default `False`. Indicate whether we allow missing parameters
  // when `arg_params` and `aux_params` are not `None`.
  // If this is `True`, then the missing parameters will be
  // initialized via the `initializer`.
  def setAllowMissing(allowMissing: Boolean): FitParams = {
    this.allowMissing = allowMissing
    this
  }

  // Default `False`. Whether to force rebinding the executors if already binded.
  def setForceRebind(forceRebind: Boolean): FitParams = {
    this.forceRebind = forceRebind
    this
  }

  // Default `False`. Indicate whether we should force initialization even if the
  // parameters are already initialized.
  def setForceInit(forceInit: Boolean): FitParams = {
    this.forceInit = forceInit
    this
  }

  // Default `0`. Indicate the starting epoch. Usually, if we are resuming from a
  // checkpoint saved at a previous training phase at epoch N,
  // then we should specify this value as N+1.
  def setBeginEpoch(beginEpoch: Int): FitParams = {
    require(beginEpoch >= 0)
    this.beginEpoch = beginEpoch
    this
  }

  def setValidationMetric(metric: EvalMetric): FitParams = {
    this.validationMetric = Option(metric)
    this
  }

  def setMonitor(monitor: Monitor): FitParams = {
    this.monitor = Option(monitor)
    this
  }
}
