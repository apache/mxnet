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

import ml.dmlc.mxnet.io.NDArrayIter
import ml.dmlc.mxnet.optimizer.SGD
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable.ListBuffer

/**
 * Model class of MXNet for training and predicting feedforward nets.
 * This class is designed for a single-data single output supervised network.
 * @param symbol The symbol configuration of computation network.
 * @param symGen Symbol generator for bucketing.
 * @param ctx The device context of training and prediction.
 *            To use multi GPU training, pass in a list of gpu contexts.
 * @param numEpoch Training parameter, number of training epochs(epochs).
 * @param epochSize Number of batches in a epoch. In default, it is set to
 *                  ceil(num_train_examples / batch_size)
 * @param optimizer Training parameter, name or optimizer object for training.
 * @param initializer Training parameter, the initialization scheme used.
 * @param batchSize The batch size of training data.
 * @param argParams Model parameter, dict of name to NDArray of net's weights.
 * @param auxParams Model parameter, dict of name to NDArray of net's auxiliary states.
 * @param allowExtraParams Whether allow extra parameters that are not needed by symbol
 *                         to be passed by aux_params and arg_params.
 *                         If this is True, no error will be thrown when aux_params and arg_params
 *                         contain extra parameters than needed.
 * @param beginEpoch The beginning training epoch.
 */
class FeedForward private(
    private var symbol: Symbol,
    symGen: SymbolGenerator,
    ctx: Array[Context],
    numEpoch: Int, val epochSize: Int,
    optimizer: Optimizer,
    initializer: Initializer,
    batchSize: Int,
    argParams: Map[String, NDArray],
    auxParams: Map[String, NDArray],
    private val allowExtraParams: Boolean,
    val beginEpoch: Int) {

  val logger: Logger = LoggerFactory.getLogger(classOf[FeedForward])
  private var argumentChecked = false
  private var _argParams = argParams
  private var _auxParams = auxParams
  if (symGen == null) {
    checkArguments()
  }

  def getArgParams: Map[String, NDArray] = _argParams
  def getAuxParams: Map[String, NDArray] = _auxParams

  // internal helper state
  var predExec: Executor = null

  private var monitor: Option[Monitor] = None

  // scalastyle:off parameterNum
  def this(symbol: Symbol, ctx: Array[Context] = Array(Context.cpu()),
           numEpoch: Int = -1, epochSize: Int = -1,
           optimizer: Optimizer = new SGD(),
           initializer: Initializer = new Uniform(0.01f),
           batchSize: Int = 128,
           argParams: Map[String, NDArray] = null,
           auxParams: Map[String, NDArray] = null,
           allowExtraParams: Boolean = false,
           beginEpoch: Int = 0) {
    this(symbol, null, ctx, numEpoch, epochSize, optimizer, initializer, batchSize,
          argParams, auxParams, allowExtraParams, beginEpoch)
  }

  def this(symbol: SymbolGenerator, ctx: Array[Context], numEpoch: Int, epochSize: Int,
           optimizer: Optimizer, initializer: Initializer, batchSize: Int,
           argParams: Map[String, NDArray], auxParams: Map[String, NDArray],
           allowExtraParams: Boolean, beginEpoch: Int) {
    this(null, symbol, ctx, numEpoch, epochSize, optimizer, initializer, batchSize,
      argParams, auxParams, allowExtraParams, beginEpoch)
  }
  // scalastyle:on parameterNum

  // verify the argument of the default symbol and user provided parameters
  def checkArguments(): Unit = {
    if (!argumentChecked) {
      require(symbol != null)
      // check if symbol contain duplicated names.
      ExecutorManager.checkArguments(symbol)
      // rematch parameters to delete useless ones
      if (allowExtraParams) {
        if (_argParams != null) {
          val argNames = symbol.listArguments().toSet
          _argParams = _argParams.filter { case (k, v) => argNames.contains(k) }
        }
        if (auxParams != null) {
          val auxNames = symbol.listAuxiliaryStates().toSet
          _auxParams = _auxParams.filter { case (k, v) => auxNames.contains(k) }
        }
      }
      argumentChecked = true
    }
  }

  def setMonitor(m: Monitor): Unit = {
    monitor = Option(m)
  }

  def unsetMonitor(): Unit = {
    setMonitor(null)
  }

  // Initialize weight parameters and auxiliary states
  private def initParams(inputShapes: Map[String, Shape], overwrite: Boolean = false)
  : (IndexedSeq[String], IndexedSeq[String], IndexedSeq[String]) = {
    val (argShapes, _, auxShapes) = symbol.inferShape(inputShapes)
    val argNames = symbol.listArguments()
    val inputNames = inputShapes.keys.toSet
    val paramNames = argNames.filter(!inputNames.contains(_))
    val auxNames = symbol.listAuxiliaryStates()

    val paramNameShapes = (argNames zip argShapes).filter { case (name, _) =>
      paramNames.contains(name)
    }
    val argParams = paramNameShapes.map { case (name, shape) =>
      (name, NDArray.zeros(shape))
    }.toMap
    val auxParams = (auxNames zip auxShapes).map { case (name, shape) =>
      (name, NDArray.zeros(shape))
    }.toMap

    for ((k, v) <- argParams) {
      if (_argParams != null && _argParams.contains(k) && (!overwrite)) {
        argParams(k).set(_argParams(k))
      } else {
        initializer(k, v)
      }
    }

    for ((k, v) <- auxParams) {
      if (_auxParams != null && _auxParams.contains(k) && (!overwrite)) {
        auxParams(k).set(_auxParams(k))
      } else {
        initializer(k, v)
      }
    }

    _argParams = argParams
    _auxParams = auxParams
    (argNames, paramNames, auxNames)
  }

  // Initialize the predictor module for running prediction.
  private def initPredictor(inputShapes: Map[String, Shape]): Unit = {
    if (this.predExec != null) {
      val (argShapes, _, _) = symbol.inferShape(inputShapes)
      require(argShapes != null, "Incomplete input shapes")
      val predShapes = this.predExec.argArrays.map(_.shape)
      if (argShapes.sameElements(predShapes)) {
        return
      }
    }
    // for now only use the first device
    val predExec = symbol.simpleBind(ctx(0), gradReq = "null", shapeDict = inputShapes)
    predExec.copyParamsFrom(_argParams, _auxParams)
    ExecutorManager.checkArguments(symbol)
    this.predExec = predExec
  }

  // Initialize the iterator given input.
  private def initIter(X: NDArray, y: NDArray, isTrain: Boolean): DataIter = {
    require(y != null || !isTrain, "y must be specified")
    val label = if (y == null) NDArray.zeros(X.shape(0)) else y
    require(label.shape.length == 1, "Label must be 1D")
    require(X.shape(0) == label.shape(0), "The numbers of data points and labels not equal")
    if (isTrain) {
      new NDArrayIter(IndexedSeq(X), IndexedSeq(label), batchSize,
        shuffle = isTrain, lastBatchHandle = "roll_over")
    } else {
      new NDArrayIter(IndexedSeq(X), IndexedSeq(label), batchSize, shuffle = false)
    }
  }

  // Initialize the iterator given eval_data.
  private def initEvalIter(evalData: (NDArray, NDArray)): DataIter = {
    if (evalData == null) {
      null
    } else {
      initIter(evalData._1, evalData._2, isTrain = true)
    }
  }

  /**
   * Run the prediction, always only use one device.
   * @param data eval data
   * @param numBatch the number of batch to run. Go though all batches if set -1
   * @return The predicted value of the output.
   *         Note the network may have multiple outputs, thus it return an array of [[NDArray]]
   */
  def predict(data: DataIter, numBatch: Int = -1): Array[NDArray] = {
    data.reset()
    val dataShapes = data.provideData
    val dataNames = dataShapes.map(_._1).toArray
    initPredictor(dataShapes)
    val batchSize = data.batchSize
    val dataArrays = dataNames.map(predExec.argDict(_))
    val outputs = Array.fill(predExec.outputs.length)(ListBuffer.empty[NDArray])

    var i = 0
    while (data.hasNext && i != numBatch) {
      val batch = data.next()
      i += 1
      ExecutorManager.loadData(batch, dataArrays)
      predExec.forward(isTrain = false)
      val padded = batch.pad
      val realSize = batchSize - padded
      for ((list, nd) <- outputs zip predExec.outputs) {
        list += nd.slice(0, realSize).copy()
      }
    }
    // TODO(Yizhi): we can use Symbol.concat to do the same thing. Can it be more efficient?
    val results = outputs.map(NDArray.concatenate(_))
    for (output <- outputs) {
      output.foreach(_.dispose())
    }
    results
  }

  /**
   * Fit the model.
   * @param trainData Training data
   * @param evalData Evaluation data
   * @param evalMetric The evaluation metric, cannot be null
   * @param epochEndCallback A callback that is invoked at end of each epoch.
   *                         This can be used to checkpoint model each epoch.
   * @param batchEndCallback A callback that is invoked at end of each batch
   *                         For print purpose
   * @param kvStoreType A string kvstore type:
   *                    'local' : multi-devices on a single machine, will automatically
   *                    choose one from 'local_update_cpu', 'local_allreduce_cpu', and
   *                    'local_allreduce_device'
   *                    'dist_sync' : multi-machines with BSP
   *                    'dist_async' : multi-machines with partical asynchronous
   *                    In default uses 'local', often no need to change for single machine.
   * @param logger When not specified, default logger will be used.
   * @param workLoadList The list of work load for different devices, in the same order as ctx
   */
  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric, kvStoreType: String,
          epochEndCallback: EpochEndCallback, batchEndCallback: BatchEndCallback,
          logger: Logger, workLoadList: Seq[Float]): Unit = {
    // init params first to allow kv store use _argParams to decide its type
    initSymbolParams(trainData)
    // create kvstore
    val (kvStore, updateOnKVStore) = Model.createKVStore(kvStoreType, ctx.length, _argParams)
    fit(trainData, evalData, evalMetric, kvStore, updateOnKVStore,
      epochEndCallback, batchEndCallback, logger, workLoadList)
    kvStore.foreach(_.dispose())
  }

  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric,
          kvStoreType: String, epochEndCallback: EpochEndCallback,
          batchEndCallback: BatchEndCallback): Unit = {
    fit(trainData, evalData, evalMetric, kvStoreType,
      epochEndCallback, batchEndCallback, FeedForward.logger, null)
  }

  def fit(trainData: DataIter, evalData: DataIter,
          evalMetric: EvalMetric, kvStoreType: String): Unit = {
    fit(trainData, evalData, evalMetric, kvStoreType,
      epochEndCallback = null, batchEndCallback = null)
  }

  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric): Unit = {
    fit(trainData, evalData, evalMetric, kvStoreType = "local")
  }

  def fit(trainData: DataIter, evalData: DataIter): Unit = {
    fit(trainData, evalData, new Accuracy())
  }

  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric,
          kv: KVStore,
          epochEndCallback: EpochEndCallback,
          batchEndCallback: BatchEndCallback, logger: Logger,
          workLoadList: Seq[Float]): Unit = {
    // init params first to allow kv store use _argParams to decide its type
    initSymbolParams(trainData)
    // create kvstore
    val (kvStore, updateOnKVStore) = Model.createKVStore(kv)
    fit(trainData, evalData, evalMetric, kvStore, updateOnKVStore,
      epochEndCallback, batchEndCallback, logger, workLoadList)
  }

  def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric,
          kvStore: KVStore,
          epochEndCallback: EpochEndCallback,
          batchEndCallback: BatchEndCallback): Unit = {
    fit(trainData, evalData, evalMetric, kvStore, epochEndCallback,
        batchEndCallback, FeedForward.logger, null)
  }

  def fit(trainData: DataIter, evalData: DataIter,
          evalMetric: EvalMetric, kvStore: KVStore): Unit = {
    fit(trainData, evalData, evalMetric, kvStore, epochEndCallback = null, batchEndCallback = null)
  }

  def fit(trainData: DataIter, evalData: DataIter, kvStore: KVStore): Unit = {
    fit(trainData, evalData, new Accuracy(), kvStore)
  }

  private def initSymbolParams(trainData: DataIter)
    : (IndexedSeq[String], IndexedSeq[String], IndexedSeq[String]) = {
    if (symGen != null) {
      this.symbol = symGen.generate(trainData.defaultBucketKey)
      checkArguments()
    }
    initParams(trainData.provideData ++ trainData.provideLabel)
  }

  private def fit(trainData: DataIter, evalData: DataIter, evalMetric: EvalMetric = new Accuracy(),
                  kvStore: Option[KVStore], updateOnKVStore: Boolean,
                  epochEndCallback: EpochEndCallback = null,
                  batchEndCallback: BatchEndCallback = null, logger: Logger = FeedForward.logger,
                  workLoadList: Seq[Float] = null): Unit = {
    require(evalMetric != null, "evalMetric cannot be null")
    val (argNames, paramNames, auxNames) = initSymbolParams(trainData)

    // init optimizer
    val batchSizeMultiplier = kvStore.map { kv =>
      if (kv.`type` == "dist_sync") {
        kv.numWorkers
      } else {
        1
      }
    }
    val batchSize = trainData.batchSize * batchSizeMultiplier.getOrElse(1)
    this.optimizer.setArgNames(argNames)
    this.optimizer.setRescaleGrad(1f / batchSize)
    this.optimizer.setSymbol(this.symbol)
    val paramIdx2Name =
      if (updateOnKVStore) {
        paramNames.zipWithIndex.map { case (name, idx) => idx -> name }.toMap
      } else {
        paramNames.zipWithIndex.flatMap { case (name, idx) =>
          (0 until ctx.length).map(k => (idx * ctx.length + k) -> name).toMap
        }.toMap
      }
    this.optimizer.setIdx2Name(paramIdx2Name)

    logger.debug("Start training on multi-device")
    Model.trainMultiDevice(
      symbol, ctx, argNames, paramNames, auxNames,
      _argParams, _auxParams,
      this.beginEpoch, this.numEpoch,
      this.epochSize, this.optimizer,
      kvStore, updateOnKVStore,
      trainData = trainData, evalData = Option(evalData),
      evalMetric = evalMetric,
      epochEndCallback = Option(epochEndCallback),
      batchEndCallback = Option(batchEndCallback),
      workLoadList = workLoadList,
      monitor = monitor,
      symGen = symGen)
  }

  /**
   * Checkpoint the model checkpoint into file.
   * You can also use pickle to do the job if you only work on python.
   * The advantage of load/save is the file is language agnostic.
   * This means the file saved using save can be loaded by other language binding of mxnet.
   * You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)
   * @param prefix Prefix of model name.
   * @see FeedForward.load : the method to load the model back.
   * @note
   * - ``prefix-symbol.json`` will be saved for symbol.
   * - ``prefix-epoch.params`` will be saved for parameters.
   */
  def save(prefix: String, epoch: Int = this.numEpoch): Unit = {
    require(epoch >= 0)
    Model.saveCheckpoint(prefix, epoch, this.symbol, getArgParams, getAuxParams)
  }

  /**
   * Serialize the model to Java byte array
   * @return serialized model bytes
   */
  def serialize(): Array[Byte] = {
    Model.serialize(this.symbol, getArgParams, getAuxParams)
  }
}

object FeedForward {
  private val logger: Logger = LoggerFactory.getLogger(classOf[FeedForward])
  // Check if name is a data argument.
  private def isDataArg(name: String): Boolean = {
    name.endsWith("data") || name.endsWith("label")
  }

  /**
   * Load model checkpoint from file.
   * @param prefix Prefix of model name.
   * @param epoch epoch number of model we would like to load.
   * @return The loaded model that can be used for prediction.
   * @note
   * - ``prefix-symbol.json`` will be saved for symbol.
   * - ``prefix-epoch.params`` will be saved for parameters.
   */
  def load(prefix: String, epoch: Int,
           ctx: Array[Context] = Array(Context.cpu()),
           numEpoch: Int = -1,
           epochSize: Int = -1,
           optimizer: Optimizer = new SGD(),
           initializer: Initializer = new Uniform(0.01f),
           batchSize: Int = 128,
           allowExtraParams: Boolean = false): FeedForward = {
    val (symbol, argParams, auxParams) = Model.loadCheckpoint(prefix, epoch)
    new FeedForward(symbol, ctx = ctx,
      argParams = argParams, auxParams = auxParams,
      beginEpoch = epoch, numEpoch = numEpoch,
      epochSize = epochSize, optimizer = optimizer,
      initializer = initializer, batchSize = batchSize,
      allowExtraParams = allowExtraParams)
  }

  /**
   * Deserialize bytes to model.
   * @param bytes serialized model bytes.
   * @return The loaded model that can be used for prediction.
   */
  def deserialize(bytes: Array[Byte], epoch: Int = 0,
                  ctx: Array[Context] = Array(Context.cpu()),
                  numEpoch: Int = -1,
                  epochSize: Int = -1,
                  optimizer: Optimizer = new SGD(),
                  initializer: Initializer = new Uniform(0.01f),
                  batchSize: Int = 128,
                  allowExtraParams: Boolean = false): FeedForward = {
    val (symbol, argParams, auxParams) = Model.deserialize(bytes)
    new FeedForward(symbol, ctx = ctx,
      argParams = argParams, auxParams = auxParams,
      beginEpoch = epoch, numEpoch = numEpoch,
      epochSize = epochSize, optimizer = optimizer,
      initializer = initializer, batchSize = batchSize,
      allowExtraParams = allowExtraParams)
  }

  def newBuilder(modelDef: Symbol): Builder = new Builder(modelDef, null)
  def newBuilder(symGen: SymbolGenerator): Builder = new Builder(null, symGen)

  class Builder private[FeedForward](private val modelDef: Symbol,
                                     private val symGen: SymbolGenerator) {
    private var ctx: Array[Context] = Array(Context.cpu())
    private var numEpoch: Int = -1
    private var epochSize: Int = -1
    private var optimizer: Optimizer = new SGD()
    private var initializer: Initializer = new Uniform(0.01f)
    private var batchSize: Int = 128
    private var argParams: Map[String, NDArray] = null
    private var auxParams: Map[String, NDArray] = null
    private var allowExtraParams: Boolean = false
    private var beginEpoch: Int = 0
    private var trainData: DataIter = null
    private var evalData: DataIter = null
    private var evalMetric: EvalMetric = new Accuracy()

    private var kvStoreInst: KVStore = null
    private var kvStoreType: String = "local"

    private var epochEndCallback: EpochEndCallback = null
    private var batchEndCallback: BatchEndCallback = null
    private var logger: Logger = FeedForward.logger
    private var workLoadList: Seq[Float] = null

    /**
     * Set ctx The device context of training and prediction.
     * To use multi GPU training, pass in a list of gpu contexts.
     */
    def setContext(ctx: Array[Context]): Builder = {
      this.ctx = ctx
      this
    }

    /**
     * Set number of training epochs
     */
    def setNumEpoch(numEpoch: Int): Builder = {
      this.numEpoch = numEpoch
      this
    }

    /**
     * Set number of batches in a epoch. In default, it is set to
     * ceil(num_train_examples / batch_size)
     */
    def setEpochSize(epochSize: Int): Builder = {
      this.epochSize = epochSize
      this
    }

    /**
     * Set optimizer for training. Default SGD.
     */
    def setOptimizer(opt: Optimizer): Builder = {
      this.optimizer = opt
      this
    }

    /**
     * Set the initialization scheme used. Default Uniform(0.01f).
     */
    def setInitializer(initializer: Initializer): Builder = {
      this.initializer = initializer
      this
    }

    /**
     * Set the batch size of training data.
     */
    def setBatchSize(batchSize: Int): Builder = {
      this.batchSize = batchSize
      this
    }

    /**
     * Set the model parameter, dict of name to NDArray of net's weights.
     */
    def setArgParams(argParams: Map[String, NDArray]): Builder = {
      this.argParams = argParams
      this
    }

    /**
     * Set the model parameter, dict of name to NDArray of net's auxiliary states
     */
    def setAuxParams(auxParams: Map[String, NDArray]): Builder = {
      this.auxParams = auxParams
      this
    }

    /**
     * Whether allow extra parameters that are not needed by symbol
     * to be passed by aux_params and arg_params.
     * If this is True, no error will be thrown when aux_params and arg_params
     * contain extra parameters than needed.
     */
    def setAllowExtraParams(allowExtraParams: Boolean): Builder = {
      this.allowExtraParams = allowExtraParams
      this
    }

    /**
     * Set the beginning training epoch.
     */
    def setBeginEpoch(beginEpoch: Int): Builder = {
      this.beginEpoch = beginEpoch
      this
    }

    /**
     * Set the training data
     */
    def setTrainData(trainData: DataIter): Builder = {
      this.trainData = trainData
      this
    }

    /**
     * Set the evaluation data
     */
    def setEvalData(evalData: DataIter): Builder = {
      this.evalData = evalData
      this
    }

    /**
     * Set the evaluation metric. Default Accuracy()
     */
    def setEvalMetric(metric: EvalMetric): Builder = {
      this.evalMetric = metric
      this
    }

    /**
     * this will take precedence over the setKVStore(String) version
     */
    def setKVStore(kv: KVStore): Builder = {
      this.kvStoreInst = kv
      this
    }

    /**
     * A string kvstore type:
     * 'local' : multi-devices on a single machine, will automatically
     * choose one from 'local_update_cpu', 'local_allreduce_cpu', and
     * 'local_allreduce_device'
     * 'dist_sync' : multi-machines with BSP
     * 'dist_async' : multi-machines with partical asynchronous
     * In default uses 'local', often no need to change for single machine.
     */
    def setKVStore(kv: String): Builder = {
      this.kvStoreType = kv
      this
    }

    /**
     * A callback that is invoked at end of each epoch.
     * This can be used to checkpoint model each epoch.
     */
    def setEpochEndCallback(epochEndCallback: EpochEndCallback): Builder = {
      this.epochEndCallback = epochEndCallback
      this
    }

    /**
     * batchEndCallback A callback that is invoked at end of each batch.
     * For print purpose.
     */
    def setBatchEndCallback(batchEndCallback: BatchEndCallback): Builder = {
      this.batchEndCallback = batchEndCallback
      this
    }

    /**
     * When not specified, default logger will be used.
     */
    def setLogger(logger: Logger): Builder = {
      this.logger = logger
      this
    }

    /**
     * Set the list of work load for different devices, in the same order as ctx
     */
    def setWorkLoadList(workLoadList: Seq[Float]): Builder = {
      this.workLoadList = workLoadList
      this
    }

    /**
     * Construct the FeedForward model and fit on the input training data
     * @return the trained model
     */
    def build(): FeedForward = {
      require(trainData != null, "Training data missing")
      val model = new FeedForward(
        modelDef, symGen, ctx, numEpoch, epochSize,
        optimizer, initializer, batchSize,
        argParams, auxParams, allowExtraParams, beginEpoch)
      if (kvStoreInst == null) {
        model.fit(trainData, evalData, evalMetric, kvStoreType,
                  epochEndCallback, batchEndCallback, logger, workLoadList)
      } else {
        model.fit(trainData, evalData, evalMetric, kvStoreInst,
                  epochEndCallback, batchEndCallback, logger, workLoadList)
      }
      model
    }

    /**
     * Construct the FeedForward model but do NOT train
     * @return the un-trained model
     */
    def setup(): FeedForward = {
      new FeedForward(
        modelDef, symGen, ctx, numEpoch, epochSize,
        optimizer, initializer, batchSize,
        argParams, auxParams, allowExtraParams, beginEpoch)
    }
  }
}
