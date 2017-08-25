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
import scala.collection.mutable.ArrayBuffer
import ml.dmlc.mxnet.optimizer.SGD
import scala.collection.immutable.ListMap

/**
 * A SequentialModule is a container module that can chain multiple modules together.
 * Note building a computation graph with this kind of imperative container is less
 * flexible and less efficient than the symbolic graph.
 * So this should be only used as a handy utility.
 */
class SequentialModule extends BaseModule {

  private val logger = LoggerFactory.getLogger(classOf[SequentialModule])

  private val META_TAKE_LABELS = "take_labels"
  private val META_AUTO_WIRING = "auto_wiring"
  private val metaKeys = Set(META_TAKE_LABELS, META_AUTO_WIRING)

  private val modules = ArrayBuffer[BaseModule]()
  private val metas = ArrayBuffer[Map[String, Boolean]]()
  private var labelShapesVar: Option[IndexedSeq[DataDesc]] = None

  /**
   * Add a module to the chain.
   * An example of addinging two modules to a chain:
   * val seqMod = new SequentialModule()
   * seqMod.add(mod1).add(mod2)
   * @param module The new module to add.
   * @param kwargs All the keyword arguments are saved as meta information
   *                                for the added module. The currently known meta includes
   *                                - "take_labels": indicating whether the module expect to
   *                                take labels when doing computation. Note any module in
   *                                the chain can take labels (not necessarily only the top
   *                                most one), and they all take the same labels passed
   *                                from the original data batch for the `SequentialModule`.
   * @return This function returns `this` to allow us to easily chain a series of `add` calls.
   */
  def add(module: BaseModule, kwargs: (String, Boolean)*): SequentialModule = {
    this.modules += module

    // a sanity check to avoid typo
    kwargs.foreach { case (k, v) =>
      require(this.metaKeys.contains(k), s"Unknown meta $k,auxParams a typo?")
    }

    this.metas += kwargs.map(kw => kw._1 -> kw._2).toMap

    // after adding new modules, we are reset back to raw states, needs
    // to bind, init_params, etc.
    this.binded = false
    this.paramsInitialized = false
    this.optimizerInitialized = false

    this
  }

  /**
   * @return A list of names for data required by this module.
   */
  override def dataNames: IndexedSeq[String] = {
    if (this.modules.length > 0) this.modules.head.dataNames
    else IndexedSeq[String]()
  }

  /**
   * @return A list of names for the outputs of this module.
   */
  override def outputNames: IndexedSeq[String] = {
    if (this.modules.length > 0) this.modules.reverse.head.outputNames
    else IndexedSeq[String]()
  }

  /**
   * Get data shapes.
   * @return The data shapes of the first module is the data shape of a SequentialModule.
   */
  override def dataShapes: IndexedSeq[DataDesc] = {
    require(this.binded)
    this.modules.head.dataShapes
  }

  /**
   * Get label shapes.
   * @return The return value could be null if
   * the module does not need labels, or if the module is not binded for
   * training (in this case, label information is not available).
   */
  override def labelShapes: IndexedSeq[DataDesc] = {
    require(this.binded)
    this.labelShapesVar.orNull
  }

  /**
   * Get output shapes.
   * @return The output shapes of the last
   * module is the output shape of a SequentialModule.
   */
  override def outputShapes: IndexedSeq[(String, Shape)] = {
    require(this.binded)
    this.modules.reverse.head.outputShapes
  }

    /**
   * Get current parameters.
   * @return (argParams, auxParams),
   * each a Map of name to parameters (in NDArray) mapping.
   */
  override def getParams: (Map[String, NDArray], Map[String, NDArray]) = {
    require(this.binded && this.paramsInitialized)
    ((Map[String, NDArray](), Map[String, NDArray]()) /: this.modules){ (result, module) =>
      val (arg, aux) = module.getParams
      (result._1 ++ arg, result._2 ++ aux)
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
   */
  override def initParams(initializer: Initializer = new Uniform(0.01f),
                          argParams: Map[String, NDArray] = null,
                          auxParams: Map[String, NDArray] = null,
                          allowMissing: Boolean = false, forceInit: Boolean = false): Unit = {
    if (this.paramsInitialized && !forceInit) {
      return
    }
    require(this.binded, "call bind before initializing the parameters")

    for (module <- this.modules) {
      module.initParams(initializer = initializer, argParams = argParams,
          auxParams = auxParams, allowMissing = allowMissing, forceInit = forceInit)
    }

    // Internal function to help checking duplicated names,
    // make sure we do not have duplicated parameter names.
    def checkName(knownNames: scala.collection.mutable.Map[String, Int],
      newNames: Array[String], modules: ArrayBuffer[BaseModule], i: Int): Unit = {
      for (name <- newNames) {
        require(!knownNames.contains(name), s"Duplicated parameter names: " +
            s"name $name in layer $i (${modules(i).getClass.getName}) is already " +
            s"used in layer ${knownNames("name")}" +
            s"(${modules(knownNames("name")).getClass.getName})")
        knownNames(name) = i
      }
    }

    val argNames = scala.collection.mutable.Map[String, Int]()
    val auxNames = scala.collection.mutable.Map[String, Int]()
    for ((module, iLayer) <- this.modules.zipWithIndex) {
      val (argParams, auxParams) = module.getParams
      checkName(argNames, argParams.keys.toArray, this.modules, iLayer)
      checkName(auxNames, auxParams.keys.toArray, this.modules, iLayer)
    }
    this.paramsInitialized = true
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
   * @param gradReq Requirement for gradient accumulation (globally).
   *                Can be 'write', 'add', or 'null' (default to 'write').
   */
  override def bind(dataShapes: IndexedSeq[DataDesc],
                    labelShapes: Option[IndexedSeq[DataDesc]] = None,
                    forTraining: Boolean = true, inputsNeedGrad: Boolean = false,
                    forceRebind: Boolean = false, sharedModule: Option[BaseModule] = None,
                    gradReq: String = "write"): Unit = {
    if (this.binded && !forceRebind) {
      logger.warn(s"Already binded, ignoring bind()")
      return
    }

    if (inputsNeedGrad) {
      require(forTraining == true)
    }

    require(sharedModule == None, "Shared module is not supported")
    require(this.modules.length > 0, "Attempting to bind an empty SequentialModule")

    this.forTraining = forTraining
    this.inputsNeedGrad = inputsNeedGrad
    this.binded = true

    // the same label shapes are used for all chained modules
    this.labelShapesVar = labelShapes

    var myDataShapes = dataShapes
    var myLabelShapes = labelShapes
    var anybodyEverNeedsLabel = false
    for ((module, iLayer) <- this.modules.zipWithIndex) {
      val meta = this.metas(iLayer)
      if (meta.contains(META_TAKE_LABELS) && meta(META_TAKE_LABELS)) {
        myLabelShapes = labelShapes
        anybodyEverNeedsLabel = true
      } else myLabelShapes = None

      val myInputsNeedGrad = if (inputsNeedGrad || (forTraining && iLayer > 0)) true else false
      if (meta.contains(META_AUTO_WIRING) && meta(META_AUTO_WIRING)) {
        val dataNames = module.dataNames
        require(dataNames.length == myDataShapes.length)
        myDataShapes = dataNames.zip(myDataShapes).map { case (newName, dataDes) =>
          DataDesc(newName, dataDes.shape)
        }
      }

      module.bind(myDataShapes, myLabelShapes, forTraining, myInputsNeedGrad,
          forceRebind, sharedModule = None, gradReq)
      // the output of the previous module is the data of the next module
      myDataShapes = module.outputShapes.map{case (name, shape) => DataDesc(name, shape)}
    }


    if (!anybodyEverNeedsLabel) {
      // then I do not need label either
      this.labelShapesVar = None
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
  override def initOptimizer(kvstore: String = "local", optimizer: Optimizer = new SGD(),
      resetOptimizer: Boolean = true, forceInit: Boolean = false): Unit = {
    require(this.binded && this.paramsInitialized)
    if (optimizerInitialized && !forceInit) {
      logger.warn("optimizer already initialized, ignoring ...")
    } else {
      for (module <- this.modules) {
        module.initOptimizer(kvstore, optimizer, resetOptimizer, forceInit)
      }
    }
    this.optimizerInitialized = true
  }

  /**
   * Forward computation.
   * @param dataBatch input data
   * @param isTrain Default is `None`, which means `isTrain` takes the value of `forTraining`.
   */
  override def forward(dataBatch: DataBatch, isTrain: Option[Boolean] = None): Unit = {
    require(this.binded && this.paramsInitialized)

    var data = dataBatch
    for ((module, iLayer) <- this.modules.zipWithIndex) {
      module.forward(data, isTrain = isTrain)
      // the last layer, do not need to do the followings
      if (iLayer < this.modules.length - 1) {
        val out = module.getOutputs()
        // need to update this, in case the internal module is using bucketing
        // or whatever
        val dataNames = module.outputShapes.map(_._1)
        require(dataNames.length == data.data.length)
        var provideData = ListMap[String, Shape]()
        for ((name, x) <- dataNames.zip(out.head)) {
          provideData += name -> x.shape
        }
        data = new DataBatch(out.head, data.label, data.index,
            data.pad, data.bucketKey, provideData, data.provideLabel)
      }
    }
  }

  /**
   * Backward computation.
   * @param outGrads Gradient on the outputs to be propagated back.
   *                 This parameter is only needed when bind is called
   *                 on outputs that are not a loss function.
   */
  override def backward(outGrads: Array[NDArray] = null): Unit = {
    require(this.binded && this.paramsInitialized)
    var grad = outGrads
    for ((module, iLayer) <- this.modules.zipWithIndex.reverse) {
      module.backward(outGrads = grad)
      if (iLayer > 0) {
        grad = module.getInputGradsMerged().toArray
      }
    }
  }

  // Update parameters according to the installed optimizer and the gradients computed
  // in the previous forward-backward batch.
  override def update(): Unit = {
    require(this.binded && this.paramsInitialized && this.optimizerInitialized)
    this.modules.foreach(_.update())
  }

  /**
   * Get outputs of the previous forward computation.
   * @return In the case when data-parallelism is used,
   *         the outputs will be collected from multiple devices.
   *         The results will look like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`,
   *         those `NDArray` might live on different devices.
   */
  def getOutputs(): IndexedSeq[IndexedSeq[NDArray]] = {
    require(this.binded && this.paramsInitialized)
    this.modules.reverse.head.getOutputs()
  }

  /**
   * Get outputs of the previous forward computation.
   * @return In the case when data-parallelism is used,
   *         the outputs will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[out1, out2]`
   */
  def getOutputsMerged(): IndexedSeq[NDArray] = {
    require(this.binded && this.paramsInitialized)
    this.modules.reverse.head.getOutputsMerged()
  }

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be collected from multiple devices.
   *         The results will look like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`,
   *         those `NDArray` might live on different devices.
   */
  def getInputGrads(): IndexedSeq[IndexedSeq[NDArray]] = {
    require(this.binded && this.paramsInitialized && inputsNeedGrad)
    this.modules.head.getInputGrads()
  }

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[grad1, grad2]`
   */
  def getInputGradsMerged(): IndexedSeq[NDArray] = {
    require(this.binded && this.paramsInitialized && inputsNeedGrad)
    this.modules.head.getInputGradsMerged()
  }

  /**
   * Evaluate and accumulate evaluation metric on outputs of the last forward computation.
   * @param evalMetric
   * @param labels
   */
  def updateMetric(evalMetric: EvalMetric, labels: IndexedSeq[NDArray]): Unit = {
    require(this.binded && this.paramsInitialized)
    for ((meta, module) <- this.metas.zip(this.modules)) {
      if (meta.contains(META_TAKE_LABELS) && meta(META_TAKE_LABELS)) {
        module.updateMetric(evalMetric, labels)
      }
    }
  }

  // Install monitor on all executors
  def installMonitor(monitor: Monitor): Unit = {
    require(this.binded)
    this.modules.foreach(_.installMonitor(monitor))
  }
}
