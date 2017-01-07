package ml.dmlc.mxnet.module

import ml.dmlc.mxnet.DType.DType
import ml.dmlc.mxnet._
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.immutable.ListMap
import scala.collection.mutable

/**
 * Module is a basic module that wrap a `Symbol`. It is functionally the same
 * as the `FeedForward` model, except under the module API.
 * @param symbol : Symbol
    data_names : list of str
        Default is `('data')` for a typical model used in image classification.
    label_names : list of str
        Default is `('softmax_label')` for a typical model used in image
        classification.
    logger : Logger
        Default is `logging`.
    context : Context or list of Context
        Default is `cpu()`.
    work_load_list : list of number
        Default `None`, indicating uniform workload.
    fixed_param_names: list of str
        Default `None`, indicating no network parameters are fixed.
 */
class Module(private val symbol: Symbol, val dataNames: IndexedSeq[String] = IndexedSeq("data"),
             labelNames: IndexedSeq[String] = IndexedSeq("softmax_label"),
             context: Array[Context] = Context.cpu(), workLoadList: Option[Seq[Float]] = None,
             fixedParamNames: Option[Seq[String]] = None) extends BaseModule {
  private val logger = LoggerFactory.getLogger(classOf[Module])
  // TODO: module.DataParallelExecutorGroup
  private var execGroup: DataParallelExecutorGroup = null
  private var paramsInitialized: Boolean = false

  private val workLoads = workLoadList.getOrElse(context.map(_ => 1f).toSeq)
  require(workLoads.size == context.length)

  private val labelNameList = if (labelNames == null) IndexedSeq.empty[String] else labelNames

  private val argNames = symbol.listArguments()
  private val inputNames = dataNames ++ labelNameList
  private val paramNames = argNames.filterNot(inputNames.toSet)
  private val auxNames = symbol.listAuxiliaryStates()
  val outputNames = symbol.listOutputs()

  private var argParams: Map[String, NDArray] = null
  private var auxParams: Map[String, NDArray] = null
  private var paramsDirty = false

  private var optimizer: Optimizer = null
  private var kvstore: KVStore = null
  private var updateOnKvstore = None
  private var updater = None

  private var dataShapes: Seq[DataDesc] = null
  private var labelShapes: Option[Seq[DataDesc]] = None

  // Internal function to reset binded state.
  private def resetBind(): Unit = {
    binded = false
    execGroup = null
    dataShapes = null
    labelShapes = null
  }

  def getDataShapes(): ListMap[String, Shape] = {
    require(binded)
    dataShapes
  }

  def getLabelShapes(): ListMap[String, Shape] = {
    require(binded)
    labelShapes
  }

  def getOutputShapes(): ListMap[String, Shape] = {
    require(binded)
    null // TODO
  }

  /**
   * Get current parameters.
   * `(arg_params, aux_params)`, each a dictionary of name to parameters (in
   * `NDArray`) mapping.
   */
  def getParams(): (Map[String, NDArray], Map[String, NDArray]) = {
    require(binded && paramsInitialized)
    if (paramsDirty) {
      // TODO: sync_params_from_devices()
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
  def initParams(initializer: Initializer = new Uniform(0.01f),
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
      _impl(name, arr, allowMissing, Some(initializer), argParams)
    }

    this.auxParams.foreach { case (name, arr) =>
      _impl(name, arr, allowMissing, Some(initializer), auxParams)
    }

    this.paramsInitialized = true
    this.paramsDirty = false // copy the initialized parameters to devices
    // TODO: this.execGroup.setParams(self._arg_params, self._aux_params)
  }

  // Internal helper for parameter initialization
  private def _impl(name: String, arr: NDArray, allowMissing: Boolean,
                    initializer: Option[Initializer] = None,
                    cache: Map[String, NDArray] = null): Unit = {
    if (cache != null) {
      if (cache.contains(name)) {
        val cacheArr = cache(name) // just in case the cached array is just the target itself
        if (cacheArr != arr) {
          cacheArr.copyTo(arr)
        }
      } else {
        if (allowMissing) {
          throw new RuntimeException(s"$name is not presented")
        }
        initializer.foreach(inst => inst(name, arr))
      }
    } else {
      initializer.foreach(inst => inst(name, arr))
    }
  }

  // Internal function to reset binded state.
  private def _reset_bind(): Unit = {
    binded = false
    execGroup = null
    dataShapes = null
    labelShapes = null
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
  def bind(dataShapes: Seq[DataDesc], labelShapes: Option[Seq[DataDesc]] = None,
           forTraining: Boolean = true, inputsNeedGrad: Boolean = false,
           forceRebind: Boolean = false, sharedModule: Module = null,
           gradReq: String = "write"): Unit = {
    // force rebinding is typically used when one want to switch from training to prediction phase.
    if (forceRebind) {
      _reset_bind()
    }

    if (binded) {
      logger.warn("Already binded, ignoring bind()")
      return
    }

    this.forTraining = forTraining
    this.inputsNeedGrad = inputsNeedGrad
    this.binded = true

    require(forTraining || !inputsNeedGrad)
    // this is not True, as some module might not contains a loss function
    // that consumes the labels
    // assert label_shapes is not None

    this.dataShapes = dataShapes
    this.labelShapes = labelShapes

    val sharedGroup =
      if (sharedModule != null) {
        require(sharedModule.binded && sharedModule.paramsInitialized)
        sharedModule.execGroup
      } else {
        null
      }

    val inputTypes = this.dataShapes.map(dataDesc => (dataDesc.name, dataDesc.dtype)).toMap ++
      labelShapes.map(shapes => shapes.map(dataDesc => (dataDesc.name, dataDesc.dtype)).toMap)
                 .getOrElse(Map.empty[String, DType])

    /* TODO
    self._exec_group = DataParallelExecutorGroup(self._symbol, self._context,
      self._work_load_list, self._data_shapes,
      self._label_shapes, self._param_names,
      for_training, inputs_need_grad,
      shared_group, logger = self.logger,
      fixed_param_names = self._fixed_param_names,
      grad_req = grad_req, input_types = input_types)
    */
    if (sharedModule != null) {
      paramsInitialized = true
      argParams = sharedModule.argParams
      auxParams = sharedModule.auxParams
    } else if (paramsInitialized) {
      // if the parameters are already initialized, we are re-binding
      // so automatically copy the already initialized params
      // TODO
      execGroup.setParams(argParams, auxParams)
    }

    if (sharedModule != null && sharedModule.optimizerInitialized) {
      borrowOptimizer(sharedModule)
    }
  }
}
