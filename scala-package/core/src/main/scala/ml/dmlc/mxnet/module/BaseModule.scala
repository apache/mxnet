package ml.dmlc.mxnet.module

import java.io.IOException

import ml.dmlc.mxnet.optimizer.SGD

import scala.collection.immutable.ListMap
import scala.util.control.Breaks._
import ml.dmlc.mxnet._

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
 *    - `for_training`: whether the module is binded for training (if binded).
 *    - `params_initialized`: `bool`, indicating whether the parameters of this modules
 *      has been initialized.
 *    - `optimizer_initialized`: `bool`, indicating whether an optimizer is defined
 *      and initialized.
 *    - `inputs_need_grad`: `bool`, indicating whether gradients with respect to the
 *      input data is needed. Might be useful when implementing composition of modules.
 *
 *  - input/output information
 *    - `data_shapes`: a list of `(name, shape)`. In theory, since the memory is allocated,
 *      we could directly provide the data arrays. But in the case of data parallelization,
 *      the data arrays might not be of the same shape as viewed from the external world.
 *    - `label_shapes`: a list of `(name, shape)`. This might be `[]` if the module does
 *      not need labels (e.g. it does not contains a loss function at the top), or a module
 *      is not binded for training.
 *    - `output_shapes`: a list of `(name, shape)` for outputs of the module.
 *
 *  - parameters (for modules with parameters)
 *    - `get_params()`: return a tuple `(arg_params, aux_params)`. Each of those
 *      is a dictionary of name to `NDArray` mapping. Those `NDArray` always lives on
 *      CPU. The actual parameters used for computing might live on other devices (GPUs),
 *      this function will retrieve (a copy of) the latest parameters. Therefore, modifying
 *    - `set_params(arg_params, aux_params)`: assign parameters to the devices
 *      doing the computation.
 *    - `init_params(...)`: a more flexible interface to assign or initialize the parameters.
 *
 *  - setup
 *    - `bind()`: prepare environment for computation.
 *    - `init_optimizer()`: install optimizer for parameter updating.
 *
 *  - computation
 *    - `forward(data_batch)`: forward operation.
 *    - `backward(out_grads=None)`: backward operation.
 *    - `update()`: update parameters according to installed optimizer.
 *    - `get_outputs()`: get outputs of the previous forward operation.
 *    - `get_input_grads()`: get the gradients with respect to the inputs computed
 *      in the previous backward operation.
 *    - `update_metric(metric, labels)`: update performance metric for the previous forward
 *      computed results.
 *
 *  - other properties (mostly for backward compatability)
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
 * @author Yizhi Liu
 */
abstract class BaseModule {
  var binded: Boolean = false
  var forTraining: Boolean = false
  var inputsNeedGrad: Boolean = false
  val paramsInitialized: Boolean = false
  val optimizerInitialized: Boolean = false
  val symbol: Symbol = null
  val layoutMapper = None

  // High Level API

  // A convenient function that calls both `forward` and `backward`.
  def forwardBackward(dataBatch: DataBatch): Unit = {
    forward(dataBatch, isTrain= Option(true))
    backward()
  }

  /**
   * Run prediction on `eval_data` and evaluate the performance according to `eval_metric`.
   * @param evalData : DataIter
   * @param evalMetric : EvalMetric
   * @param numBatch Number of batches to run. Default is `None`,
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

    // TODO: eval_metric.get_name_value()
    evalMetric
  }

  // Symbol information
  // A list of names for data required by this module.
  def dataNames: IndexedSeq[String]

  // A list of names for the outputs of this module.
  def outputNames: IndexedSeq[String]

  // Input/Output information
  // A list of (name, shape) pairs specifying the data inputs to this module.
  def dataShapes: IndexedSeq[(String, Shape)]

  /**
   * A list of (name, shape) pairs specifying the label inputs to this module.
   * If this module does not accept labels -- either it is a module without loss
   * function, or it is not binded for training, then this should return an empty
   * list `[]`.
   */
  def labelShapes: IndexedSeq[(String, Shape)]

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
                 argParams: Option[Map[String, NDArray]] = None,
                 auxParams: Option[Map[String, NDArray]] = None,
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
    initParams(initializer = null, argParams = Option(argParams), auxParams = Option(auxParams),
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
      argParams.map { case (k, v) => (k, v.asInContext(Context.cpu())) }
      ++ auxParams.map { case (k, v) => (k, v.asInContext(Context.cpu())) }
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
        case Array(argType, name) if argType == "arg " => argParams.put(name, value)
        case Array(argType, name) if argType == "aux " => auxParams.put(name, value)
        case _ => throw new IOException("Invalid param file " + fname)
      }
    }
    setParams(argParams.toMap, auxParams.toMap)
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
   * @param gradReqs Requirement for gradient accumulation (for each argument)
   *                 Can be 'write', 'add', or 'null' (default to 'write').
   */
  def bind(dataShapes: ListMap[String, Shape], labelShapes: Option[ListMap[String, Shape]] = None,
           forTraining: Boolean = true, inputsNeedGrad: Boolean = false,
           forceRebind: Boolean = false, sharedModule: Option[BaseModule] = None,
           gradReq: String = "write", gradReqs: Map[String, String] = null): Unit

  // Install and initialize optimizers.
  def initOptimizer(kvstore: String = "local",
                    optimizer: Optimizer = new SGD(learningRate = 0.01f)): Unit
}
