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

import ml.dmlc.mxnet.DType.DType
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.module.DataParallelExecutorGroup.Builder
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

private object DataParallelExecutorGroup {
  private val logger: Logger = LoggerFactory.getLogger(classOf[DataParallelExecutorGroup])
  // Load a list of arrays into a list of arrays specified by slices
  private def loadGeneralMulti(data: Seq[NDArray],
                               targets: Seq[Array[((Int, Int), NDArray)]],
                               majorAxis: Seq[Int]): Unit = {
    for (((dSrc, dTargets), axis) <- data zip targets zip majorAxis) {
      for (((sliceIdxStart, sliceIdxStop), dDst) <- dTargets) {
        if (axis >= 0) {
          // copy slice
          val shape = dSrc.shape
          val begin = Array.fill(shape.length)(0)
          val end = shape.toArray
          begin(axis) = sliceIdxStart
          end(axis) = sliceIdxStop
          if (dSrc.context == dDst.context) {
            NDArray.crop(Map(
              "begin" -> new Shape(begin),
              "end" -> new Shape(end),
              "out" -> dDst))(dSrc)
          } else {
            // on different device, crop and then do cross device copy
            val dDstCopy: NDArray = NDArray.crop(Map(
              "begin" -> new Shape(begin),
              "end" -> new Shape(end)))(dSrc)
            dDstCopy.copyTo(dDst)
          }
        } else {
          dSrc.copyTo(dDst)
        }
      }
    }
  }

  private def loadGeneral(data: Seq[NDArray], targets: Seq[NDArray]): Unit = {
    for ((dSrc, dTarget) <- data zip targets) {
      dSrc.copyTo(dTarget)
    }
  }

  // Load data into sliced arrays
  private def loadData(batch: DataBatch,
                       targets: Seq[Array[((Int, Int), NDArray)]],
                       majorAxis: Seq[Int]): Unit = {
    loadGeneralMulti(batch.data, targets, majorAxis)
  }


  // Load label into sliced arrays
  private def loadLabel(batch: DataBatch,
                        targets: Seq[Array[((Int, Int), NDArray)]],
                        majorAxis: Seq[Int]): Unit = {
    loadGeneralMulti(batch.label, targets, majorAxis)
  }

  // Merge outputs that lives on multiple context into one,
  // so that they look like living on one context.
  private def mergeMultiContext(outputs: IndexedSeq[IndexedSeq[NDArray]], majorAxis: Seq[Int])
    : IndexedSeq[NDArray] = {
    (outputs zip majorAxis).map { case (tensors, axis) =>
      if (axis >= 0) {
        NDArray.concatenate(tensors, axis = axis, alwaysCopy = false)
      } else {
        // negative axis means the there is no batch_size axis, and all the
        // results should be the same on each device. We simply take the first one,
        // without checking they are actually the same
        tensors(0)
      }
    }
  }

  private object Builder {
    private[module] def convertGradReq(
        gradReq: String, argNames: IndexedSeq[String], paramNames: IndexedSeq[String],
        fixedParamNames: Set[String], dataNames: Seq[String], inputsNeedGrad: Boolean)
        : Map[String, String] = {
      require(argNames != null)
      require(paramNames != null)
      require(fixedParamNames != null)
      require(dataNames != null)
      argNames.map(k => {
        if (paramNames.contains(k)) {
          (k, if (fixedParamNames.contains(k)) "null" else gradReq)
        } else if (dataNames.contains(k)) {
          (k, if (inputsNeedGrad) gradReq else "null")
        } else {
          (k, "null")
        }
      }).toMap
    }
  }

  class Builder private[module](private val symbol: Symbol,
                                private val contexts: Array[Context],
                                private val paramNames: IndexedSeq[String]) {

    private var workLoadList: IndexedSeq[Float] = null
    private var dataShapes: IndexedSeq[DataDesc] = null
    private var labelShapes: Option[IndexedSeq[DataDesc]] = None
    private var forTraining: Boolean = true
    private var inputsNeedGrad: Boolean = false
    private var sharedGroup: Option[DataParallelExecutorGroup] = None
    private var inputTypes: Option[Map[String, DType]] = None
    private var fixedParamNames: Set[String] = Set.empty[String]
    private var gradReqs: Map[String, String] = null

    val argNames = symbol.listArguments()

    def setWorkLoadList(workLoad: IndexedSeq[Float]): Builder = {
      this.workLoadList = workLoad
      this
    }

    def setDataShapes(shapes: IndexedSeq[DataDesc]): Builder = {
      require(shapes != null)
      this.dataShapes = shapes
      this
    }

    def setDataShapesByName(shapes: IndexedSeq[(String, Shape)]): Builder = {
      require(shapes != null)
      this.dataShapes = shapes.map { case (k, s) => new DataDesc(k, s) }
      this
    }

    def setLabelShapes(shapes: IndexedSeq[DataDesc]): Builder = {
      this.labelShapes = Option(shapes)
      this
    }

    def setLabelShapesByName(shapes: IndexedSeq[(String, Shape)]): Builder = {
      this.labelShapes = Option(shapes).map(shapesInst =>
        shapesInst.map { case (k, s) => new DataDesc(k, s) }
      )
      this
    }

    def setForTraining(forTraining: Boolean): Builder = {
      this.forTraining = forTraining
      this
    }

    def setInputsNeedGrad(needGrad: Boolean): Builder = {
      this.inputsNeedGrad = needGrad
      this
    }

    def setSharedGroup(sharedGroup: DataParallelExecutorGroup): Builder = {
      this.sharedGroup = Option(sharedGroup)
      this
    }

    def setInputTypes(inputTypes: Map[String, DType]): Builder = {
      this.inputTypes = Option(inputTypes)
      this
    }

    def setFixedParamNames(fixedParamNames: Set[String]): Builder = {
      this.fixedParamNames = Option(fixedParamNames).getOrElse(Set.empty[String])
      this
    }

    def setGradReq(gradReq: Map[String, String]): Builder = {
      require(dataShapes != null)
      val gradReqTmp = mutable.HashMap.empty[String, String]
      val dataNames = dataShapes.map(_.name)
      for (k <- argNames) {
        if (paramNames.contains(k)) {
          gradReqTmp.put(k, if (fixedParamNames.contains(k)) "null" else "write")
        } else if (dataNames.contains(k)) {
          gradReqTmp.put(k, if (inputsNeedGrad) "write" else "null")
        } else {
          gradReqTmp.put(k, "null")
          gradReqTmp ++= gradReq
        }
      }
      this.gradReqs = gradReqTmp.toMap
      this
    }

    def setGradReq(gradReq: String): Builder = {
      require(dataShapes != null)
      val dataNames = dataShapes.map(_.name)
      this.gradReqs = Builder.convertGradReq(
        gradReq, argNames, paramNames, fixedParamNames, dataNames, inputsNeedGrad)
      this
    }

    def setGradReq(gradReq: Seq[(String, String)]): Builder = {
      require(gradReq.size == argNames.size)
      this.gradReqs = gradReq.toMap
      this
    }

    def build(): DataParallelExecutorGroup = {
      new DataParallelExecutorGroup(
        symbol, contexts, workLoadList, dataShapes, labelShapes, paramNames, forTraining,
        inputsNeedGrad, sharedGroup, inputTypes, fixedParamNames, this.gradReqs)
    }
  }
}

/**
 * DataParallelExecutorGroup is a group of executors that lives on a group of devices.
 * This is a helper class used to implement data parallelism. Each mini-batch will
 * be split and run on the devices.
 * @param symbol The common symbolic computation graph for all executors.
 * @param contexts A list of contexts.
 * @param workLoadList If not `None`, could be a list of numbers that
 *                     specify the workload to be assigned to different context.
 *                     Larger number indicate heavier workload.
 * @param dataShapes Should be a list of (name, shape) tuples, for the shapes of data.
 *                   Note the order is important and should be the same as the order that
 *                   the `DataIter` provide the data.
 * @param labelShapes Should be a list of (name, shape) tuples, for the shapes of label.
 *                    Note the order is important and should be the same as the order that
 *                    the `DataIter` provide the label.
 * @param paramNames A list of strings, indicating the names of parameters
 *                   (e.g. weights, filters, etc.) in the computation graph.
 * @param forTraining Indicate whether the executors should be bind for training.
 *                    When not doing training, the memory for gradients will not be allocated.
 * @param inputsNeedGrad Indicate whether the gradients for the input data should be computed.
 *                       This is currently not used.
 *                       It will be useful for implementing composition of modules.
 * @param sharedGroup Default is `None`. This is used in bucketing. When not `None`,
 *                    it should be a executor group corresponding to a different bucket.
 *                    In other words, it will correspond to a different symbol but
 *                    with the same set of parameters (e.g. unrolled RNNs with different lengths).
 *                    In this case, many memory will be shared.
 * @param inputTypes Default is `None`. When not `None`,
 *                   can be used to specify the data type for each of the data/label inputs.
 * @param fixedParamNames Indicate parameters to be fixed during training.
 *                        Parameters in this list will not allocate space for gradient,
 *                        nor do gradient calculation.
 * @param gradReq Requirement for gradient accumulation. Can be 'write', 'add', or 'null',
 *                be specified for each argument.
 */
class DataParallelExecutorGroup private[module](
    symbol: Symbol,
    contexts: Array[Context],
    workLoadList: IndexedSeq[Float],
    dataShapes: IndexedSeq[DataDesc],
    labelShapes: Option[IndexedSeq[DataDesc]] = None,
    private[module] val paramNames: IndexedSeq[String],
    forTraining: Boolean,
    inputsNeedGrad: Boolean,
    sharedGroup: Option[DataParallelExecutorGroup] = None,
    inputTypes: Option[Map[String, DType]] = None,
    fixedParamNames: Set[String] = Set.empty[String],
    gradReq: Map[String, String] = null) {

  require(symbol != null)
  require(contexts != null)

  private val argNames = symbol.listArguments()
  private val auxNames = symbol.listAuxiliaryStates()

  private val gradReqRun =
    if (!forTraining) {
      val dataNames = dataShapes.map(_.name)
      Builder.convertGradReq("null",
        argNames, paramNames, fixedParamNames, dataNames, inputsNeedGrad)
    } else {
      gradReq
    }

  private val sharedDataArrays: Array[mutable.Map[String, NDArray]] =
    sharedGroup.map(_.sharedDataArrays).getOrElse(
    Array.fill(contexts.length)(mutable.Map.empty[String, NDArray]))

  private var batchSize: Int = -1
  private var slices: Array[(Int, Int)] = null
  private var _defaultExecs: Array[Executor] = null
  private var execs: Array[Executor] = null
  private var dataArrays: Seq[Array[((Int, Int), NDArray)]] = null
  private var labelArrays: Option[Seq[Array[((Int, Int), NDArray)]]] = None
  private[module] var paramArrays: IndexedSeq[Array[NDArray]] = null
  private[module] var gradArrays: IndexedSeq[Array[NDArray]] = null
  private[module] var auxArrays: IndexedSeq[Array[NDArray]] = null
  private var inputGradArrays: IndexedSeq[Array[NDArray]] = null

  private var dataLayouts = decideSlices(dataShapes)
  private var labelLayouts =
    // call it to make sure labels has the same batch size as data
    if (labelShapes != None) decideSlices(labelShapes.get)
    else null

  private val outputLayouts = symbol.listOutputs().map(name => {
    val sym = symbol.get(name)
    val layout = sym.attr("__layout__")
    sym.dispose()
    DataDesc.getBatchAxis(layout)
  }
  )
  bindExec(dataShapes, labelShapes, sharedGroup)

  def getBatchSize: Int = batchSize

  /**
   * Decide the slices for each context according to the workload.
   * @param dataShapes list of DataDesc(name, shape) specifying
   *                   the shapes for the input data or label.
   */
  private def decideSlices(dataShapes: Seq[DataDesc]): Seq[Int] = {
    require(dataShapes.size > 0)
    val majorAxis = dataShapes.map(data => DataDesc.getBatchAxis(Option(data.layout)))

    for ((dataDesc, axis) <- dataShapes.zip(majorAxis)) {
      if (axis != -1) {
        val batchSize = dataDesc.shape(axis)
        if (this.batchSize != -1) {
          require(batchSize == this.batchSize,
            s"all data must have the same batch size: $batchSize," +
            s"but ${dataDesc.name} has shape ${dataDesc.shape}")
        } else {
          this.batchSize = batchSize
          require(this.workLoadList != null)
          this.slices = ExecutorManager.splitInputSlice(this.batchSize, this.workLoadList)
        }
      }
    }
    majorAxis
  }

  /**
   * Bind executors on their respective devices.
   * @param dataShapes DataDesc for input data.
   * @param labelShapes DataDesc for input labels.
   * @param sharedGroup
   * @param reshape
   */
  def bindExec(dataShapes: Seq[DataDesc], labelShapes: Option[Seq[DataDesc]],
               sharedGroup: Option[DataParallelExecutorGroup], reshape: Boolean = false): Unit = {
    this.batchSize = -1
    dataLayouts = decideSlices(dataShapes)
    labelLayouts = {
      // call it to make sure labels has the same batch size as data
      if (labelShapes != None) decideSlices(labelShapes.get)
      else null
    }
    if (reshape) {
      (0 until contexts.length).foreach { i =>
        val dataShapesSliced = slicedShape(dataShapes, i, dataLayouts)
        val labelShapesSliced = labelShapes.map(slicedShape(_, i, labelLayouts))
        val inputShapes
          = dataShapesSliced.toMap ++ labelShapesSliced.getOrElse(Map.empty[String, Shape])
        execs(i) = _defaultExecs(i).reshape(allowUpSizing = true, kwargs = inputShapes)
      }
    } else {
      execs = (0 until contexts.length).map(i =>
        bindIthExec(i, dataShapes, labelShapes, sharedGroup)
      ).toArray
    }

    // convenient data structures
    dataArrays = dataShapes.map(dataDesc =>
      this.execs.zipWithIndex.map { case (e, i) => (this.slices(i), e.argDict(dataDesc.name)) }
    )

    labelArrays = labelShapes.map(shapes =>
      shapes.map(labelDesc =>
        this.execs.zipWithIndex.map { case (e, i) => (this.slices(i), e.argDict(labelDesc.name)) }
      )
    )

    paramArrays = argNames.zipWithIndex.withFilter {
      case (name, i) => paramNames.contains(name)
    }.map { case (name, i) =>
      execs.map(_.argArrays(i))
    }

    gradArrays =
      if (forTraining) {
        argNames.zipWithIndex.withFilter {
          case (name, i) => paramNames.contains(name)
        }.map { case (name, i) =>
          execs.map(_.gradArrays(i))
        }
      } else {
        null
      }

    val dataNames = dataShapes.map(_.name)
    inputGradArrays =
      if (inputsNeedGrad) {
        argNames.zipWithIndex.withFilter {
          case (name, i) => dataNames.contains(name)
        }.map { case (name, i) =>
          execs.map(_.gradArrays(i))
        }
      } else {
        null
      }

    auxArrays = (0 until auxNames.length).map(i => execs.map(_.auxArrays(i)))
  }

  /**
   * Reshape executors.
   * @param dataShapes
   * @param labelShapes
   */
  def reshape(dataShapes: Seq[DataDesc], labelShapes: Option[Seq[DataDesc]]): Unit = {
    if (!(dataShapes == this.dataShapes && labelShapes == this.labelShapes)) {
      if (this._defaultExecs == null) {
        this._defaultExecs = this.execs.map(x => x)
      }
      this.bindExec(dataShapes, labelShapes, None, reshape = true)
    }
  }

  /**
   * Assign, i.e. copy parameters to all the executors.
   * @param argParams A dictionary of name to `NDArray` parameter mapping.
   * @param auxParams A dictionary of name to `NDArray` auxiliary variable mapping.
   * @param allowExtra hether allow extra parameters that are not needed by symbol.
   *         If this is True, no error will be thrown when argParams or auxParams
   *         contain extra parameters that is not needed by the executor.
   */
  def setParams(argParams: Map[String, NDArray], auxParams: Map[String, NDArray],
    allowExtra: Boolean = false): Unit = {
    execs.foreach(_.copyParamsFrom(argParams, auxParams, allowExtraParams = allowExtra))
  }

  /**
   * Copy data from each executor to `arg_params` and `aux_params`.
   * @param argParams target parameter arrays
   * @param auxParams target aux arrays
   * Note this function will inplace update the NDArrays in arg_params and aux_params.
   */
  def getParams(argParams: Map[String, NDArray], auxParams: Map[String, NDArray]): Unit = {
    for ((name, block) <- paramNames.zip(paramArrays)) {
      val weight = (block.map(_.copyTo(Context.cpu())).reduce((a: NDArray, b: NDArray) =>
        (a + b).disposeDeps()
      ) / block.length).disposeDeps()
      val weightNewType = weight.asType(argParams(name).dtype)
      weightNewType.copyTo(argParams(name))
      weight.dispose()
      weightNewType.dispose()
    }
    for ((name, block) <- auxNames.zip(auxArrays)) {
      val weight = (block.map(_.copyTo(Context.cpu())).reduce((a: NDArray, b: NDArray) =>
        (a + b).disposeDeps()
      ) / block.length).disposeDeps()
      val weightNewType = weight.asType(auxParams(name).dtype)
      weightNewType.copyTo(auxParams(name))
      weight.dispose()
      weightNewType.dispose()
    }
  }

  /**
   * Split `dataBatch` according to workload and run forward on each devices.
   * @param dataBatch
   * @param isTrain The hint for the backend, indicating whether we are during training phase.
   *                Default is `None`, then the value `self.for_training` will be used.
   */
  def forward(dataBatch: DataBatch, isTrain: Option[Boolean] = None): Unit = {
    DataParallelExecutorGroup.loadData(dataBatch, dataArrays, dataLayouts)
    val isTrainOpt = isTrain.getOrElse(this.forTraining)
    labelArrays.foreach(labels => {
      require(!isTrainOpt || dataBatch.label != null)
      if (dataBatch.label != null) {
        require(labelLayouts != null)
        DataParallelExecutorGroup.loadLabel(dataBatch, labels, labelLayouts)
      }
    })
    execs.foreach(_.forward(isTrainOpt))
  }

  // Get the shapes of the outputs.
  def getOutputShapes: IndexedSeq[(String, Shape)] = {
    val outputs = execs(0).outputs
    val shapes = outputs.map(_.shape)
    (symbol.listOutputs() zip shapes zip outputLayouts) map { case ((key, theShape), axis) =>
      val shape = theShape.toArray
      if (axis >= 0) {
        shape(axis) = batchSize
      }
      (key, Shape(shape))
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
    (0 until execs(0).outputs.length).map(i => execs.map(_.outputs(i)).toIndexedSeq)
  }

  /**
   * Get outputs of the previous forward computation.
   * @return In the case when data-parallelism is used,
   *         the outputs will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[out1, out2]`
   */
  def getOutputsMerged(): IndexedSeq[NDArray] = {
    DataParallelExecutorGroup.mergeMultiContext(getOutputs(), outputLayouts)
  }

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be collected from multiple devices.
   *         The results will look like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`,
   *         those `NDArray` might live on different devices.
   */
  def getInputGrads(): IndexedSeq[IndexedSeq[NDArray]] = {
    require(inputsNeedGrad)
    inputGradArrays.map(_.toIndexedSeq)
  }

  /**
   * Get the gradients to the inputs, computed in the previous backward computation.
   * @return In the case when data-parallelism is used,
   *         the grads will be merged from multiple devices,
   *         as they look like from a single executor.
   *         The results will look like `[grad1, grad2]`
   */
  def getInputGradsMerged(): IndexedSeq[NDArray] = {
    DataParallelExecutorGroup.mergeMultiContext(getInputGrads(), dataLayouts)
  }

  /**
   * Run backward on all devices. A backward should be called after
   * a call to the forward function. Backward cannot be called unless
   * `this.for_training` is `True`.
   * @param outGrads Gradient on the outputs to be propagated back.
   *                 This parameter is only needed when bind is called
   *                 on outputs that are not a loss function.
   */
  def backward(outGrads: Array[NDArray] = null): Unit = {
    require(forTraining, "re-bind with forTraining = true to run backward")

    for (((exec, islice), i) <- (execs zip slices).zipWithIndex) {
      val outGradsSlice =
        if (outGrads != null) {
          (outGrads zip outputLayouts).map { case (grad, axis) =>
            if (axis >= 0) {
              val ogMySlice: NDArray = NDArray.slice_axis(
                Map("axis" -> axis, "begin" -> islice._1, "end" -> islice._2))(grad)
              ogMySlice.asInContext(contexts(i))
            } else {
              grad.copyTo(contexts(i))
            }
          }
        } else {
          Array.empty[NDArray]
        }
      exec.backward(outGrads = outGradsSlice)
    }
  }

  /**
   * Accumulate the performance according to `eval_metric` on all devices.
   * @param evalMetric The metric used for evaluation.
   * @param labels Typically comes from `label` of a `DataBatch`.
   */
  def updateMetric(evalMetric: EvalMetric, labels: IndexedSeq[NDArray]): Unit = {
    for ((texec, islice) <- this.execs zip this.slices) {
      val labelsSlice =
        (labels zip this.labelLayouts) map { case (label, axis) =>
          if (axis == 0) {
            label.slice(islice)
          } else if (axis > 0) {
            val labelMySlice: NDArray = NDArray.slice_axis(Map(
              "axis" -> axis, "begin" -> islice._1, "end" -> islice._2))(label)
              .asInContext(label.context)
            labelMySlice
          } else {
            label
          }
        }

      evalMetric.update(labelsSlice, texec.outputs)

      // Clear up any slices we created (sometimes we don't slice so check for this)
      (labels zip labelsSlice).foreach { case (label, labelSlice) =>
        if (label ne labelSlice) {
          labelSlice.dispose()
        }
      }
    }
  }

  // Internal utility function to bind the i-th executor.
  private def bindIthExec(i: Int, dataShapes: Seq[DataDesc],
                          labelShapes: Option[Seq[DataDesc]],
                          sharedGroup: Option[DataParallelExecutorGroup]): Executor = {
    val dataShapesSliced = slicedShape(dataShapes, i, dataLayouts)
    val labelShapesSliced = labelShapes.map(slicedShape(_, i, labelLayouts))
    val sharedExec = sharedGroup.map(_.execs(i))
    val context = contexts(i)
    val sharedDataArrays = this.sharedDataArrays(i)

    val inputShapes
      = dataShapesSliced.toMap ++ labelShapesSliced.getOrElse(Map.empty[String, Shape])

    val (argShapes, _, auxShapes) = symbol.inferShape(inputShapes)
    require(argShapes != null, "shape inference failed")

    val inputTypesGot = inputTypes.getOrElse(inputShapes.map { case (k, v) =>
      (k, Base.MX_REAL_TYPE)
    })
    val (argTypes, _, auxTypes) = symbol.inferType(inputTypesGot)
    require(argTypes != null, "type inference failed")

    val argArrays = ArrayBuffer.empty[NDArray]
    val gradArrayMap = mutable.HashMap.empty[String, NDArray]

    // create or borrow arguments and gradients
    for (j <- 0 until argNames.length) {
      val name = argNames(j)
      val argArr =
        if (paramNames.contains(name)) {
          // model parameter
          sharedExec match {
            case None =>
              val argArr = NDArray.zeros(argShapes(j), context, dtype = argTypes(j))
              if (gradReqRun(name) != "null") {
                val gradArr = NDArray.zeros(argShapes(j), context, dtype = argTypes(j))
                gradArrayMap.put(name, gradArr)
              }
              argArr
            case Some(sharedExecInst) =>
              val argArr = sharedExecInst.argDict(name)
              require(argArr.shape == argShapes(j))
              require(argArr.dtype == argTypes(j))
              if (gradReqRun(name) != "null") {
                gradArrayMap.put(name, sharedExecInst.gradDict(name))
              }
              argArr
          }
        } else {
          // data or label
          val argArr = getOrReshape(name, sharedDataArrays, argShapes(j), argTypes(j), context)
          // data might also need grad if inputs_need_grad is True
          if (gradReqRun(name) != "null") {
            gradArrayMap.put(name,
              getOrReshape(s"grad of $name", sharedDataArrays, argShapes(j), argTypes(j), context))
          }
          argArr
        }
      argArrays.append(argArr)
    }

    // create or borrow aux variables
    val auxArrays =
      sharedExec match {
        case None => (auxShapes zip auxTypes).map { case (s, t) =>
          NDArray.zeros(s, context, dtype = t)
        }.toArray
        case Some(sharedExecInst) =>
          for ((arr, j) <- sharedExecInst.auxArrays.zipWithIndex) {
            require(auxShapes(j) == arr.shape)
            require(auxTypes(j) == arr.dtype)
          }
          sharedExecInst.auxArrays.map(identity)
      }
    symbol.bind(ctx = context, args = argArrays.toSeq, argsGrad = gradArrayMap.toMap,
      gradsReq = gradReqRun, auxStates = auxArrays.toSeq, group2ctx = null,
      sharedExec = sharedExec.orNull)
  }

  /**
   * Get the sliced shapes for the i-th executor.
   * @param shapes : The original (name, shape) pairs.
   * @param i Which executor we are dealing with.
   * @param majorAxis
   */
  private def slicedShape(shapes: Seq[DataDesc], i: Int, majorAxis: Seq[Int])
    : Seq[(String, Shape)] = {
    (shapes zip majorAxis).map { case (DataDesc(k, shape, _ , _), axis) =>
      val shapeArr = shape.toArray
      if (axis >= 0) {
        shapeArr(axis) = slices(i)._2 - slices(i)._1
      }
      (k, Shape(shapeArr))
    }
  }

  // Install monitor on all executors
  def installMonitor(monitor: Monitor): Unit = {
    execs.foreach(monitor.install)
  }

  // Internal helper to get a memory block or re-use by re-shaping
  private def getOrReshape(name: String,
                           sharedDataArrays: mutable.Map[String, NDArray],
                           argShape: Shape,
                           argType: DType,
                           context: Context): NDArray = {
    if (sharedDataArrays.contains(name)) {
      val argArr = sharedDataArrays(name)
      if (argArr.shape.product >= argShape.product) {
        // nice, we can directly re-use this data blob
        require(argArr.dtype == argType)
        argArr.reshape(argShape)
      } else {
        DataParallelExecutorGroup.logger.warn(s"bucketing: data $name has a shape $argShape," +
          s"which is larger than already allocated shape ${argArr.shape}." +
          "Need to re-allocate. Consider putting default_bucket_key to be the bucket" +
          "taking the largest input for better memory sharing.")
        val argArrNew = NDArray.zeros(argShape, context, dtype = argType)
        // replace existing shared array because the new one is bigger
        sharedDataArrays.put(name, argArrNew)
        argArrNew
      }
    } else {
      val argArrNew = NDArray.zeros(argShape, context, dtype = argType)
      sharedDataArrays.put(name, argArrNew)
      argArrNew
    }
  }
}
