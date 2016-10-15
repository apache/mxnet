package ml.dmlc.mxnet.examples.neuralstyle.end2end

import ml.dmlc.mxnet.Context
import org.slf4j.LoggerFactory
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Optimizer
import ml.dmlc.mxnet.Executor
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Uniform
import ml.dmlc.mxnet.Initializer
import ml.dmlc.mxnet.DataBatch

/**
 * @author Depeng Liang
 */
class Module(symbol: Symbol,
             context: Context,
             dataShapes: Map[String, Shape],
             labelShapes: Map[String, Shape] = Map[String, Shape](),
             initializer: Initializer = new Uniform(0.01f),
             forTraining: Boolean = true,
             inputsNeedGrad: Boolean = false) {

  private val logger = LoggerFactory.getLogger(classOf[Module])

  private val dataLabelShape = dataShapes ++ labelShapes
  private val (argDict, gradDict, auxDict) = {
    val (argShapes, outShapes, auxShapes) = symbol.inferShape(dataLabelShape)
    val argNames = symbol.listArguments()
    val argDict = argNames.zip(argShapes.map(NDArray.empty(_, context))).toMap

    val filterShapes = if (inputsNeedGrad) labelShapes else dataLabelShape
    val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
      !filterShapes.contains(name)
    }.map(x => x._1 -> NDArray.empty(x._2, context) ).toMap

    val auxDict = symbol.listAuxiliaryStates().zip(auxShapes.map(NDArray.empty(_, context))).toMap

    (argDict, gradDict, auxDict)
  }

  private val dataArrs = dataShapes.keys.toArray.map(argDict(_))
  private val labelArrs = labelShapes.keys.toArray.map(argDict(_))
  private val dataGrads = {
    if (inputsNeedGrad) dataShapes.keys.toArray.map(gradDict(_))
    else null
  }

  argDict.foreach { case (name, ndArray) =>
    if (!dataLabelShape.contains(name)) initializer(name, ndArray)
  }

  private val executor = symbol.bind(context, argDict, gradDict, "write", auxDict, null, null)

  private var optimizer: Optimizer = null
  private var paramsGrads: List[(Int, String, NDArray, AnyRef)] = null
  private var optimizerInitialized: Boolean = false

  def initOptimizer(opt: Optimizer): Unit = {
    this.optimizer = opt
    this.paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
      (idx, name, grad, this.optimizer.createState(idx, argDict(name)))
    }
    this.optimizerInitialized = true
  }

  def forward(datas: Array[NDArray], labels: Array[NDArray] = Array[NDArray]()): Unit = {
    datas.zip(this.dataArrs).foreach { case (src, dest) => dest.set(src) }
    labels.zip(this.labelArrs).foreach { case (src, dest) => dest.set(src) }
    this.executor.forward(isTrain = forTraining)
  }

  def backward(outGrads: Array[NDArray]): Unit = {
    this.executor.backward(outGrads)
  }

  def update(): Unit = {
    assert(this.optimizerInitialized)
    paramsGrads.foreach { case (idx, name, grad, optimState) =>
      this.optimizer.update(idx, argDict(name), grad, optimState)
    }
  }

  def dispose(): Unit = {
    this.executor.dispose()
    this.argDict.foreach(_._2.dispose())
    this.gradDict.foreach(_._2.dispose())
    this.auxDict.foreach(_._2.dispose())
  }

  def setParams(params: Map[String, NDArray]): Unit = {
    params.foreach { case (name, arr) =>
      if (this.argDict.contains(name)) {
        this.argDict(name).set(arr)
      }
      else if (this.auxDict.contains(name)) {
        this.auxDict(name).set(arr)
      }
      else logger.info(name)
    }
  }

  def loadParams(fName: String): Unit = {
    val saveDict = NDArray.load2Map(fName)
    var params = Map[String, NDArray]()
    saveDict.foreach { case (k, v) =>
      val (argType, name) = {
        val tmp = k.split(":")
        (tmp(0), tmp(1))
      }
      if (argType == "arg" || argType == "aux") {
        params += name -> v
      }
    }
    this.setParams(params)
  }

  def saveParams(fName: String): Unit = {
    val saveDict = {
      argDict.filter(x => !dataLabelShape.contains(x._1))
      .map { case (k, v) => s"arg:$k" -> v } ++
      auxDict.map { case (k, v) => s"aux:$k" -> v }
    }
    NDArray.save(fName, saveDict)
  }

  def getOutputs(): Array[NDArray] = this.executor.outputs

  def getInputGrads(): Array[NDArray] = this.dataGrads
}
