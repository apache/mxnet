package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import scala.collection.mutable.ArrayBuffer

/**
 * Base class for operators implemented in Scala
 * @author Depeng Liang
 */
abstract class CustomOp {

  /**
   * forward interface. override to create new operators.
   * @param isTrain : Boolean
   *            whether this is for training
   * @param req : array of String
   *            how to assign to outData. can be 'null', 'write', or 'add'.
   *            You can optionally use this.assign(dst, req, src) to handle this.
   * @param inData, outData, aux : array of NDArrays
   *            input, output, and auxiliary states for forward. See document for
   *            corresponding arguments of Operator::Forward
   */
  def forward(isTrain: Boolean, req: Array[String],
    inData: Array[NDArray], outData: Array[NDArray], aux: Array[NDArray]): Unit

  /**
   * backward interface. override to create new operators
   * @param req : array of String
   *            how to assign to inGrad. can be 'null', 'write', or 'add'.
   *            You can optionally use this.assign(dst, req, src) to handle this.
   * @param outGrad, inData, outData, inGrad, aux : array of NDArrays
   *            input, output, and auxiliary states for backward. See document for
   *            corresponding arguments of Operator::Backward
   */
  def backward(req: Array[String], outGrad: Array[NDArray],
    inData: Array[NDArray], outData: Array[NDArray],
    inGrad: Array[NDArray], aux: Array[NDArray]): Unit

  /**
   * Helper function for assigning into dst depending on requirements.
   */
  def assign(dst: NDArray, req: String, src: NDArray): Unit = req match {
    case "write" | "inplace" => dst.set(src)
    case "add" => dst += src
    case "null" => {}
  }

  /**
   * Scala Callback for CustomOp::Forward
   */
  private[mxnet] def forwardEntry(numNdarray: Int, ndarraies: Array[NDArrayHandle],
    tags: Array[Int], reqs: Array[Int], isTrain: Boolean): Boolean = {
    var success = true
    try {
      val tensors = (0 until 5).toArray.map( x => ArrayBuffer[NDArray]() )
      for (i <- 0 until numNdarray) {
        if (tags(i) == 1 || tags(i) == 4) {
          tensors(tags(i)) += new NDArray(ndarraies(i), writable = true)
        } else {
          tensors(tags(i)) += new NDArray(ndarraies(i), writable = false)
        }
      }
      val reqEnum = Array("null", "write", "inplace", "add")
      val reqsArr = tensors(1).indices.map(i => reqEnum(reqs(i))).toArray
      this.forward(isTrain = isTrain, req = reqsArr,
        inData = tensors(0).toArray, outData = tensors(1).toArray,
        aux = tensors(4).toArray)
    } catch {
      case ex: Throwable => {
        success = false
        ex.printStackTrace()
      }
    }
    success
  }

  /**
   * Scala Callback for CustomOp::Backward
   */
  private[mxnet] def backwardEntry(numNdarray: Int, ndarraies: Array[NDArrayHandle],
    tags: Array[Int], reqs: Array[Int], isTrain: Boolean): Boolean = {
    var success = true
    try {
      val tensors = (0 until 5).toArray.map( x => ArrayBuffer[NDArray]() )
      for (i <- 0 until numNdarray) {
        if (tags(i) == 2 || tags(i) == 4) {
          tensors(tags(i)) += new NDArray(ndarraies(i), writable = true)
        } else {
          tensors(tags(i)) += new NDArray(ndarraies(i), writable = false)
        }
      }
      val reqEnum = Array("null", "write", "inplace", "add")
      val reqsArr = tensors(2).indices.map(i => reqEnum(reqs(i))).toArray
      this.backward(req = reqsArr,
        inData = tensors(0).toArray, outData = tensors(1).toArray,
        inGrad = tensors(2).toArray, outGrad = tensors(3).toArray,
        aux = tensors(4).toArray)
    } catch {
      case ex: Throwable => {
        success = false
        ex.printStackTrace()
      }
    }
    success
  }
}

/**
 * Base class for operator property class implemented in Scala.
 * MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
 * @param needTopGrad : Boolean
 *           The default declareBackwardDependency function use this value
 *            to determine whether this operator needs gradient input for above.
 */
abstract class CustomOpProp(needTopGrad: Boolean = false) {

  protected var kwargs: Map[String, String] = Map[String, String]()

  private[mxnet] def init(keys: Array[String], vals: Array[String]): Unit = {
    require(keys.length == vals.length)
    kwargs = keys.zip(vals).toMap
  }

  /**
   * inferShape interface. override to create new operators
   * @param inShape : array of array
   *           list of argument shapes in the same order as declared in listArguments().
   * @return
   * inShapes : array of array
   *            array of argument shapes. Can be modified from inShape.
   * outShapes : array of array
   *            array of output shapes calculated from inShape,
   *            in the same order as declared in listOutputs().
   * auxShapes : array of array
   *            array of aux shapes calculated from in_shape,
   *            in the same order as declared in listAuxiliaryStates().
   */
  def inferShape(inShape: Array[Shape]):
    (Array[Shape], Array[Shape], Array[Shape])

  /**
   * Scala Callback for CustomOp::InferShape
   */
  private[mxnet] def inferShapeEntry(
    numTensor: Int, intputShapes: Array[Array[Int]]): Array[Array[Int]] = {
    val nIn = this.listArguments().length
    val nOut = this.listOutputs().length
    val nAux = {
      val tmp = this.listAuxiliaryStates()
      if (tmp == null) 0 else tmp.length
    }
    require(numTensor == (nIn + nOut + nAux))
    val (inShapes, outShapes, auxShapes) =
      inferShape(intputShapes.map(Shape(_)))
    require(inShapes != null && inShapes.length != 0)
    require(outShapes != null && outShapes.length != 0)
    if (auxShapes != null && auxShapes.length != 0) {
      inShapes.map(_.toArray) ++ outShapes.map(_.toArray) ++ auxShapes.map(_.toArray)
    } else inShapes.map(_.toArray) ++ outShapes.map(_.toArray)
  }

  /**
   * listOutputs interface. override to create new operators
   * @return
   * outputs : array of String
   *            list of output blob names.
   */
  def listOutputs(): Array[String]

  /**
   * listArguments interface. override to create new operators
   * @return
   * arguments : array of String
   *            list of argument blob names.
   */
  def listArguments(): Array[String]

  /**
   * listAuxiliaryStates interface. override to create new operators
   * @return
   * auxs : array of String
   *            list of auxiliary state blob names.
   */
  def listAuxiliaryStates(): Array[String] = null

  /**
   * Declare dependencies of this operator for backward pass.
   * @param outGrad : array of Int
   *           ids of outGrad blobs.
   * @param inData : array of Int
   *           ids of inData blobs.
   * @param outData : array of Int
   *           ids of outData blobs.
   * @return
   * deps : array of Int
   *            ids of the needed blobs.
   */
  def declareBackwardDependency(outGrad: Array[Int],
    inData: Array[Int], outData: Array[Int]): Array[Int] = {
    val deps = ArrayBuffer[Array[Int]]()
    if (this.needTopGrad) deps += outGrad
    deps += inData
    deps += outData
    deps.toArray.flatten
  }

  /**
   * Create an operator that carries out the real computation
   * given the context, input shapes, and input data types.
   */
  def createOperator(ctx: String, inShapes: Array[Array[Int]], inDtypes: Array[Int]): CustomOp

}

object Operator {
  def register(regName: String, opProp: CustomOpProp): Unit = {
    checkCall(_LIB.mxCustomOpRegister(regName, opProp))
  }
}
