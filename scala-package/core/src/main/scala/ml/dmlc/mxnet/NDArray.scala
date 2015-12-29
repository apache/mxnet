package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.LoggerFactory

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
 * NDArray API of mxnet
 * @author Yizhi Liu, Yuan Tang
 */
object NDArray {
  private val logger = LoggerFactory.getLogger(classOf[NDArray])
  private val functions: Map[String, NDArrayFunction] = _initNdarrayModule()

  // Definition of internal functions.
  // Internal binary function
  private[mxnet] def _binaryNDArrayFunction(funcName: String,
                                            lhs: NDArray,
                                            rhs: NDArray,
                                            out: NDArray = null): NDArray = {
    var output = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    require(output == null || output.writable, "out must be writable")
    function match {
      case BinaryNDArrayFunction(handle: NDArrayHandle, acceptEmptyMutate: Boolean) =>
        if (output == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          output = new NDArray(_newEmptyHandle())
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          Array(lhs.handle.value, rhs.handle.value),
          Array[MXFloat](),
          Array(output.handle.value)))
      case _ => throw new RuntimeException(s"call $funcName as binary function")
    }
    output
  }

  // internal NDArray function
  private[mxnet] def _unaryNDArrayFunction(funcName: String, src: NDArray, out: NDArray = null): NDArray = {
    var output = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    require(output == null || output.writable, "out must be writable")
    function match {
      case UnaryNDArrayFunction(handle: NDArrayHandle, acceptEmptyMutate: Boolean) =>
        if (output == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          output = new NDArray(_newEmptyHandle())
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          Array(src.handle.value),
          Array[MXFloat](),
          Array(output.handle.value)))
      case _ => throw new RuntimeException(s"call $funcName as unary function")
    }
    output
  }

  /**
   * Invoke this function by passing in parameters
   *
   * @param args Positional arguments of input scalars and NDArray
   * @param out NDArray or tuple of NDArray, optional
   *            Output NDArray, used to hold the output result.
   * @return The result NDArray(tuple) of result of computation.
   */
  private[mxnet] def _genericNDArrayFunction(funcName: String,
                                             args: Array[Any],
                                             out: Array[NDArray] = null): Array[NDArray] = {
    var mutateVars = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    function match {
      case GenericNDArrayFunction(handle: FunctionHandle,
                                  acceptEmptyMutate: Boolean,
                                  nMutateVars: Int,
                                  useVarsRange: Range,
                                  scalarRange: Range) =>
        require(mutateVars == null || nMutateVars == mutateVars.length,
          s"expect $nMutateVars in $funcName")
        if (mutateVars == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          mutateVars = Array.fill[NDArray](nMutateVars)(new NDArray(_newEmptyHandle()))
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          useVarsRange.map(args(_).asInstanceOf[NDArray].handle.value).toArray,
          scalarRange.map(args(_).asInstanceOf[MXFloat]).toArray,
          mutateVars.map(_.handle.value).array))
      case _ => throw new RuntimeException(s"call $funcName as generic function")
    }
    mutateVars
  }

  /**
   * Return a new empty handle.
   * Empty handle can be used to hold result
   *
   * @return a new empty ndarray handle
   */
  private def _newEmptyHandle(): NDArrayHandle = {
    val hdl: NDArrayHandle = new NDArrayHandle
    checkCall(_LIB.mxNDArrayCreateNone(hdl))
    hdl
  }

  /**
   * Return a new handle with specified shape and context.
   * Empty handle is only used to hold results
   *
   * @return a new empty ndarray handle
   */
  private def _newAllocHandle(shape: Array[Int], ctx: Context, delayAlloc: Boolean): NDArrayHandle = {
    val hdl = new NDArrayHandle
    checkCall(_LIB.mxNDArrayCreate(
      shape,
      shape.length,
      ctx.deviceTypeid,
      ctx.deviceId,
      if (delayAlloc) 1 else 0,
      hdl))
    hdl
  }

  /**
   * Wait all async operation to finish in MXNet
   * This function is used for benchmark only
   */
  def waitall(): Unit = {
    checkCall(_LIB.mxNDArrayWaitAll())
  }

  // Create a NDArray function from the FunctionHandle.
  private def _makeNdarrayFunction(handle: FunctionHandle): (String, NDArrayFunction) = {
    val NDARRAY_ARG_BEFORE_SCALAR = 1
    val ACCEPT_EMPTY_MUTATE_TARGET = 1 << 2
    // Get the property of NDArray
    val nUsedVars = new MXUintRef
    val nScalars = new MXUintRef
    val nMutateVars = new MXUintRef
    val typeMask = new RefInt
    checkCall(_LIB.mxFuncDescribe(handle, nUsedVars, nScalars, nMutateVars, typeMask))
    val acceptEmptyMutate = (typeMask.value & ACCEPT_EMPTY_MUTATE_TARGET) != 0
    // infer type of the function
    val ndarrayArgBeforeScalar = (typeMask.value & NDARRAY_ARG_BEFORE_SCALAR) != 0
    val useVarsRange: Range =
      if (ndarrayArgBeforeScalar) 0 until nUsedVars.value
      else nScalars.value until (nUsedVars.value + nScalars.value)
    val scalarRange: Range =
      if (ndarrayArgBeforeScalar) nUsedVars.value until (nUsedVars.value + nScalars.value)
      else 0 until nScalars.value
    // Get the information from the function
    val name = new RefString
    val desc = new RefString
    val numArgs = new MXUintRef
    val argNames = ListBuffer[String]()
    val argTypes = ListBuffer[String]()
    val argDescs = ListBuffer[String]()

    checkCall(_LIB.mxFuncGetInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs))
    val paramStr = Base.ctypes2docstring(argNames, argTypes, argDescs)
    val docStr = s"${name.value}\n${desc.value}\n\n$paramStr\n"
    logger.debug("NDArray function defination:\n{}", docStr)
    if (nMutateVars.value == 1 && nUsedVars.value == 2 && nScalars.value == 0) {
      (name.value, BinaryNDArrayFunction(handle, acceptEmptyMutate))
    } else if (nMutateVars.value == 1 && nUsedVars.value == 1 && nScalars.value == 0) {
      (name.value, UnaryNDArrayFunction(handle, acceptEmptyMutate))
    } else {
      (name.value, GenericNDArrayFunction(handle, acceptEmptyMutate, nMutateVars.value, useVarsRange, scalarRange))
    }
  }

  // List and add all the ndarray functions to current module.
  private def _initNdarrayModule(): Map[String, NDArrayFunction] = {
    val functions = ListBuffer[FunctionHandle]()
    checkCall(_LIB.mxListFunctions(functions))
    functions.map(_makeNdarrayFunction).toMap
  }

  /**
   * One hot encoding indices into matrix out.
   * @param indices An NDArray containing indices of the categorical features.
   * @param out The result holder of the encoding.
   * @return Same as out.
   */
  def onehotEncode(indices: NDArray, out: NDArray): NDArray = ???

  /**
   * Create an empty uninitialized new NDArray, with specified shape.
   *
   * @param shape shape of the NDArray.
   * @param ctx The context of the NDArray, default to current default context.
   *
   * @return The created NDArray.
   */
  def empty(shape: Array[Int], ctx: Context=null): NDArray = {
    val context = if (ctx == null) Context.defaultCtx else ctx
    new NDArray(handle = NDArray._newAllocHandle(shape, context, delayAlloc = false))
  }

  def empty(shape: Int *): NDArray = empty(shape.toArray)

  def empty(ctx: Context, shape: Int *): NDArray = empty(shape.toArray, ctx)

  /**
   * Create a new NDArray filled with 0, with specified shape.
   *
   * @param shape shape of the NDArray.
   * @param ctx The context of the NDArray, default to current default context.
   *
   * @return The created NDArray.
   */
  def zeros(shape: Array[Int], ctx: Context=null): NDArray = {
    val arr = empty(shape, ctx)
    arr.set(0f)
    arr
  }

  def zeros(shape: Int *): NDArray = zeros(shape.toArray)

  def zeros(ctx: Context, shape: Int *): NDArray = zeros(shape.toArray, ctx)

  /**
   * Create a new NDArray filled with 1, with specified shape.
   * @param shape shape of the NDArray.
   * @param ctx The context of the NDArray, default to current default context.
   * @return The created NDArray.
   */
  def ones(shape: Array[Int], ctx: Context=null): NDArray = {
    val arr = empty(shape, ctx)
    arr.set(1f)
    arr
  }

  def ones(shape: Int *): NDArray = ones(shape.toArray)

  def ones(ctx: Context, shape: Int *): NDArray = ones(shape.toArray, ctx)

  /**
   * Clip ndarray elements to range (from, to)
   * @param array ndarray to be clipped
   * @param min array min elements
   * @param max array max elements
   * @return a new clipped [[NDArray]]
   */
  def clip(array: NDArray, min: Float, max: Float): NDArray = {
    NDArray._genericNDArrayFunction("clip", Array(array, min, max))(0)
  }

  /**
   * Take sqrt of the src
   * @param src Source input to the function
   * @return new [[NDArray]]
   */
  def sqrt(src: NDArray): NDArray = {
    NDArray._unaryNDArrayFunction("sqrt", src)
  }

  /**
   * Take L2 norm of the src.
   * @param src Source input to the function
   * @return new [[NDArray]] of shape (1,) on the same device
   */
  def norm(src: NDArray): NDArray = {
    NDArray._unaryNDArrayFunction("norm", src)
  }

  // TODO
  def _randomUniform(low: Float, high: Float, out: NDArray) = ???

  def _randomGaussian(mean: Float, stdvar: Float, out: NDArray) = ???


  /**
   * Create a new NDArray that copies content from source_array.
   * @param sourceArr Source data to create NDArray from.
   * @param ctx The context of the NDArray, default to current default context.
   * @return The created NDArray.
   */
  def array(sourceArr: Array[Float], ctx: Context=null): NDArray = ???

  /**
   * Load ndarray from binary file.
   *
   * You can also use pickle to do the job if you only work on python.
   * The advantage of load/save is the file is language agnostic.
   * This means the file saved using save can be loaded by other language binding of mxnet.
   * You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)
   *
   * @param fname
   *     The name of the file.Can be S3 or HDFS address (remember built with S3 support).
   *     Example of fname:
   *     - `s3://my-bucket/path/my-s3-ndarray`
   *     - `hdfs://my-bucket/path/my-hdfs-ndarray`
   *     - `/path-to/my-local-ndarray`
   * @return dict of str->NDArray to be saved
   */
  def load(fname: String): Map[String, NDArray] = ???

  /**
   * Save list of NDArray or dict of str->NDArray to binary file.
   *
   * You can also use pickle to do the job if you only work on python.
   * The advantage of load/save is the file is language agnostic.
   * This means the file saved using save can be loaded by other language binding of mxnet.
   * You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)
   *
   * @param fname
   *     The name of the file.Can be S3 or HDFS address (remember built with S3 support).
   *     Example of fname:
   *     - `s3://my-bucket/path/my-s3-ndarray`
   *     - `hdfs://my-bucket/path/my-hdfs-ndarray`
   *     - `/path-to/my-local-ndarray`
   * @param data dict of str->NDArray
   */
  def save(fname: String, data: Map[String, NDArray]): Unit = ???
}

/**
 * NDArray object in mxnet.
 * NDArray is basic ndarray/Tensor like data structure in mxnet.
 */
class NDArray(val handle: NDArrayHandle, val writable: Boolean = true) {
  override def finalize() = {
    checkCall(_LIB.mxNDArrayFree(handle))
  }

  /**
   * Peform an synchronize copy from the array.
   * @param source The data source we should like to copy from.
   */
  private def syncCopyfrom(source: Array[Float]): Unit = {
    require(source.length == size, "array size do not match the size of NDArray")
    checkCall(_LIB.mxNDArraySyncCopyFromCPU(handle, source, source.length))
  }

  /**
   * Return a sliced NDArray that shares memory with current one.
   * NDArray only support continuous slicing on axis 0
   *
   * @param start Starting index of slice.
   * @param stop Finishing index of slice.
   *
   * @return a sliced NDArray that shares memory with current one.
   */
  def slice(start: Int, stop: Int): NDArray = {
    val sliceHandle = new NDArrayHandle()
    checkCall(_LIB.mxNDArraySlice(handle, start, stop, sliceHandle))
    new NDArray(handle = sliceHandle, writable = this.writable)
  }

  def slice(start: Int): NDArray = {
    slice(start, shape(0))
  }

  /**
   * Block until all pending writes operations on current NDArray are finished.
   * This function will return when all the pending writes to the current
   * NDArray finishes. There can still be pending read going on when the
   * function returns.
   */
  def waitToRead(): Unit = ???

  /**
   * Get context of current NDArray.
   * @return The context of current NDArray.
   */
  def context: Context = ???

  /**
   * Set the values of the NDArray
   * @param value Value to set
   * @return Current NDArray
   */
  def set(value: Float): NDArray = {
    require(writable, "trying to assign to a readonly NDArray")
    NDArray._genericNDArrayFunction("_set_value", Array[Any](value), out=Array(this))
    this
  }

  def set(other: NDArray): NDArray = {
    require(writable, "trying to assign to a readonly NDArray")
    other.copyTo(this)
  }

  def set(other: Array[Float]): NDArray = {
    require(writable, "trying to assign to a readonly NDArray")
    syncCopyfrom(other)
    this
  }

  def +(other: NDArray): NDArray = {
    NDArray._binaryNDArrayFunction("_plus", this, other)
  }

  def +(other: Float): NDArray = {
    NDArray._genericNDArrayFunction("_plus_scalar", Array[Any](this, other))(0)
  }

  def +=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray._binaryNDArrayFunction("_plus", this, other, out=this)
  }

  def +=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray._genericNDArrayFunction("_plus_scalar", Array[Any](this, other), out=Array(this))
    this
  }

  def -(other: NDArray): NDArray = {
    NDArray._binaryNDArrayFunction("_minus", this, other)
  }

  def -(other: Float): NDArray = {
    NDArray._genericNDArrayFunction("_minus_scalar", Array[Any](this, other))(0)
  }

  def -=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray._binaryNDArrayFunction("_minus", this, other, out=this)
  }

  def -=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray._genericNDArrayFunction("_minus_scalar", Array[Any](this, other), out=Array(this))
    this
  }

  def *(other: NDArray): NDArray = {
    NDArray._binaryNDArrayFunction("_mul", this, other)
  }

  def *(other: Float): NDArray = {
    NDArray._genericNDArrayFunction("_mul_scalar", Array[Any](this, other))(0)
  }

  def unary_-(): NDArray = {
    NDArray._genericNDArrayFunction("_mul_scalar", Array[Any](this, -1f))(0)
  }

  def *=(other: NDArray) = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray._binaryNDArrayFunction("_mul", this, other, out=this)
  }

  def *=(other: Float) = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray._genericNDArrayFunction("_mul_scalar", Array[Any](this, other), out=Array(this))
    this
  }

  def /(other: NDArray): NDArray = {
    NDArray._binaryNDArrayFunction("_div", this, other)
  }

  def /(other: Float): NDArray = {
    NDArray._genericNDArrayFunction("_div_scalar", Array[Any](this, other))(0)
  }

  def /=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray._binaryNDArrayFunction("_div", this, other, out=this)
  }

  def /=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray._genericNDArrayFunction("_div_scalar", Array[Any](this, other), out=Array(this))
    this
  }

  /**
   * Return a copied flat java array of current array.
   * @return  A copy of array content.
   */
  // TODO: Shall we use column-major or row-major ?
  def toArray: Array[Float] = {
    val data = Array.ofDim[Float](size)
    checkCall(_LIB.mxNDArraySyncCopyToCPU(handle, data, size))
    data
  }

  /**
   * Return a CPU scalar(float) of current ndarray.
   * This ndarray must have shape (1,)
   *
   * @return The scalar representation of the ndarray.
   */
  def toScalar: Float = {
    require(shape.sameElements(Array(1)), "The current array is not a scalar")
    this.toArray(0)
  }

  /**
   * Copy the content of current array to other.
   *
   * @param other Target NDArray or context we want to copy data to.
   * @return The copy target NDArray
   */
  def copyTo(other: NDArray): NDArray = ???

  /**
   * Copy the content of current array to a new NDArray in the context.
   *
   * @param ctx Target context we want to copy data to.
   * @return The copy target NDArray
   */
  def copyTo(ctx: Context): NDArray = ???

  /**
   * Get shape of current NDArray.
   * @return an array representing shape of current ndarray
   */
  def shape: Array[Int] = {
    val ndim = new MXUintRef
    val data = ArrayBuffer[Int]()
    checkCall(_LIB.mxNDArrayGetShape(handle, ndim, data))
    require(ndim.value == data.length, s"ndim=$ndim, while len(pdata)=${data.length}")
    data.toArray
  }

  // Get size of current NDArray.
  def size: Int = shape.product
}

object NDArrayConversions {
  implicit def int2Scalar(x: Int): NDArrayConversions = new NDArrayConversions(x.toFloat)
  implicit def double2Scalar(x: Double): NDArrayConversions = new NDArrayConversions(x.toFloat)
  implicit def float2Scalar(x: Float): NDArrayConversions = new NDArrayConversions(x)
}

class NDArrayConversions(val value: Float) {
  def +(other: NDArray): NDArray = {
    other + value
  }

  def -(other: NDArray): NDArray = {
    NDArray._genericNDArrayFunction("_rminus_scalar", Array[Any](other, value))(0)
  }

  def *(other: NDArray): NDArray = {
    other * value
  }

  def /(other: NDArray): NDArray = {
    NDArray._genericNDArrayFunction("_rdiv_scalar", Array[Any](other, value))(0)
  }
}

sealed class NDArrayFunction
case class BinaryNDArrayFunction(handle: FunctionHandle,
                                 acceptEmptyMutate: Boolean) extends NDArrayFunction
case class UnaryNDArrayFunction(handle: FunctionHandle,
                                acceptEmptyMutate: Boolean) extends NDArrayFunction
case class GenericNDArrayFunction(handle: FunctionHandle,
                                  acceptEmptyMutate: Boolean,
                                  nMutateVars: Int,
                                  useVarsRange: Range,
                                  scalarRange: Range) extends NDArrayFunction
