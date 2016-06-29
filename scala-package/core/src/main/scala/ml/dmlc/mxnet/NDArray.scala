package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.ref.WeakReference

/**
 * NDArray API of mxnet
 * @author Yizhi Liu, Yuan Tang
 */
object NDArray {
  private val logger = LoggerFactory.getLogger(classOf[NDArray])

  private[mxnet] val DTYPE_NATIVE_TO_MX: Map[Class[_ >: Float with Int with Double], Int] = Map(
    classOf[Float] -> 0,
    classOf[Double] -> 1,
    classOf[Int] -> 4
  )

  private[mxnet] val DTYPE_MX_TO_NATIVE: Map[Int, Class[_ >: Float with Int with Double]] = Map(
    0 -> classOf[Float],
    1 -> classOf[Double],
    2 -> classOf[Float],
    3 -> classOf[Int],
    4 -> classOf[Int]
  )

  private val functions: Map[String, NDArrayFunction] = initNDArrayModule()

  private def addDependency(froms: Array[NDArray], tos: Array[NDArray]): Unit = {
    froms.foreach { from =>
      val weakRef = new WeakReference(from)
      tos.foreach { to =>
        to.dependencies.put(from.handle, weakRef)
        // we add all dep's dep to prevent (recursively) recomputing at runtime.
        to.dependencies ++= from.dependencies
      }
    }
  }

  // Definition of internal functions.
  // Internal binary function
  def invokeBinaryFunc(funcName: String,
                       lhs: NDArray, rhs: NDArray,
                       out: NDArray = null): NDArray = {
    var output = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    require(output == null || output.writable, "out must be writable")
    function match {
      case BinaryNDArrayFunction(handle: FunctionHandle, acceptEmptyMutate: Boolean) =>
        if (output == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          output = new NDArray(newEmptyHandle())
          addDependency(Array(lhs, rhs), Array(output))
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          Array(lhs.handle, rhs.handle),
          Array[MXFloat](),
          Array(output.handle)))
      case _ => throw new IllegalArgumentException(s"call $funcName as binary function")
    }
    output
  }

  def invokeUnaryFunc(funcName: String, src: NDArray, out: NDArray = null): NDArray = {
    var output = out
    val function = functions(funcName)
    require(function != null, s"invalid function name $funcName")
    require(output == null || output.writable, "out must be writable")
    function match {
      case UnaryNDArrayFunction(handle: NDArrayHandle, acceptEmptyMutate: Boolean) =>
        if (output == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          output = new NDArray(newEmptyHandle())
          addDependency(Array(src), Array(output))
        }
        checkCall(_LIB.mxFuncInvoke(handle,
          Array(src.handle),
          Array[MXFloat](),
          Array(output.handle)))
      case _ => throw new IllegalArgumentException(s"call $funcName as unary function")
    }
    output
  }

  /**
   * Invoke this function by passing in parameters
   *
   * @param args Positional arguments of input scalars and NDArray
   * @param kwargs: Key-value arguments for functions. e.g.,
   *            out: NDArray or tuple of NDArray, optional
   *            Output NDArray, used to hold the output result.
   * @return The result NDArray(tuple) of result of computation.
   */
  def invokeGenericFunc(funcName: String,
                        args: Array[Any] = null,
                        kwargs: Map[String, Any] = null): Array[NDArray] = {
    var mutateVars: Array[NDArray] = null
    val realKwargs =
      if (kwargs != null && kwargs.contains("out")) {
        val out = kwargs("out")
        mutateVars =
          if (out.isInstanceOf[NDArray]) {
            Array(kwargs("out").asInstanceOf[NDArray])
          } else {
            kwargs("out").asInstanceOf[Array[NDArray]]
          }
        kwargs - "out"
      } else {
        kwargs
      }
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
        val useVars = useVarsRange.map(args(_).asInstanceOf[NDArray]).toArray
        val scalarVars = scalarRange.map(args(_).asInstanceOf[MXFloat]).toArray
        if (mutateVars == null) {
          require(acceptEmptyMutate, s"argument out is required to call $funcName")
          mutateVars = Array.fill[NDArray](nMutateVars)(new NDArray(newEmptyHandle()))
          addDependency(useVars, mutateVars)
        }
        val (numKwargs: Int,
              kwargKeys: Option[Array[Array[Byte]]],
              kwargVals: Option[Array[Array[Byte]]]) =
          if (realKwargs == null) {
            (0, None, None)
          } else {
            (realKwargs.size,
              Some(realKwargs.keys.map(_.getBytes("ASCII") ++ Array(0.toByte)).toArray),
              Some(realKwargs.values.map(_.toString.getBytes("ASCII") ++ Array(0.toByte)).toArray))
          }
        checkCall(_LIB.mxFuncInvokeEx(handle,
          useVars.map(_.handle),
          scalarVars,
          mutateVars.map(_.handle).array,
          numKwargs, kwargKeys.orNull, kwargVals.orNull))
      case _ => throw new IllegalArgumentException(s"call $funcName as generic function")
    }
    mutateVars
  }

  /**
   * Return a new empty handle.
   * Empty handle can be used to hold result
   *
   * @return a new empty ndarray handle
   */
  private def newEmptyHandle(): NDArrayHandle = {
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayCreateNone(hdl))
    hdl.value
  }

  /**
   * Return a new handle with specified shape and context.
   * Empty handle is only used to hold results
   *
   * @return a new empty ndarray handle
   */
  private def newAllocHandle(shape: Shape,
                             ctx: Context,
                             delayAlloc: Boolean): NDArrayHandle = {
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayCreate(
      shape.toArray,
      shape.length,
      ctx.deviceTypeid,
      ctx.deviceId,
      if (delayAlloc) 1 else 0,
      hdl))
    hdl.value
  }

  /**
   * Wait all async operation to finish in MXNet
   * This function is used for benchmark only
   */
  def waitall(): Unit = {
    checkCall(_LIB.mxNDArrayWaitAll())
  }

  // Create a NDArray function from the FunctionHandle.
  private def makeNdarrayFunction(handle: FunctionHandle): (String, NDArrayFunction) = {
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
      (name.value, GenericNDArrayFunction(handle,
                                          acceptEmptyMutate,
                                          nMutateVars.value,
                                          useVarsRange,
                                          scalarRange))
    }
  }

  // List and add all the ndarray functions to current module.
  private def initNDArrayModule(): Map[String, NDArrayFunction] = {
    val functions = ListBuffer[FunctionHandle]()
    checkCall(_LIB.mxListFunctions(functions))
    functions.map(makeNdarrayFunction).toMap
  }

  /**
   * One hot encoding indices into matrix out.
   * @param indices An NDArray containing indices of the categorical features.
   * @param out The result holder of the encoding.
   * @return Same as out.
   */
  def onehotEncode(indices: NDArray, out: NDArray): NDArray = {
    NDArray.invokeBinaryFunc("_onehot_encode", indices, out, out)
  }

  /**
   * Create an empty uninitialized new NDArray, with specified shape.
   *
   * @param shape shape of the NDArray.
   * @param ctx The context of the NDArray, default to current default context.
   *
   * @return The created NDArray.
   */
  def empty(shape: Shape, ctx: Context = null): NDArray = {
    val context = if (ctx == null) Context.defaultCtx else ctx
    new NDArray(handle = NDArray.newAllocHandle(shape, context, delayAlloc = false))
  }

  def empty(shape: Int *): NDArray = empty(Shape(shape: _*))

  def empty(ctx: Context, shape: Int *): NDArray = empty(Shape(shape: _*), ctx)

  /**
   * Create a new NDArray filled with 0, with specified shape.
   *
   * @param shape shape of the NDArray.
   * @param ctx The context of the NDArray, default to current default context.
   *
   * @return The created NDArray.
   */
  def zeros(shape: Shape, ctx: Context = null): NDArray = {
    val arr = empty(shape, ctx)
    arr.set(0f)
    arr
  }

  def zeros(shape: Int *): NDArray = zeros(Shape(shape: _*))

  def zeros(ctx: Context, shape: Int *): NDArray = zeros(Shape(shape: _*), ctx)

  /**
   * Create a new NDArray filled with 1, with specified shape.
   * @param shape shape of the NDArray.
   * @param ctx The context of the NDArray, default to current default context.
   * @return The created NDArray.
   */
  def ones(shape: Shape, ctx: Context = null): NDArray = {
    val arr = empty(shape, ctx)
    arr.set(1f)
    arr
  }

  def ones(shape: Int *): NDArray = ones(Shape(shape: _*))

  def ones(ctx: Context, shape: Int *): NDArray = ones(Shape(shape: _*), ctx)

  /**
   * Clip ndarray elements to range (from, to)
   * @param array ndarray to be clipped
   * @param min array min elements
   * @param max array max elements
   * @return a new clipped [[NDArray]]
   */
  def clip(array: NDArray, min: Float, max: Float): NDArray = {
    NDArray.invokeGenericFunc("clip", Array(array, min, max))(0)
  }

  /**
   * Take sqrt of the src
   * @param src Source input to the function
   * @return new [[NDArray]]
   */
  def sqrt(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("sqrt", src)
  }

  /**
   * Take rsqrt of the src
   * @param src Source input to the function
   * @return new [[NDArray]]
   */
  def rsqrt(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("rsqrt", src)
  }

  /**
   * Calculate 2D matrix multiplication
   * @param lhs left ndarray
   * @param rhs right ndarray
   * @return a new [[NDArray]]
   */
  def dot(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.invokeBinaryFunc("dot", lhs, rhs)
  }

  /**
   * Take L2 norm of the src.
   * @param src Source input to the function
   * @return a new [[NDArray]] of shape (1,) on the same device
   */
  def norm(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("norm", src)
  }

  /**
   * Take absolute value of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def abs(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("abs", src)
  }

  /**
   * Take sign value of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def sign(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("sign", src)
  }

  /**
   * Take round value of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def round(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("round", src)
  }

  /**
   * Take ceil value of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def ceil(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("ceil", src)
  }

  /**
   * Take floor value of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def floor(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("floor", src)
  }

  /**
   * Take square of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def square(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("square", src)
  }

  /**
   * Take exp of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def exp(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("exp", src)
  }

  /**
   * Take log of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def log(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("log", src)
  }

  /**
   * Take cos of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def cos(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("cos", src)
  }

  /**
   * Take sin of the src
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def sin(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("sin", src)
  }

  /**
   * Take max of the src. The result will be ndarray of shape (1,) on the same device.
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def max(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("max", src)
  }

  /**
   * Take max of the src.The result will be ndarray of shape (1,) on the same device.
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def min(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("min", src)
  }

  /**
   * Take sum of the src. The result will be ndarray of shape (1,) on the same device.
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def sum(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("sum", src)
  }

  /**
   * Take the argmax index of each channel (row) in src.
   * @param src Source ndarray
   * @return a new [[NDArray]]
   */
  def argmaxChannel(src: NDArray): NDArray = {
    NDArray.invokeUnaryFunc("argmax_channel", src)
  }

  /**
   * Choose one element from each row in array according to the index.
   * This function assume index uses 0-based index.
   * @param array source array
   * @param index index array
   * @return a new [[NDArray]]
   */
  def chooseElement0Index(array: NDArray, index: NDArray): NDArray = {
    NDArray.invokeBinaryFunc("choose_element_0index", array, index)
  }

  def randomUniform(low: Float, high: Float, out: NDArray): NDArray = {
    require(out != null)
    NDArray.invokeGenericFunc("_sample_uniform", kwargs = Map[String, Any](
      "low" -> low, "high" -> high, "shape" -> out.shape, "out" -> out))(0)
  }

  def randomGaussian(loc: Float, scale: Float, out: NDArray): NDArray = {
    require(out != null)
    NDArray.invokeGenericFunc("_sample_normal", kwargs = Map[String, Any](
      "loc" -> loc, "scale" -> scale, "shape" -> out.shape, "out" -> out))(0)
  }

  /**
   * Create a new NDArray that copies content from source_array.
   * @param sourceArr Source data to create NDArray from.
   * @param shape shape of the NDArray
   * @param ctx The context of the NDArray, default to current default context.
   * @return The created NDArray.
   */
  def array(sourceArr: Array[Float], shape: Shape, ctx: Context = null): NDArray = {
     val arr = empty(shape, ctx)
     arr.set(sourceArr)
     arr
  }

  /**
   * Join a sequence of arrays at axis-0
   * TODO: shall we make it native?
   * @param arrays
   */
  def concatenate(arrays: Seq[NDArray], ctx: Context = null): NDArray = {
    require(arrays != null && arrays.size > 0, "arrays empty")
    val array0 = arrays.head
    val shape = array0.shape.drop(1)
    var axis0 = array0.shape(0)
    arrays.drop(1).foreach { array =>
      require(shape == array.shape.drop(1),
        s"shape mismatch between ${array.shape} and $shape")
      axis0 += array.shape(0)
    }

    val output = NDArray.empty(Shape(axis0) ++ shape, ctx)
    axis0 = 0
    arrays.foreach { array =>
      output.slice(axis0, axis0 + array.shape(0)).set(array)
      axis0 += array.shape(0)
    }

    output
  }

  def concatenate(arrays: NDArray *): NDArray = {
    concatenate(arrays.toSeq)
  }

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
  def load(fname: String): (Array[String], Array[NDArray]) = {
    val outSize = new MXUintRef
    val outNameSize = new MXUintRef
    val handles = ArrayBuffer.empty[NDArrayHandle]
    val names = ArrayBuffer.empty[String]
    checkCall(_LIB.mxNDArrayLoad(fname, outSize, handles, outNameSize, names))
    require(outNameSize.value == 0 || outNameSize.value == outSize.value)
    (names.toArray, handles.map(new NDArray(_)).toArray)
  }

  def load2Map(fname: String): Map[String, NDArray] = {
    val (keys, vals) = load(fname)
    require(keys.length == vals.length, "Loaded NDArrays have no name")
    (keys zip vals).toMap
  }

  def load2Array(fname: String): Array[NDArray] = {
    load(fname)._2
  }

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
  def save(fname: String, data: Map[String, NDArray]): Unit = {
    val keys = data.keys.toArray
    val handles = data.values.map(_.handle).toArray
    save(fname, keys, handles)
  }

  def save(fname: String, data: Traversable[NDArray]): Unit = {
    save(fname, null, data.map(_.handle).toArray)
  }

  private def save(fname: String, keys: Array[String], handles: Array[NDArrayHandle]): Unit = {
    checkCall(_LIB.mxNDArraySave(fname, handles, keys))
  }

  def deserialize(bytes: Array[Byte]): NDArray = {
    val handleRef = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayLoadFromRawBytes(bytes, handleRef))
    new NDArray(handleRef.value)
  }
}

/**
 * NDArray object in mxnet.
 * NDArray is basic ndarray/Tensor like data structure in mxnet. <br />
 * <b>
 * WARNING: it is your responsibility to clear this object through dispose().
 * NEVER rely on the GC strategy
 * </b>
 */
// scalastyle:off finalize
class NDArray private[mxnet](private[mxnet] val handle: NDArrayHandle,
                             val writable: Boolean = true) {
  // record arrays who construct this array instance
  // we use weak reference to prevent gc blocking
  private[mxnet] val dependencies = mutable.HashMap.empty[Long, WeakReference[NDArray]]
  private var disposed = false
  def isDisposed: Boolean = disposed
  override protected def finalize(): Unit = {
    dispose()
  }

  def serialize(): Array[Byte] = {
    val buf = ArrayBuffer.empty[Byte]
    checkCall(_LIB.mxNDArraySaveRawBytes(handle, buf))
    buf.toArray
  }

  /**
   * Release the native memory. <br />
   * The NDArrays it depends on will NOT be disposed. <br />
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (!disposed) {
      _LIB.mxNDArrayFree(handle)
      dependencies.clear()
      disposed = true
    }
  }

  /**
   * Dispose all NDArrays who help to construct this array. <br />
   * e.g. (a * b + c).disposeDeps() will dispose a, b, c (including their deps) and a * b
   * @return this array
   */
  def disposeDeps(): NDArray = {
    disposeDepsExcept()
  }

  /**
   * Dispose all NDArrays who help to construct this array, excepts those in the arguments. <br />
   * e.g. (a * b + c).disposeDepsExcept(a, b)
   * will dispose c and a * b.
   * Note that a, b's dependencies will not be disposed either.
   * @return this array
   */
  def disposeDepsExcept(arrs: NDArray*): NDArray = {
    if (dependencies != null) {
      val excepts = mutable.HashSet.empty[Long]
      arrs.foreach { arr =>
        excepts += arr.handle
        excepts ++= arr.dependencies.keys
      }
      dependencies.retain { case (addr, weak) =>
        if (excepts.contains(addr)) {
          true
        } else {
          weak.get match {
            case Some(arr) => arr.dispose()
            case None =>
          }
          false
        }
      }
    }
    this
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
    val sliceHandle = new NDArrayHandleRef
    checkCall(_LIB.mxNDArraySlice(handle, start, stop, sliceHandle))
    new NDArray(handle = sliceHandle.value, writable = this.writable)
  }

  def slice(range: (Int, Int)): NDArray = {
    slice(range._1, range._2)
  }

  /**
   * Return a sliced NDArray at the ith position of axis0
   * NDArray only support continuous slicing on axis 0
   * @param i
   * @return a sliced NDArray that shares memory with current one.
   */
  def slice(i: Int): NDArray = {
    slice(i, i + 1)
  }

  /**
   * Return a reshaped NDArray that shares memory with current one.
   *
   * @param dims New shape.
   *
   * @return a reshaped NDArray that shares memory with current one.
   */
  def reshape(dims: Array[Int]): NDArray = {
    val reshapeHandle = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayReshape(handle, dims.length, dims, reshapeHandle))
    new NDArray(handle = reshapeHandle.value, writable = this.writable)
  }

  /**
   * Block until all pending writes operations on current NDArray are finished.
   * This function will return when all the pending writes to the current
   * NDArray finishes. There can still be pending read going on when the
   * function returns.
   */
  def waitToRead(): Unit = {
    checkCall(_LIB.mxNDArrayWaitToRead(handle))
  }

  /**
   * Get context of current NDArray.
   * @return The context of current NDArray.
   */
  def context: Context = {
    val devTypeId = new RefInt
    val devId = new RefInt
    checkCall(_LIB.mxNDArrayGetContext(handle, devTypeId, devId))
    new Context(Context.devtype2str(devTypeId.value), devId.value)
  }

  /**
   * Set the values of the NDArray
   * @param value Value to set
   * @return Current NDArray
   */
  def set(value: Float): NDArray = {
    require(writable, "trying to assign to a readonly NDArray")
    NDArray.invokeGenericFunc("_set_value", Array[Any](value), Map[String, Any]("out" -> this))
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
    NDArray.invokeBinaryFunc("_plus", this, other)
  }

  def +(other: Float): NDArray = {
    NDArray.invokeGenericFunc("_plus_scalar", Array[Any](this, other))(0)
  }

  def +=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray.invokeBinaryFunc("_plus", this, other, out = this)
  }

  def +=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray.invokeGenericFunc("_plus_scalar", Array[Any](this, other),
      Map[String, Any]("out" -> this))
    this
  }

  def -(other: NDArray): NDArray = {
    NDArray.invokeBinaryFunc("_minus", this, other)
  }

  def -(other: Float): NDArray = {
    NDArray.invokeGenericFunc("_minus_scalar", Array[Any](this, other))(0)
  }

  def -=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray.invokeBinaryFunc("_minus", this, other, out = this)
  }

  def -=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray.invokeGenericFunc("_minus_scalar", Array[Any](this, other),
      Map[String, Any]("out" -> this))
    this
  }

  def *(other: NDArray): NDArray = {
    NDArray.invokeBinaryFunc("_mul", this, other)
  }

  def *(other: Float): NDArray = {
    NDArray.invokeGenericFunc("_mul_scalar", Array[Any](this, other))(0)
  }

  def unary_-(): NDArray = {
    NDArray.invokeGenericFunc("_mul_scalar", Array[Any](this, -1f))(0)
  }

  def *=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray.invokeBinaryFunc("_mul", this, other, out = this)
  }

  def *=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray.invokeGenericFunc("_mul_scalar", Array[Any](this, other),
      Map[String, Any]("out" -> this))
    this
  }

  def /(other: NDArray): NDArray = {
    NDArray.invokeBinaryFunc("_div", this, other)
  }

  def /(other: Float): NDArray = {
    NDArray.invokeGenericFunc("_div_scalar", Array[Any](this, other))(0)
  }

  def /=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray.invokeBinaryFunc("_div", this, other, out = this)
  }

  def /=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray.invokeGenericFunc("_div_scalar", Array[Any](this, other),
      Map[String, Any]("out" -> this))
    this
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * @return  A copy of array content.
   */
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
    require(shape == Shape(1), "The current array is not a scalar")
    this.toArray(0)
  }

  /**
   * Copy the content of current array to other.
   *
   * @param other Target NDArray or context we want to copy data to.
   * @return The copy target NDArray
   */
  def copyTo(other: NDArray): NDArray = {
    if (other.handle == this.handle) {
      NDArray.logger.warn("copy an array to itself, is it intended ?")
      other
    } else {
      NDArray.invokeUnaryFunc("_copyto", this, out = other)
    }
  }

  /**
   * Copy the content of current array to a new NDArray in the context.
   *
   * @param ctx Target context we want to copy data to.
   * @return The copy target NDArray
   */
  def copyTo(ctx: Context): NDArray = {
    val ret = new NDArray(NDArray.newAllocHandle(shape, ctx, delayAlloc = true))
    copyTo(ret)
  }

  /**
   * Clone the current array
   * @return the copied NDArray in the same context
   */
  def copy(): NDArray = copyTo(this.context)

  /**
   * Get shape of current NDArray.
   * @return an array representing shape of current ndarray
   */
  def shape: Shape = {
    val ndim = new MXUintRef
    val data = ArrayBuffer[Int]()
    checkCall(_LIB.mxNDArrayGetShape(handle, ndim, data))
    require(ndim.value == data.length, s"ndim=$ndim, while len(pdata)=${data.length}")
    Shape(data)
  }

  // Get size of current NDArray.
  def size: Int = shape.product

  override def equals(o: Any): Boolean = o match {
    case that: NDArray =>
      that != null && that.shape == this.shape && that.toArray.sameElements(this.toArray)
    case _ => false
  }

  override def hashCode: Int = {
    // TODO: naive implementation
    shape.hashCode + toArray.hashCode
  }
}
// scalastyle:on finalize

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
    NDArray.invokeGenericFunc("_rminus_scalar", Array[Any](other, value))(0)
  }

  def *(other: NDArray): NDArray = {
    other * value
  }

  def /(other: NDArray): NDArray = {
    NDArray.invokeGenericFunc("_rdiv_scalar", Array[Any](other, value))(0)
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
