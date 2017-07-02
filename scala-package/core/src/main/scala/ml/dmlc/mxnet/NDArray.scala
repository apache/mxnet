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

import java.nio.{ByteOrder, ByteBuffer}

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.DType.DType
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.ref.WeakReference

/**
 * NDArray API of mxnet
 */
@AddNDArrayFunctions(false)
object NDArray {
  implicit def getFirstResult(ret: NDArrayFuncReturn): NDArray = ret(0)
  private val logger = LoggerFactory.getLogger(classOf[NDArray])

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

  /**
   * Used by NDArrayMacro.
   * Invoke this function by passing in parameters.
   * Parameters
   * ----------
   * @param args Positional arguments of input scalars and NDArray
   * @param kwargs Key-value arguments of input scalars
   * @return The result NDArrays of result of computation.
   */
  private[mxnet] def genericNDArrayFunctionInvoke(
    funcName: String, args: Seq[Any], kwargs: Map[String, Any] = null): NDArrayFuncReturn = {
    val function = functions(funcName)
    val ndArgs = ArrayBuffer.empty[NDArray]
    val posArgs = ArrayBuffer.empty[String]
    args.foreach {
      case arr: NDArray =>
        ndArgs.append(arr)
      case arrFunRet: NDArrayFuncReturn =>
        arrFunRet.arr.foreach(ndArgs.append(_))
      case arg =>
        posArgs.append(arg.toString)
    }

    require(posArgs.length <= function.arguments.length,
      s"len(posArgs) = ${posArgs.length}, should be less or equal to len(arguments) " +
      s"= ${function.arguments.length}")
    val updatedKwargs: Map[String, String] =
      (Option(kwargs).getOrElse(Map.empty[String, String])
        ++ function.arguments.slice(0, posArgs.length).zip(posArgs) - "out"
      ).map { case (k, v) => k -> v.toString }

    val (oriOutputs, outputVars) =
      if (kwargs != null && kwargs.contains("out")) {
        val output = kwargs("out")
        output match {
          case nd: NDArray => (Array(nd), Array(nd.handle))
          case ndFuncRet: NDArrayFuncReturn => (ndFuncRet.arr, ndFuncRet.arr.map(_.handle))
          case ndArr: Seq[NDArray] => (ndArr.toArray, ndArr.toArray.map(_.handle))
          case _ => throw new IllegalArgumentException(
            "Unsupported out var type, should be NDArray or subclass of Seq[NDArray]")
        }
      } else {
        (null, null)
      }

    val outputs = ArrayBuffer.empty[NDArrayHandle]
    checkCall(_LIB.mxImperativeInvoke(function.handle, ndArgs.map(_.handle).toArray, outputVars,
      outputs, updatedKwargs.size, updatedKwargs.keys.toArray, updatedKwargs.values.toArray))
    new NDArrayFuncReturn(Option(oriOutputs).getOrElse {
      val outputArrs = outputs.map(new NDArray(_)).toArray
      addDependency(ndArgs.toArray, outputArrs)
      outputArrs
    })
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
                             delayAlloc: Boolean,
                             dtype: DType = DType.Float32): NDArrayHandle = {
    val hdl = new NDArrayHandleRef
    checkCall(_LIB.mxNDArrayCreateEx(
      shape.toArray,
      shape.length,
      ctx.deviceTypeid,
      ctx.deviceId,
      if (delayAlloc) 1 else 0,
      dtype.id,
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

  // List and add all the atomic symbol functions to current module.
  private def initNDArrayModule(): Map[String, NDArrayFunction] = {
    val opNames = ListBuffer.empty[String]
    checkCall(_LIB.mxListAllOpNames(opNames))
    opNames.map(opName => {
      val opHandle = new RefLong
      checkCall(_LIB.nnGetOpHandle(opName, opHandle))
      makeNDArrayFunction(opHandle.value, opName)
    }).toMap
  }

  // Create an atomic symbol function by handle and function name.
  private def makeNDArrayFunction(handle: NDArrayHandle, aliasName: String)
    : (String, NDArrayFunction) = {
    val name = new RefString
    val desc = new RefString
    val keyVarNumArgs = new RefString
    val numArgs = new RefInt
    val argNames = ListBuffer.empty[String]
    val argTypes = ListBuffer.empty[String]
    val argDescs = ListBuffer.empty[String]

    checkCall(_LIB.mxSymbolGetAtomicSymbolInfo(
      handle, name, desc, numArgs, argNames, argTypes, argDescs, keyVarNumArgs))
    val arguments = (argTypes zip argNames).filter { case (dtype, _) =>
      !(dtype.startsWith("NDArray") || dtype.startsWith("Symbol")
        || dtype.startsWith("NDArray-or-Symbol"))
    }.map { case (_, argName) =>
      argName
    }
    (aliasName, new NDArrayFunction(handle, arguments.toList))
  }

  /**
   * One hot encoding indices into matrix out.
   * @param indices An NDArray containing indices of the categorical features.
   * @param out The result holder of the encoding.
   * @return Same as out.
   */
  def onehotEncode(indices: NDArray, out: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke(
      "_onehot_encode", Seq(indices, out), Map("out" -> out))(0)
  }

  /**
   * Create an empty uninitialized new NDArray, with specified shape.
   *
   * @param shape shape of the NDArray.
   * @param ctx The context of the NDArray, default to current default context.
   *
   * @return The created NDArray.
   */
  def empty(shape: Shape, ctx: Context = null, dtype: DType = Base.MX_REAL_TYPE): NDArray = {
    val context = if (ctx == null) Context.defaultCtx else ctx
    new NDArray(handle = NDArray.newAllocHandle(shape, context, delayAlloc = false, dtype))
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
  def zeros(shape: Shape, ctx: Context = null, dtype: DType = Base.MX_REAL_TYPE): NDArray = {
    val arr = empty(shape, ctx, dtype)
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
  def ones(shape: Shape, ctx: Context = null, dtype: DType = Base.MX_REAL_TYPE): NDArray = {
    val arr = empty(shape, ctx, dtype)
    arr.set(1f)
    arr
  }

  def ones(shape: Int *): NDArray = ones(Shape(shape: _*))

  def ones(ctx: Context, shape: Int *): NDArray = ones(Shape(shape: _*), ctx)

  /**
   * Create a new NDArray filled with given value, with specified shape.
   * @param shape shape of the NDArray.
   * @param value value to be filled with
   * @param ctx The context of the NDArray, default to current default context
   */
  def full(shape: Shape, value: Float, ctx: Context = null): NDArray = {
    val arr = empty(shape, ctx)
    arr.set(value)
    arr
  }

  // Perform power operator
  def power(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_power", Seq(lhs, rhs))
  }

  def power(lhs: NDArray, rhs: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_power_scalar", Seq(lhs, rhs))
  }

  def power(lhs: Float, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_rpower_scalar", Seq(lhs, rhs))
  }

  // Perform maximum operator
  def maximum(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_maximum", Seq(lhs, rhs))
  }

  def maximum(lhs: NDArray, rhs: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_maximum_scalar", Seq(lhs, rhs))
  }

  def maximum(lhs: Float, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_maximum_scalar", Seq(lhs, rhs))
  }

  // Perform minimum operator
  def minimum(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_minimum", Seq(lhs, rhs))
  }

  def minimum(lhs: NDArray, rhs: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_minimum_scalar", Seq(lhs, rhs))
  }

  def minimum(lhs: Float, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_minimum_scalar", Seq(lhs, rhs))
  }

  /**
   * Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.
   * For each element in input arrays, return 1(true) if corresponding elements are same,
   * otherwise return 0(false).
   */
  def equal(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("broadcast_equal", Seq(lhs, rhs))
  }

  def equal(lhs: NDArray, rhs: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_equal_scalar", Seq(lhs, rhs))
  }

  /**
   * Returns the result of element-wise **not equal to** (!=) comparison operation
   * with broadcasting.
   * For each element in input arrays, return 1(true) if corresponding elements are different,
   * otherwise return 0(false).
   */
  def notEqual(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("broadcast_not_equal", Seq(lhs, rhs))
  }

  def notEqual(lhs: NDArray, rhs: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_not_equal_scalar", Seq(lhs, rhs))
  }

  /**
   * Returns the result of element-wise **greater than** (>) comparison operation
   * with broadcasting.
   * For each element in input arrays, return 1(true) if lhs elements are greater than rhs,
   * otherwise return 0(false).
   */
  def greater(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("broadcast_greater", Seq(lhs, rhs))
  }

  def greater(lhs: NDArray, rhs: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_greater_scalar", Seq(lhs, rhs))
  }

  /**
   * Returns the result of element-wise **greater than or equal to** (>=) comparison
   * operation with broadcasting.
   * For each element in input arrays, return 1(true) if lhs elements are greater than equal to rhs,
   * otherwise return 0(false).
   */
  def greaterEqual(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("broadcast_greater_equal", Seq(lhs, rhs))
  }

  def greaterEqual(lhs: NDArray, rhs: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_greater_equal_scalar", Seq(lhs, rhs))
  }

  /**
   * Returns the result of element-wise **lesser than** (<) comparison operation
   * with broadcasting.
   * For each element in input arrays, return 1(true) if lhs elements are less than rhs,
   * otherwise return 0(false).
   */
  def lesser(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("broadcast_lesser", Seq(lhs, rhs))
  }

  def lesser(lhs: NDArray, rhs: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_lesser_scalar", Seq(lhs, rhs))
  }

  /**
   * Returns the result of element-wise **lesser than or equal to** (<=) comparison
   * operation with broadcasting.
   * For each element in input arrays, return 1(true) if lhs elements are
   * lesser than equal to rhs, otherwise return 0(false).
   */
  def lesserEqual(lhs: NDArray, rhs: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("broadcast_lesser_equal", Seq(lhs, rhs))
  }

  def lesserEqual(lhs: NDArray, rhs: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_lesser_equal_scalar", Seq(lhs, rhs))
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
   * Concatenate a list of NDArrays along the specified dimension.
   * @param arrays Arrays to be concatenate.
   *               They must have identical shape except the first dimension.
   *               They also must have the same data type.
   * @param axis The axis along which to concatenate.
   * @param alwaysCopy Default `True`. When not `True`,
   *                   if the arrays only contain one `NDArray`,
   *                   that element will be returned directly, avoid copying.
   * @return An `NDArray` that lives on the same context as `arrays[0].context`.
   */
  def concatenate(arrays: Seq[NDArray], axis: Int = 0, alwaysCopy: Boolean = true): NDArray = {
    require(arrays.size > 0)

    val array0 = arrays(0)
    if (!alwaysCopy && arrays.size == 1) {
      array0
    } else {
      val shapeRest1 = array0.shape.slice(0, axis)
      val shapeRest2 = array0.shape.slice(axis + 1, array0.shape.length)
      val dtype = array0.dtype

      val shapeAxis =
        arrays.map(arr => {
          require(shapeRest1 == arr.shape.slice(0, axis))
          require(shapeRest2 == arr.shape.slice(axis + 1, arr.shape.length))
          require(dtype == arr.dtype)
          arr.shape(axis)
        }).sum
      val retShape = shapeRest1 ++ Shape(shapeAxis) ++ shapeRest2
      val ret = NDArray.empty(retShape, ctx = array0.context, dtype = dtype)

      var idx = 0
      val begin = Array.fill(retShape.length)(0)
      val end = retShape.toArray
      for (arr <- arrays) {
        if (axis == 0) {
          ret.slice(idx, idx + arr.shape(0)).set(arr)
        } else {
          begin(axis) = idx
          end(axis) = idx + arr.shape(axis)
          NDArray._crop_assign(Map("out" -> ret,
            "begin" -> Shape(begin),
            "end" -> Shape(end)))(ret, arr)
        }
        idx += arr.shape(axis)
      }
      ret
    }
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

  // TODO: imdecode
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
          weak.get.foreach(_.dispose())
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
    require(source.length == size,
      s"array size (${source.length}) do not match the size of NDArray ($size)")
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
   * @param i
   * @return a sliced NDArray that shares memory with current one.
   */
  def slice(i: Int): NDArray = {
    slice(i, i + 1)
  }

  /**
   * Return a sub NDArray that shares memory with current one.
   * the first axis will be rolled up, which causes its shape different from slice(i, i+1)
   * @param idx index of sub array.
   */
  def at(idx: Int): NDArray = {
    val handleRef = new NDArrayHandleRef()
    checkCall(_LIB.mxNDArrayAt(this.handle, idx, handleRef))
    new NDArray(handle = handleRef.value, writable = this.writable)
  }

  // Get transpose of current NDArray
  def T: NDArray = {
    require(this.shape.size == 2, "Only 2D matrix is allowed to be transposed")
    NDArray.genericNDArrayFunctionInvoke("transpose", Seq(this))
  }

  /**
   * Get data type of current NDArray.
   * @return class representing type of current ndarray
   */
  def dtype: DType = {
    val mxDtype = new RefInt
    checkCall(_LIB.mxNDArrayGetDType(handle, mxDtype))
    DType(mxDtype.value)
  }

  /**
   * Return a copied numpy array of current array with specified type.
   * @param dtype Desired type of result array.
   * @return A copy of array content.
   */
  def asType(dtype: DType): NDArray = {
    val res = NDArray.empty(this.shape, ctx = this.context, dtype = dtype)
    this.copyTo(res)
    res
  }

  /**
   * Return a reshaped NDArray that shares memory with current one.
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
   * Return a reshaped NDArray that shares memory with current one.
   * @param dims New shape.
   *
   * @return a reshaped NDArray that shares memory with current one.
   */
  def reshape(dims: Shape): NDArray = {
    reshape(dims.toArray)
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
    NDArray.genericNDArrayFunctionInvoke("_set_value", Seq(value), Map("out" -> this))
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
    NDArray.genericNDArrayFunctionInvoke("_plus", Seq(this, other))
  }

  def +(other: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_plus_scalar", Seq(this, other))
  }

  def +=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_plus", Seq(this, other), Map("out" -> this))
    this
  }

  def +=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to add to a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_plus_scalar", Seq(this, other), Map("out" -> this))
    this
  }

  def -(other: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_minus", Seq(this, other))
  }

  def -(other: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_minus_scalar", Seq(this, other))
  }

  def -=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_minus", Seq(this, other), Map("out" -> this))
    this
  }

  def -=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to subtract from a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_minus_scalar", Seq(this, other), Map("out" -> this))
    this
  }

  def *(other: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_mul", Seq(this, other))
  }

  def *(other: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_mul_scalar", Seq(this, other))
  }

  def unary_-(): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_mul_scalar", Seq(this, -1f))
  }

  def *=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_mul", Seq(this, other), Map("out" -> this))
    this
  }

  def *=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to multiply to a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_mul_scalar", Seq(this, other), Map("out" -> this))
    this
  }

  def /(other: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_div", Seq(this, other))
  }

  def /(other: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_div_scalar", Seq(this, other))
  }

  def /=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_div", Seq(this, other), Map("out" -> this))
    this
  }

  def /=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to divide from a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_div_scalar", Seq(this, other), Map("out" -> this))
    this
  }

  def **(other: NDArray): NDArray = {
    NDArray.power(this, other)
  }

  def **(other: Float): NDArray = {
    NDArray.power(this, other)
  }

  def **=(other: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_power", Seq(this, other), Map("out" -> this))
  }

  def **=(other: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_power_scalar", Seq(this, other), Map("out" -> this))
  }

  def >(other: NDArray): NDArray = {
    NDArray.greater(this, other)
  }

  def >(other: Float): NDArray = {
    NDArray.greater(this, other)
  }

  def >=(other: NDArray): NDArray = {
    NDArray.greaterEqual(this, other)
  }

  def >=(other: Float): NDArray = {
    NDArray.greaterEqual(this, other)
  }

  def <(other: NDArray): NDArray = {
    NDArray.lesser(this, other)
  }

  def <(other: Float): NDArray = {
    NDArray.lesser(this, other)
  }

  def <=(other: NDArray): NDArray = {
    NDArray.lesserEqual(this, other)
  }

  def <=(other: Float): NDArray = {
    NDArray.lesserEqual(this, other)
  }

  def %(other: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_mod", Seq(this, other))
  }

  def %(other: Float): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_mod_scalar", Seq(this, other))
  }

  def %=(other: NDArray): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to take modulo from a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_mod", Seq(this, other), Map("out" -> this))
    this
  }

  def %=(other: Float): NDArray = {
    if (!writable) {
      throw new IllegalArgumentException("trying to take modulo from a readonly NDArray")
    }
    NDArray.genericNDArrayFunctionInvoke("_mod_scalar", Seq(this, other), Map("out" -> this))
    this
  }

  /**
   * Return a copied flat java array of current array (row-major).
   * @return  A copy of array content.
   */
  def toArray: Array[Float] = {
    internal.toFloatArray
  }

  def internal: NDArrayInternal = {
    val myType = dtype
    val arrLength = DType.numOfBytes(myType) * size
    val arr = Array.ofDim[Byte](arrLength)
    checkCall(_LIB.mxNDArraySyncCopyToCPU(handle, arr, size))
    new NDArrayInternal(arr, myType)
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
    } else {
      NDArray.genericNDArrayFunctionInvoke("_copyto", Seq(this), Map("out" -> other))
    }
    other
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

  /**
   * Return an `NDArray` that lives in the target context. If the array
   * is already in that context, `self` is returned. Otherwise, a copy is made.
   * @param context The target context we want the return value to live in.
   * @return A copy or `self` as an `NDArray` that lives in the target context.
   */
  def asInContext(context: Context): NDArray = {
    if (this.context == context) this else this.copyTo(context)
  }

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

private[mxnet] object NDArrayConversions {
  implicit def int2Scalar(x: Int): NDArrayConversions = new NDArrayConversions(x.toFloat)
  implicit def double2Scalar(x: Double): NDArrayConversions = new NDArrayConversions(x.toFloat)
  implicit def float2Scalar(x: Float): NDArrayConversions = new NDArrayConversions(x)
}

private[mxnet] class NDArrayConversions(val value: Float) {
  def +(other: NDArray): NDArray = {
    other + value
  }
  def +(other: NDArrayFuncReturn): NDArray = {
    other.head + value
  }

  def -(other: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_rminus_scalar", Seq(other, value))
  }
  def -(other: NDArrayFuncReturn): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_rminus_scalar", Seq(other.head, value))
  }

  def *(other: NDArray): NDArray = {
    other * value
  }
  def *(other: NDArrayFuncReturn): NDArray = {
    other.head * value
  }

  def /(other: NDArray): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_rdiv_scalar", Seq(other, value))
  }
  def /(other: NDArrayFuncReturn): NDArray = {
    NDArray.genericNDArrayFunctionInvoke("_rdiv_scalar", Seq(other.head, value))
  }

  def **(other: NDArray): NDArray = {
    NDArray.power(value, other)
  }
  def **(other: NDArrayFuncReturn): NDArray = {
    NDArray.power(value, other.head)
  }

  def >(other: NDArray): NDArray = {
    NDArray.lesser(other, value)
  }
  def >(other: NDArrayFuncReturn): NDArray = {
    NDArray.lesser(other.head, value)
  }

  def >=(other: NDArray): NDArray = {
    NDArray.lesserEqual(other, value)
  }
  def >=(other: NDArrayFuncReturn): NDArray = {
    NDArray.lesserEqual(other.head, value)
  }

  def <(other: NDArray): NDArray = {
    NDArray.greater(other, value)
  }
  def <(other: NDArrayFuncReturn): NDArray = {
    NDArray.greater(other.head, value)
  }

  def <=(other: NDArray): NDArray = {
    NDArray.greaterEqual(other, value)
  }
  def <=(other: NDArrayFuncReturn): NDArray = {
    NDArray.greaterEqual(other.head, value)
  }
}

private case class NDArrayFunction(handle: NDArrayHandle, arguments: List[String])

private[mxnet] class NDArrayFuncReturn(private[mxnet] val arr: Array[NDArray]) {
  def head: NDArray = apply(0)
  def get: NDArray = {
    require(arr.length == 1, s"return array length = ${arr.length}")
    head
  }
  def apply(i: Int): NDArray = {
    if (arr == null || arr.length <= i) {
      null
    } else {
      arr(i)
    }
  }

  // copy methods from NDArray
  def isDisposed: Boolean = head.isDisposed
  def serialize(): Array[Byte] = head.serialize()
  def dispose(): Unit = head.dispose()
  def disposeDeps(): NDArray = head.disposeDeps()
  def disposeDepsExcept(arrs: NDArray*): NDArray = head.disposeDepsExcept(arrs: _*)
  def slice(start: Int, stop: Int): NDArray = head.slice(start, stop)
  def slice(range: (Int, Int)): NDArray = head.slice(range)
  def slice(i: Int): NDArray = head.slice(i)
  def reshape(dims: Array[Int]): NDArray = head.reshape(dims)
  def waitToRead(): Unit = head.waitToRead()
  def context: Context = head.context
  def set(value: Float): NDArray = head.set(value)
  def set(other: NDArray): NDArray = head.set(other)
  def set(other: Array[Float]): NDArray = head.set(other)
  def +(other: NDArray): NDArray = head + other
  def +(other: Float): NDArray = head + other
  def +=(other: NDArray): NDArray = head += other
  def +=(other: Float): NDArray = head += other
  def -(other: NDArray): NDArray = head - other
  def -(other: Float): NDArray = head - other
  def -=(other: NDArray): NDArray = head -= other
  def -=(other: Float): NDArray = head -= other
  def *(other: NDArray): NDArray = head * other
  def *(other: Float): NDArray = head * other
  def unary_-(): NDArray = -head
  def *=(other: NDArray): NDArray = head *= other
  def *=(other: Float): NDArray = head *= other
  def /(other: NDArray): NDArray = head / other
  def **(other: NDArray): NDArray = head ** other
  def **(other: Float): NDArray = head ** other
  def >(other: NDArray): NDArray = head > other
  def >(other: Float): NDArray = head > other
  def >=(other: NDArray): NDArray = head >= other
  def >=(other: Float): NDArray = head >= other
  def <(other: NDArray): NDArray = head < other
  def <(other: Float): NDArray = head < other
  def <=(other: NDArray): NDArray = head <= other
  def <=(other: Float): NDArray = head <= other
  def toArray: Array[Float] = head.toArray
  def toScalar: Float = head.toScalar
  def copyTo(other: NDArray): NDArray = head.copyTo(other)
  def copyTo(ctx: Context): NDArray = head.copyTo(ctx)
  def copy(): NDArray = head.copy()
  def shape: Shape = head.shape
  def size: Int = head.size
  def asInContext(context: Context): NDArray = head.asInContext(context)
}

private[mxnet] class NDArrayInternal (private val internal: Array[Byte], private val dtype: DType) {
  private val unitSize = DType.numOfBytes(dtype)
  require(internal.length > 0 && internal.length % unitSize == 0,
    s"$dtype size $unitSize cannot divide byte array size ${internal.length}")
  private val units: Array[Array[Byte]] = (
    for (i <- 0 until internal.length / unitSize)
      yield internal.slice(i * unitSize, (i + 1) * unitSize)
  ).toArray

  def getRaw: Array[Byte] = internal
  def toDoubleArray: Array[Double] = {
    require(dtype != DType.Float16, "Currently cannot convert float16 to native numerical types")
    dtype match {
      case DType.Float32 => units.map(wrapBytes(_).getFloat.toDouble)
      case DType.Float64 => units.map(wrapBytes(_).getDouble)
      case DType.Int32 => units.map(wrapBytes(_).getInt.toDouble)
      case DType.UInt8 => internal.map(_.toDouble)
    }
  }
  def toFloatArray: Array[Float] = {
    require(dtype != DType.Float16, "Currently cannot convert float16 to native numerical types")
    dtype match {
      case DType.Float32 => units.map(wrapBytes(_).getFloat)
      case DType.Float64 => units.map(wrapBytes(_).getDouble.toFloat)
      case DType.Int32 => units.map(wrapBytes(_).getInt.toFloat)
      case DType.UInt8 => internal.map(_.toFloat)
    }
  }
  def toIntArray: Array[Int] = {
    require(dtype != DType.Float16, "Currently cannot convert float16 to native numerical types")
    dtype match {
      case DType.Float32 => units.map(wrapBytes(_).getFloat.toInt)
      case DType.Float64 => units.map(wrapBytes(_).getDouble.toInt)
      case DType.Int32 => units.map(wrapBytes(_).getInt)
      case DType.UInt8 => internal.map(_.toInt)
    }
  }
  def toByteArray: Array[Byte] = {
    require(dtype != DType.Float16, "Currently cannot convert float16 to native numerical types")
    dtype match {
      case DType.Float16 | DType.Float32 => units.map(wrapBytes(_).getFloat.toByte)
      case DType.Float64 => units.map(wrapBytes(_).getDouble.toByte)
      case DType.Int32 => units.map(wrapBytes(_).getInt.toByte)
      case DType.UInt8 => internal.clone()
    }
  }

  private def wrapBytes(bytes: Array[Byte]): ByteBuffer = {
    val bb = ByteBuffer.wrap(bytes)
    bb.order(ByteOrder.LITTLE_ENDIAN)
    bb
  }
}
