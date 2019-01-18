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

package org.apache.mxnet.javaapi

import org.apache.mxnet.javaapi.DType.DType

import collection.JavaConverters._
import scala.language.implicitConversions

@AddJNDArrayAPIs(false)
object NDArray extends NDArrayBase {
  implicit def fromNDArray(nd: org.apache.mxnet.NDArray): NDArray = new NDArray(nd)

  implicit def toNDArray(jnd: NDArray): org.apache.mxnet.NDArray = jnd.nd

  def waitall(): Unit = org.apache.mxnet.NDArray.waitall()

  /**
    * One hot encoding indices into matrix out.
    * @param indices An NDArray containing indices of the categorical features.
    * @param out The result holder of the encoding.
    * @return Same as out.
    */
  def onehotEncode(indices: NDArray, out: NDArray): NDArray
  = org.apache.mxnet.NDArray.onehotEncode(indices, out)

  /**
    * Create an empty uninitialized new NDArray, with specified shape.
    *
    * @param shape shape of the NDArray.
    * @param ctx The context of the NDArray.
    *
    * @return The created NDArray.
    */
  def empty(shape: Shape, ctx: Context, dtype: DType.DType): NDArray
  = org.apache.mxnet.NDArray.empty(shape, ctx, dtype)
  def empty(ctx: Context, shape: Array[Int]): NDArray
  = org.apache.mxnet.NDArray.empty(new Shape(shape), ctx)
  def empty(ctx: Context, shape: java.util.List[java.lang.Integer]): NDArray
  = org.apache.mxnet.NDArray.empty(new Shape(shape), ctx)

  /**
    * Create a new NDArray filled with 0, with specified shape.
    *
    * @param shape shape of the NDArray.
    * @param ctx The context of the NDArray.
    *
    * @return The created NDArray.
    */
  def zeros(shape: Shape, ctx: Context, dtype: DType.DType): NDArray
  = org.apache.mxnet.NDArray.zeros(shape, ctx, dtype)
  def zeros(ctx: Context, shape: Array[Int]): NDArray
  = org.apache.mxnet.NDArray.zeros(new Shape(shape), ctx)
  def zeros(ctx: Context, shape: java.util.List[java.lang.Integer]): NDArray
  = org.apache.mxnet.NDArray.zeros(new Shape(shape), ctx)

  /**
    * Create a new NDArray filled with 1, with specified shape.
    * @param shape shape of the NDArray.
    * @param ctx The context of the NDArray.
    * @return The created NDArray.
    */
  def ones(shape: Shape, ctx: Context, dtype: DType.DType): NDArray
  = org.apache.mxnet.NDArray.ones(shape, ctx, dtype)
  def ones(ctx: Context, shape: Array[Int]): NDArray
  = org.apache.mxnet.NDArray.ones(new Shape(shape), ctx)
  def ones(ctx: Context, shape: java.util.List[java.lang.Integer]): NDArray
  = org.apache.mxnet.NDArray.ones(new Shape(shape), ctx)

  /**
    * Create a new NDArray filled with given value, with specified shape.
    * @param shape shape of the NDArray.
    * @param value value to be filled with
    * @param ctx The context of the NDArray
    */
  def full(shape: Shape, value: Float, ctx: Context): NDArray
  = org.apache.mxnet.NDArray.full(shape, value, ctx)

  def full(shape: Shape, value: Double, ctx: Context): NDArray
  = org.apache.mxnet.NDArray.full(shape, value, ctx)

  def power(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.power(lhs, rhs)
  def power(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.power(lhs, rhs)
  def power(lhs: Float, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.power(lhs, rhs)
  def power(lhs: NDArray, rhs: Double): NDArray = org.apache.mxnet.NDArray.power(lhs, rhs)
  def power(lhs: Double, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.power(lhs, rhs)

  def maximum(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.maximum(lhs, rhs)
  def maximum(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.maximum(lhs, rhs)
  def maximum(lhs: Float, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.maximum(lhs, rhs)
  def maximum(lhs: NDArray, rhs: Double): NDArray = org.apache.mxnet.NDArray.maximum(lhs, rhs)
  def maximum(lhs: Double, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.maximum(lhs, rhs)

  def minimum(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.minimum(lhs, rhs)
  def minimum(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.minimum(lhs, rhs)
  def minimum(lhs: Float, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.minimum(lhs, rhs)
  def minimum(lhs: NDArray, rhs: Double): NDArray = org.apache.mxnet.NDArray.minimum(lhs, rhs)
  def minimum(lhs: Double, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.minimum(lhs, rhs)


  /**
    * Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.
    * For each element in input arrays, return 1(true) if corresponding elements are same,
    * otherwise return 0(false).
    */
  def equal(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.equal(lhs, rhs)
  def equal(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.equal(lhs, rhs)
  def equal(lhs: NDArray, rhs: Double): NDArray = org.apache.mxnet.NDArray.equal(lhs, rhs)

  /**
    * Returns the result of element-wise **not equal to** (!=) comparison operation
    * with broadcasting.
    * For each element in input arrays, return 1(true) if corresponding elements are different,
    * otherwise return 0(false).
    */
  def notEqual(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.notEqual(lhs, rhs)
  def notEqual(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.notEqual(lhs, rhs)
  def notEqual(lhs: NDArray, rhs: Double): NDArray = org.apache.mxnet.NDArray.notEqual(lhs, rhs)

  /**
    * Returns the result of element-wise **greater than** (>) comparison operation
    * with broadcasting.
    * For each element in input arrays, return 1(true) if lhs elements are greater than rhs,
    * otherwise return 0(false).
    */
  def greater(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.greater(lhs, rhs)
  def greater(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.greater(lhs, rhs)
  def greater(lhs: NDArray, rhs: Double): NDArray = org.apache.mxnet.NDArray.greater(lhs, rhs)

  /**
    * Returns the result of element-wise **greater than or equal to** (>=) comparison
    * operation with broadcasting.
    * For each element in input arrays, return 1(true) if lhs elements are greater than equal to rhs
    * otherwise return 0(false).
    */
  def greaterEqual(lhs: NDArray, rhs: NDArray): NDArray
  = org.apache.mxnet.NDArray.greaterEqual(lhs, rhs)
  def greaterEqual(lhs: NDArray, rhs: Float): NDArray
  = org.apache.mxnet.NDArray.greaterEqual(lhs, rhs)
  def greaterEqual(lhs: NDArray, rhs: Double): NDArray
  = org.apache.mxnet.NDArray.greaterEqual(lhs, rhs)

  /**
    * Returns the result of element-wise **lesser than** (<) comparison operation
    * with broadcasting.
    * For each element in input arrays, return 1(true) if lhs elements are less than rhs,
    * otherwise return 0(false).
    */
  def lesser(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.lesser(lhs, rhs)
  def lesser(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.lesser(lhs, rhs)
  def lesser(lhs: NDArray, rhs: Double): NDArray = org.apache.mxnet.NDArray.lesser(lhs, rhs)

  /**
    * Returns the result of element-wise **lesser than or equal to** (<=) comparison
    * operation with broadcasting.
    * For each element in input arrays, return 1(true) if lhs elements are
    * lesser than equal to rhs, otherwise return 0(false).
    */
  def lesserEqual(lhs: NDArray, rhs: NDArray): NDArray
  = org.apache.mxnet.NDArray.lesserEqual(lhs, rhs)
  def lesserEqual(lhs: NDArray, rhs: Float): NDArray
  = org.apache.mxnet.NDArray.lesserEqual(lhs, rhs)
  def lesserEqual(lhs: NDArray, rhs: Double): NDArray
  = org.apache.mxnet.NDArray.lesserEqual(lhs, rhs)

  /**
    * Create a new NDArray that copies content from source_array.
    * @param sourceArr Source data to create NDArray from.
    * @param shape shape of the NDArray
    * @param ctx The context of the NDArray, default to current default context.
    * @return The created NDArray.
    */
  def array(sourceArr: java.util.List[java.lang.Float], shape: Shape, ctx: Context = null): NDArray
  = org.apache.mxnet.NDArray.array(
    sourceArr.asScala.map(ele => Float.unbox(ele)).toArray, shape, ctx)

  /**
    * Create a new NDArray that copies content from source_array.
    * @param sourceArr Source data (list of Doubles) to create NDArray from.
    * @param shape shape of the NDArray
    * @param ctx The context of the NDArray, default to current default context.
    * @return The created NDArray.
    */
  def arrayWithDouble(sourceArr: java.util.List[java.lang.Double], shape: Shape,
                      ctx: Context = null): NDArray
  = org.apache.mxnet.NDArray.array(
    sourceArr.asScala.map(ele => Double.unbox(ele)).toArray, shape)

  /**
    * Returns evenly spaced values within a given interval.
    * Values are generated within the half-open interval [`start`, `stop`). In other
    * words, the interval includes `start` but excludes `stop`.
    * @param start Start of interval.
    * @param stop End of interval.
    * @param step Spacing between values.
    * @param repeat Number of times to repeat each element.
    * @param ctx Device context.
    * @param dType The data type of the `NDArray`.
    * @return NDArray of evenly spaced values in the specified range.
    */
  def arange(start: Float, stop: Float, step: Float, repeat: Int,
             ctx: Context, dType: DType.DType): NDArray =
    org.apache.mxnet.NDArray.arange(start, Some(stop), step, repeat, ctx, dType)
}

/**
  * NDArray object in mxnet.
  * NDArray is basic ndarray/Tensor like data structure in mxnet. <br />
  * <b>
  * NOTE: NDArray is stored in native memory. Use NDArray in a try-with-resources() construct
  * or a [[org.apache.mxnet.ResourceScope]] in a try-with-resource to have them
  * automatically disposed. You can explicitly control the lifetime of NDArray
  * by calling dispose manually. Failure to do this will result in leaking native memory.
  * </b>
  */
class NDArray private[mxnet] (val nd: org.apache.mxnet.NDArray ) {

  def this(arr: Array[Float], shape: Shape, ctx: Context) = {
    this(org.apache.mxnet.NDArray.array(arr, shape, ctx))
  }

  def this(arr: Array[Double], shape: Shape, ctx: Context) = {
    this(org.apache.mxnet.NDArray.array(arr, shape, ctx))
  }

  def this(arr: java.util.List[java.lang.Float], shape: Shape, ctx: Context) = {
    this(NDArray.array(arr, shape, ctx))
  }

  def serialize(): Array[Byte] = nd.serialize()

  /**
    * Release the native memory. <br />
    * The NDArrays it depends on will NOT be disposed. <br />
    * The object shall never be used after it is disposed.
    */
  def dispose(): Unit = nd.dispose()

  /**
    * Dispose all NDArrays who help to construct this array. <br />
    * e.g. (a * b + c).disposeDeps() will dispose a, b, c (including their deps) and a * b
    * @return this array
    */
  def disposeDeps(): NDArray = nd.disposeDepsExcept()

  /**
    * Dispose all NDArrays who help to construct this array, excepts those in the arguments. <br />
    * e.g. (a * b + c).disposeDepsExcept(a, b)
    * will dispose c and a * b.
    * Note that a, b's dependencies will not be disposed either.
    * @param arr the Array of NDArray not to dispose
    * @return this array
    */
  def disposeDepsExcept(arr: Array[NDArray]): NDArray =
    nd.disposeDepsExcept(arr.map(NDArray.toNDArray): _*)

  /**
    * Return a sliced NDArray that shares memory with current one.
    * NDArray only support continuous slicing on axis 0
    *
    * @param start Starting index of slice.
    * @param stop Finishing index of slice.
    *
    * @return a sliced NDArray that shares memory with current one.
    */
  def slice(start: Int, stop: Int): NDArray = nd.slice(start, stop)

  /**
    * Return a sliced NDArray at the ith position of axis0
    * @param i
    * @return a sliced NDArray that shares memory with current one.
    */
  def slice (i: Int): NDArray = nd.slice(i)

  /**
    * Return a sub NDArray that shares memory with current one.
    * the first axis will be rolled up, which causes its shape different from slice(i, i+1)
    * @param idx index of sub array.
    */
  def at(idx: Int): NDArray = nd.at(idx)

  def T: NDArray = nd.T

  /**
    * Get data type of current NDArray.
    * @return class representing type of current ndarray
    */
  def dtype: DType = nd.dtype

  /**
    * Return a copied numpy array of current array with specified type.
    * @param dtype Desired type of result array.
    * @return A copy of array content.
    */
  def asType(dtype: DType): NDArray = nd.asType(dtype)

  /**
    * Return a reshaped NDArray that shares memory with current one.
    * @param dims New shape.
    *
    * @return a reshaped NDArray that shares memory with current one.
    */
  def reshape(dims: Array[Int]): NDArray = nd.reshape(dims)

  /**
    * Block until all pending writes operations on current NDArray are finished.
    * This function will return when all the pending writes to the current
    * NDArray finishes. There can still be pending read going on when the
    * function returns.
    */
  def waitToRead(): Unit = nd.waitToRead()

  /**
    * Get context of current NDArray.
    * @return The context of current NDArray.
    */
  def context: Context = nd.context

  /**
    * Set the values of the NDArray
    * @param value Value to set
    * @return Current NDArray
    */
  def set(value: Float): NDArray = nd.set(value)
  def set(value: Double): NDArray = nd.set(value)
  def set(other: NDArray): NDArray = nd.set(other)
  def set(other: Array[Float]): NDArray = nd.set(other)
  def set(other: Array[Double]): NDArray = nd.set(other)

  def add(other: NDArray): NDArray = this.nd + other.nd
  def add(other: Float): NDArray = this.nd + other
  def add(other: Double): NDArray = this.nd + other
  def addInplace(other: NDArray): NDArray = this.nd += other
  def addInplace(other: Float): NDArray = this.nd += other
  def addInplace(other: Double): NDArray = this.nd += other
  def subtract(other: NDArray): NDArray = this.nd - other
  def subtract(other: Float): NDArray = this.nd - other
  def subtract(other: Double): NDArray = this.nd - other
  def subtractInplace(other: NDArray): NDArray = this.nd -= other
  def subtractInplace(other: Float): NDArray = this.nd -= other
  def subtractInplace(other: Double): NDArray = this.nd -= other
  def multiply(other: NDArray): NDArray = this.nd * other
  def multiply(other: Float): NDArray = this.nd * other
  def multiply(other: Double): NDArray = this.nd * other
  def multiplyInplace(other: NDArray): NDArray = this.nd *= other
  def multiplyInplace(other: Float): NDArray = this.nd *= other
  def multiplyInplace(other: Double): NDArray = this.nd *= other
  def div(other: NDArray): NDArray = this.nd / other
  def div(other: Float): NDArray = this.nd / other
  def div(other: Double): NDArray = this.nd / other
  def divInplace(other: NDArray): NDArray = this.nd /= other
  def divInplace(other: Float): NDArray = this.nd /= other
  def divInplace(other: Double): NDArray = this.nd /= other
  def pow(other: NDArray): NDArray = this.nd ** other
  def pow(other: Float): NDArray = this.nd ** other
  def pow(other: Double): NDArray = this.nd ** other
  def powInplace(other: NDArray): NDArray = this.nd **= other
  def powInplace(other: Float): NDArray = this.nd **= other
  def powInplace(other: Double): NDArray = this.nd **= other
  def mod(other: NDArray): NDArray = this.nd % other
  def mod(other: Float): NDArray = this.nd % other
  def mod(other: Double): NDArray = this.nd % other
  def modInplace(other: NDArray): NDArray = this.nd %= other
  def modInplace(other: Float): NDArray = this.nd %= other
  def modInplace(other: Double): NDArray = this.nd %= other
  def greater(other: NDArray): NDArray = this.nd > other
  def greater(other: Float): NDArray = this.nd > other
  def greater(other: Double): NDArray = this.nd > other
  def greaterEqual(other: NDArray): NDArray = this.nd >= other
  def greaterEqual(other: Float): NDArray = this.nd >= other
  def greaterEqual(other: Double): NDArray = this.nd >= other
  def lesser(other: NDArray): NDArray = this.nd < other
  def lesser(other: Float): NDArray = this.nd < other
  def lesser(other: Double): NDArray = this.nd < other
  def lesserEqual(other: NDArray): NDArray = this.nd <= other
  def lesserEqual(other: Float): NDArray = this.nd <= other
  def lesserEqual(other: Double): NDArray = this.nd <= other

  /**
    * Return a copied flat java array of current array (row-major).
    * @return  A copy of array content.
    */
  def toArray: Array[Float] = nd.toArray

  /**
    * Return a copied flat java array of current array (row-major).
    * @return  A copy of array content.
    */
  def toFloat64Array: Array[Double] = nd.toFloat64Array

  /**
    * Return a CPU scalar(float) of current ndarray.
    * This ndarray must have shape (1,)
    *
    * @return The scalar representation of the ndarray.
    */
  def toScalar: Float = nd.toScalar

  /**
    * Return a CPU scalar(float) of current ndarray.
    * This ndarray must have shape (1,)
    *
    * @return The scalar representation of the ndarray.
    */
  def toFloat64Scalar: Double = nd.toFloat64Scalar

  /**
    * Copy the content of current array to other.
    *
    * @param other Target NDArray or context we want to copy data to.
    * @return The copy target NDArray
    */
  def copyTo(other: NDArray): NDArray = nd.copyTo(other)

  /**
    * Copy the content of current array to a new NDArray in the context.
    *
    * @param ctx Target context we want to copy data to.
    * @return The copy target NDArray
    */
  def copyTo(ctx: Context): NDArray = nd.copyTo(ctx)

  /**
    * Clone the current array
    * @return the copied NDArray in the same context
    */
  def copy(): NDArray = copyTo(this.context)

  /**
    * Get shape of current NDArray.
    * @return an array representing shape of current ndarray
    */
  def shape: Shape = nd.shape


  def size: Int = shape.product

  /**
    * Return an `NDArray` that lives in the target context. If the array
    * is already in that context, `self` is returned. Otherwise, a copy is made.
    * @param context The target context we want the return value to live in.
    * @return A copy or `self` as an `NDArray` that lives in the target context.
    */
  def asInContext(context: Context): NDArray = nd.asInContext(context)

  override def equals(obj: Any): Boolean = nd.equals(obj)
  override def hashCode(): Int = nd.hashCode
}
