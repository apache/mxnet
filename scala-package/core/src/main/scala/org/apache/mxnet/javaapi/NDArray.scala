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

@AddJNDArrayAPIs(false)
object NDArray {
  implicit def fromNDArray(nd: org.apache.mxnet.NDArray): NDArray = new NDArray(nd)

  implicit def toNDArray(jnd: NDArray): org.apache.mxnet.NDArray = jnd.nd

  def waitall(): Unit = org.apache.mxnet.NDArray.waitall()

  def onehotEncode(indices: NDArray, out: NDArray): NDArray = org.apache.mxnet.NDArray.onehotEncode(indices, out)

  def empty(shape: Shape, ctx: Context, dtype: DType.DType): NDArray
  = org.apache.mxnet.NDArray.empty(shape, ctx, dtype)
  def empty(ctx: Context, shape: Array[Int]): NDArray
  = org.apache.mxnet.NDArray.empty(new Shape(shape), ctx)
  def empty(ctx : Context, shape : java.util.List[java.lang.Integer]) : NDArray
  = org.apache.mxnet.NDArray.empty(new Shape(shape), ctx)
  def zeros(shape: Shape, ctx: Context, dtype: DType.DType): NDArray
  = org.apache.mxnet.NDArray.zeros(shape, ctx, dtype)
  def zeros(ctx: Context, shape: Array[Int]): NDArray
  = org.apache.mxnet.NDArray.zeros(new Shape(shape), ctx)
  def zeros(ctx : Context, shape : java.util.List[java.lang.Integer]) : NDArray
  = org.apache.mxnet.NDArray.zeros(new Shape(shape), ctx)
  def ones(shape: Shape, ctx: Context, dtype: DType.DType): NDArray
  = org.apache.mxnet.NDArray.ones(shape, ctx, dtype)
  def ones(ctx: Context, shape: Array[Int]): NDArray
  = org.apache.mxnet.NDArray.ones(new Shape(shape), ctx)
  def ones(ctx : Context, shape : java.util.List[java.lang.Integer]) : NDArray
  = org.apache.mxnet.NDArray.ones(new Shape(shape), ctx)
  def full(shape: Shape, value: Float, ctx: Context): NDArray = org.apache.mxnet.NDArray.full(shape, value, ctx)

  def power(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.power(lhs, rhs)
  def power(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.power(lhs, rhs)
  def power(lhs: Float, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.power(lhs, rhs)

  def maximum(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.maximum(lhs, rhs)
  def maximum(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.maximum(lhs, rhs)
  def maximum(lhs: Float, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.maximum(lhs, rhs)

  def minimum(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.minimum(lhs, rhs)
  def minimum(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.minimum(lhs, rhs)
  def minimum(lhs: Float, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.minimum(lhs, rhs)

  def equal(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.equal(lhs, rhs)
  def equal(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.equal(lhs, rhs)

  def notEqual(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.notEqual(lhs, rhs)
  def notEqual(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.notEqual(lhs, rhs)

  def greater(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.greater(lhs, rhs)
  def greater(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.greater(lhs, rhs)

  def greaterEqual(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.greaterEqual(lhs, rhs)
  def greaterEqual(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.greaterEqual(lhs, rhs)

  def lesser(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.lesser(lhs, rhs)
  def lesser(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.lesser(lhs, rhs)

  def lesserEqual(lhs: NDArray, rhs: NDArray): NDArray = org.apache.mxnet.NDArray.lesserEqual(lhs, rhs)
  def lesserEqual(lhs: NDArray, rhs: Float): NDArray = org.apache.mxnet.NDArray.lesserEqual(lhs, rhs)

  def array(sourceArr: java.util.List[java.lang.Float], shape: Shape, ctx: Context = null): NDArray
  = org.apache.mxnet.NDArray.array(sourceArr.asScala.map(ele => Float.unbox(ele)).toArray, shape, ctx)

  def arange(start: Float, stop: Float, step: Float, repeat: Int,
             ctx: Context, dType: DType.DType): NDArray =
    org.apache.mxnet.NDArray.arange(start, Some(stop), step, repeat, ctx, dType)
}

class NDArray(val nd : org.apache.mxnet.NDArray ) {

  def this(arr : Array[Float], shape : Shape, ctx : Context) = {
    this(org.apache.mxnet.NDArray.array(arr, shape, ctx))
  }

  def this(arr : java.util.List[java.lang.Float], shape : Shape, ctx : Context) = {
    this(NDArray.array(arr, shape, ctx))
  }

  def serialize() : Array[Byte] = nd.serialize()

  def dispose() : Unit = nd.dispose()
  def disposeDeps() : NDArray = nd.disposeDepsExcept()
  // def disposeDepsExcept(arr : Array[NDArray]) : NDArray = nd.disposeDepsExcept()

  def slice(start : Int, stop : Int) : NDArray = nd.slice(start, stop)

  def slice (i : Int) : NDArray = nd.slice(i)

  def at(idx : Int) : NDArray = nd.at(idx)

  def T : NDArray = nd.T

  def dtype : DType = nd.dtype

  def asType(dtype : DType) : NDArray = nd.asType(dtype)

  def reshape(dims : Array[Int]) : NDArray = nd.reshape(dims)

  def waitToRead(): Unit = nd.waitToRead()

  def context : Context = nd.context

  def set(value : Float) : NDArray = nd.set(value)
  def set(other : NDArray) : NDArray = nd.set(other)
  def set(other : Array[Float]) : NDArray = nd.set(other)

  def add(other : NDArray) : NDArray = this.nd + other.nd
  def add(other : Float) : NDArray = this.nd + other
  def _add(other : NDArray) : NDArray = this.nd += other
  def _add(other : Float) : NDArray = this.nd += other
  def subtract(other : NDArray) : NDArray = this.nd - other
  def subtract(other : Float) : NDArray = this.nd - other
  def _subtract(other : NDArray) : NDArray = this.nd -= other
  def _subtract(other : Float) : NDArray = this.nd -= other
  def multiply(other : NDArray) : NDArray = this.nd * other
  def multiply(other : Float) : NDArray = this.nd * other
  def _multiply(other : NDArray) : NDArray = this.nd *= other
  def _multiply(other : Float) : NDArray = this.nd *= other
  def div(other : NDArray) : NDArray = this.nd / other
  def div(other : Float) : NDArray = this.nd / other
  def _div(other : NDArray) : NDArray = this.nd /= other
  def _div(other : Float) : NDArray = this.nd /= other
  def pow(other : NDArray) : NDArray = this.nd ** other
  def pow(other : Float) : NDArray = this.nd ** other
  def _pow(other : NDArray) : NDArray = this.nd **= other
  def _pow(other : Float) : NDArray = this.nd **= other
  def mod(other : NDArray) : NDArray = this.nd % other
  def mod(other : Float) : NDArray = this.nd % other
  def _mod(other : NDArray) : NDArray = this.nd %= other
  def _mod(other : Float) : NDArray = this.nd %= other
  def greater(other : NDArray) : NDArray = this.nd > other
  def greater(other : Float) : NDArray = this.nd > other
  def greaterEqual(other : NDArray) : NDArray = this.nd >= other
  def greaterEqual(other : Float) : NDArray = this.nd >= other
  def lesser(other : NDArray) : NDArray = this.nd < other
  def lesser(other : Float) : NDArray = this.nd < other
  def lesserEqual(other : NDArray) : NDArray = this.nd <= other
  def lesserEqual(other : Float) : NDArray = this.nd <= other

  def toArray : Array[Float] = nd.toArray

  def toScalar : Float = nd.toScalar

  def copyTo(other : NDArray) : NDArray = nd.copyTo(other)

  def copyTo(ctx : Context) : NDArray = nd.copyTo(ctx)

  def copy() : NDArray = copyTo(this.context)

  def shape : Shape = nd.shape

  def size : Int = shape.product

  def asInContext(context: Context): NDArray = nd.asInContext(context)

  override def equals(obj: Any): Boolean = nd.equals(obj)
  override def hashCode(): Int = nd.hashCode
}

object NDArrayFuncReturn {
  implicit def toNDFuncReturn(javaFunReturn : NDArrayFuncReturn) : org.apache.mxnet.NDArrayFuncReturn = javaFunReturn.ndFuncReturn
  implicit def toJavaNDFuncReturn(ndFuncReturn : org.apache.mxnet.NDArrayFuncReturn) : NDArrayFuncReturn =
    new NDArrayFuncReturn(ndFuncReturn)
}

private[mxnet] class NDArrayFuncReturn(val ndFuncReturn : org.apache.mxnet.NDArrayFuncReturn) {
  def head : NDArray = ndFuncReturn.head
  def get : NDArray = ndFuncReturn.get
  def apply(i : Int) : NDArray = ndFuncReturn.apply(i)
  // TODO: Add JavaNDArray operational stuff
}