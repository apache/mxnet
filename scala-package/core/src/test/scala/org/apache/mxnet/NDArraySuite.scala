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

package org.apache.mxnet

import java.io.File
import java.util.concurrent.atomic.AtomicInteger

import org.apache.mxnet.NDArrayConversions._
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import org.slf4j.LoggerFactory
import scala.collection.mutable.ArrayBuffer

class NDArraySuite extends FunSuite with BeforeAndAfterAll with Matchers {
  private val sequence: AtomicInteger = new AtomicInteger(0)

  private val logger = LoggerFactory.getLogger(classOf[NDArraySuite])

  test("to java array") {
    val ndarray = NDArray.zeros(2, 2)
    assert(ndarray.toArray === Array(0f, 0f, 0f, 0f))

    val float64Array = NDArray.zeros(Shape(2, 2), dtype = DType.Float64)
    assert(float64Array.toFloat64Array === Array(0d, 0d, 0d, 0d))
  }

  test("to scalar") {
    val ndzeros = NDArray.zeros(1)
    assert(ndzeros.toScalar === 0f)
    val ndones = NDArray.ones(1)
    assert(ndones.toScalar === 1f)
  }

  test("to float 64 scalar") {
    val ndzeros = NDArray.zeros(Shape(1), dtype = DType.Float64)
    assert(ndzeros.toFloat64Scalar === 0d)
    val ndones = NDArray.ones(Shape(1), dtype = DType.Float64)
    assert(ndones.toFloat64Scalar === 1d)
  }

  test ("call toScalar on an ndarray which is not a scalar") {
    intercept[Exception] { NDArray.zeros(1, 1).toScalar }
    intercept[Exception] { NDArray.zeros(shape = Shape (1, 1),
      dtype = DType.Float64).toFloat64Scalar }
  }

  test("size and shape") {
    val ndzeros = NDArray.zeros(4, 1)
    assert(ndzeros.shape === Shape(4, 1))
    assert(ndzeros.size === 4)
  }

  test("dtype") {
    val arr = NDArray.zeros(3, 2)
    assert(arr.dtype === DType.Float32)

    val float64Array = NDArray.zeros(shape = Shape(3, 2), dtype = DType.Float64)
    assert(float64Array.dtype === DType.Float64)
  }

  test("set scalar value") {
    val ndarray = NDArray.empty(2, 1)
    ndarray.set(10f)
    assert(ndarray.toArray === Array(10f, 10f))

    val float64array = NDArray.empty(shape = Shape(2, 1), dtype = DType.Float64)
    float64array.set(10d)
    assert(float64array.toFloat64Array === Array(10d, 10d))

  }

  test("copy from java array") {
    val ndarray = NDArray.empty(4, 1)
    ndarray.set(Array(1f, 2f, 3f, 4f))
    assert(ndarray.toArray === Array(1f, 2f, 3f, 4f))
  }

  test("create NDArray based on Java Matrix") {
    def arrayGen(num : Any) : Array[Any] = {
      val array = num match {
        case f: Float =>
          (for (_ <- 0 until 100) yield Array(1.0f, 1.0f, 1.0f, 1.0f)).toArray
        case d: Double =>
          (for (_ <- 0 until 100) yield Array(1.0d, 1.0d, 1.0d, 1.0d)).toArray
        case _ => throw new IllegalArgumentException(s"Unsupported Type ${num.getClass}")
      }
      Array(
        Array(
          array
        ),
        Array(
          array
        )
      )
    }
    val floatData = 1.0f
    var nd = NDArray.toNDArray(arrayGen(floatData))
    require(nd.shape == Shape(2, 1, 100, 4))
    val arr2 = Array(1.0f, 1.0f, 1.0f, 1.0f)
    nd = NDArray.toNDArray(arr2)
    require(nd.shape == Shape(4))
    val doubleData = 1.0d
    nd = NDArray.toNDArray(arrayGen(doubleData))
    require(nd.shape == Shape(2, 1, 100, 4))
    require(nd.dtype == DType.Float64)
  }

  test("test Visualize") {
    var nd = NDArray.ones(Shape(1, 2, 1000, 1))
    var data : String =
      """
        |[
        | [
        |  [
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |
        |   ... with length 1000
        |  ]
        |  [
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |   [1.0]
        |
        |   ... with length 1000
        |  ]
        |  ]
        |]
        |<NDArray (1,2,1000,1) cpu(0) float32>""".stripMargin
    require(nd.toString.split("\\s+").mkString == data.split("\\s+").mkString)
    nd = NDArray.ones(Shape(1, 4))
    data =
      """
        |[
        | [1.0,1.0,1.0,1.0]
        |]
        |<NDArray (1,4) cpu(0) float32>""".stripMargin
    require(nd.toString.split("\\s+").mkString == data.split("\\s+").mkString)
  }

  test("plus") {
    var ndzeros = NDArray.zeros(2, 1)
    var ndones = ndzeros + 1f
    assert(ndones.toArray === Array(1f, 1f))
    assert((ndones + ndzeros).toArray === Array(1f, 1f))
    assert((1 + ndones).toArray === Array(2f, 2f))
    // in-place
    ndones += ndones
    assert(ndones.toArray === Array(2f, 2f))

    // Float64 method test
    ndzeros = NDArray.zeros(shape = Shape(2, 1), dtype = DType.Float64)
    ndones = ndzeros + 1d
    assert(ndones.toFloat64Array === Array(1d, 1d))
    assert((ndones + ndzeros).toFloat64Array === Array(1d, 1d))
    assert((1d + ndones).toArray === Array(2d, 2d))
    // in-place
    ndones += ndones
    assert(ndones.toFloat64Array === Array(2d, 2d))
  }

  test("minus") {
    var ndones = NDArray.ones(2, 1)
    var ndzeros = ndones - 1f
    assert(ndzeros.toArray === Array(0f, 0f))
    assert((ndones - ndzeros).toArray === Array(1f, 1f))
    assert((ndzeros - ndones).toArray === Array(-1f, -1f))
    assert((ndones - 1).toArray === Array(0f, 0f))
    // in-place
    ndones -= ndones
    assert(ndones.toArray === Array(0f, 0f))

    // Float64 methods test
    ndones = NDArray.ones(shape = Shape(2, 1))
    ndzeros = ndones - 1d
    assert(ndzeros.toFloat64Array === Array(0d, 0d))
    assert((ndones - ndzeros).toFloat64Array === Array(1d , 1d))
    assert((ndzeros - ndones).toFloat64Array === Array(-1d , -1d))
    assert((ndones - 1).toFloat64Array === Array(0d, 0d))
    // in-place
    ndones -= ndones
    assert(ndones.toArray === Array(0d, 0d))

  }

  test("multiplication") {
    var ndones = NDArray.ones(2, 1)
    var ndtwos = ndones * 2
    assert(ndtwos.toArray === Array(2f, 2f))
    assert((ndones * ndones).toArray === Array(1f, 1f))
    assert((ndtwos * ndtwos).toArray === Array(4f, 4f))
    ndtwos *= ndtwos
    // in-place
    assert(ndtwos.toArray === Array(4f, 4f))

    // Float64 methods test
    ndones = NDArray.ones(shape = Shape(2, 1), dtype = DType.Float64)
    ndtwos = ndones * 2d
    assert(ndtwos.toFloat64Array === Array(2d, 2d))
    assert((ndones * ndones).toFloat64Array === Array(1d, 1d))
    assert((ndtwos * ndtwos).toFloat64Array === Array(4d, 4d))
    ndtwos *= ndtwos
    // in-place
    assert(ndtwos.toFloat64Array === Array(4d, 4d))

  }

  test("division") {
    var ndones = NDArray.ones(2, 1)
    var ndzeros = ndones - 1f
    var ndhalves = ndones / 2
    assert(ndhalves.toArray === Array(0.5f, 0.5f))
    assert((ndhalves / ndhalves).toArray === Array(1f, 1f))
    assert((ndones / ndones).toArray === Array(1f, 1f))
    assert((ndzeros / ndones).toArray === Array(0f, 0f))
    ndhalves /= ndhalves
    // in-place
    assert(ndhalves.toArray === Array(1f, 1f))

    // Float64 methods test
    ndones = NDArray.ones(shape = Shape (2, 1), dtype = DType.Float64)
    ndzeros = ndones - 1d
    ndhalves = ndones / 2d
    assert(ndhalves.toFloat64Array === Array(0.5d, 0.5d))
    assert((ndhalves / ndhalves).toFloat64Array === Array(1d, 1d))
    assert((ndones / ndones).toFloat64Array === Array(1d, 1d))
    assert((ndzeros / ndones).toFloat64Array === Array(0d, 0d))
    ndhalves /= ndhalves
    // in-place
    assert(ndhalves.toFloat64Array === Array(1d, 1d))
  }

  test("full") {
    var arr = NDArray.full(Shape(1, 2), 3f)
    assert(arr.shape === Shape(1, 2))
    assert(arr.toArray === Array(3f, 3f))

    // Float64 methods test
    arr = NDArray.full(Shape(1, 2), value = 5d, Context.cpu())
    assert(arr.toFloat64Array === Array (5d, 5d))
  }

  test("clip") {
    var ndarray = NDArray.empty(3, 2)
    ndarray.set(Array(1f, 2f, 3f, 4f, 5f, 6f))
    assert(NDArray.clip(ndarray, 2f, 5f).toArray === Array(2f, 2f, 3f, 4f, 5f, 5f))

    // Float64 methods test
    ndarray = NDArray.empty(shape = Shape(3, 2), dtype = DType.Float64)
    ndarray.set(Array(1d, 2d, 3d, 4d, 5d, 6d))
    assert(NDArray.clip(ndarray, 2d, 5d).toFloat64Array === Array(2d, 2d, 3d, 4d, 5d, 5d))
  }

  test("sqrt") {
    var ndarray = NDArray.empty(4, 1)
    ndarray.set(Array(0f, 1f, 4f, 9f))
    assert(NDArray.sqrt(ndarray).toArray === Array(0f, 1f, 2f, 3f))

    // Float64 methods test
    ndarray = NDArray.empty(shape = Shape(4, 1), dtype = DType.Float64)
    ndarray.set(Array(0d, 1d, 4d, 9d))
    assert(NDArray.sqrt(ndarray).toFloat64Array === Array(0d, 1d, 2d, 3d))
  }

  test("rsqrt") {
    var ndarray = NDArray.array(Array(1f, 4f), shape = Shape(2, 1))
    assert(NDArray.rsqrt(ndarray).toArray === Array(1f, 0.5f))

    // Float64 methods test
    ndarray = NDArray.array(Array(1d, 4d, 25d), shape = Shape(3, 1), Context.cpu())
    assert(NDArray.rsqrt(ndarray).toFloat64Array === Array(1d, 0.5d, 0.2d))
  }

  test("norm") {
    var ndarray = NDArray.empty(3, 1)
    ndarray.set(Array(1f, 2f, 3f))
    var normed = NDArray.norm(ndarray)
    assert(normed.shape === Shape(1))
    assert(normed.toScalar === math.sqrt(14.0).toFloat +- 1e-3f)

    // Float64 methods test
    ndarray = NDArray.empty(shape = Shape(3, 1), dtype = DType.Float64)
    ndarray.set(Array(1d, 2d, 3d))
    normed = NDArray.norm(ndarray)
    assert(normed.get.dtype === DType.Float64)
    assert(normed.shape === Shape(1))
    assert(normed.toFloat64Scalar === math.sqrt(14.0) +- 1e-3d)
  }

  test("one hot encode") {
    val indices = NDArray.array(Array(1f, 0f, 2f), shape = Shape(3))
    val array = NDArray.empty(3, 3)
    NDArray.onehotEncode(indices, array)
    assert(array.shape === Shape(3, 3))
    assert(array.toArray === Array(0f, 1f, 0f,
                                   1f, 0f, 0f,
                                   0f, 0f, 1f))
  }

  test("dot") {
    val arr1 = NDArray.array(Array(1f, 2f), shape = Shape(1, 2))
    val arr2 = NDArray.array(Array(3f, 4f), shape = Shape(2, 1))
    val res = NDArray.dot(arr1, arr2)
    assert(res.shape === Shape(1, 1))
    assert(res.toArray === Array(11f))
  }

  test("arange") {
    for (i <- 0 until 5) {
      val start = scala.util.Random.nextFloat() * 5
      val stop = start + scala.util.Random.nextFloat() * 100
      val step = scala.util.Random.nextFloat() * 4
      val repeat = 1
      val result = (start.toDouble until stop.toDouble by step.toDouble)
              .flatMap(x => Array.fill[Float](repeat)(x.toFloat))
      val range = NDArray.arange(start = start, stop = Some(stop), step = step,
        repeat = repeat, ctx = Context.cpu(), dType = DType.Float32)
      assert(CheckUtils.reldiff(result.toArray, range.toArray) <= 1e-4f)
    }
  }

  test("power") {
    var arr = NDArray.array(Array(3f, 5f), shape = Shape(2, 1))

    var arrPower1 = NDArray.power(2f, arr)
    assert(arrPower1.shape === Shape(2, 1))
    assert(arrPower1.toArray === Array(8f, 32f))

    var arrPower2 = NDArray.power(arr, 2f)
    assert(arrPower2.shape === Shape(2, 1))
    assert(arrPower2.toArray === Array(9f, 25f))

    var arrPower3 = NDArray.power(arr, arr)
    assert(arrPower3.shape === Shape(2, 1))
    assert(arrPower3.toArray === Array(27f, 3125f))

    var arrPower4 = arr ** 2f

    assert(arrPower4.shape === Shape(2, 1))
    assert(arrPower4.toArray === Array(9f, 25f))

    var arrPower5 = arr ** arr
    assert(arrPower5.shape === Shape(2, 1))
    assert(arrPower5.toArray === Array(27f, 3125f))

    arr **= 2f
    assert(arr.shape === Shape(2, 1))
    assert(arr.toArray === Array(9f, 25f))

    arr.set(Array(3f, 5f))
    arr **= arr
    assert(arr.shape === Shape(2, 1))
    assert(arr.toArray === Array(27f, 3125f))

    // Float64 tests
    arr = NDArray.array(Array(3d, 5d), shape = Shape(2, 1))

    arrPower1 = NDArray.power(2d, arr)
    assert(arrPower1.shape === Shape(2, 1))
    assert(arrPower1.dtype === DType.Float64)
    assert(arrPower1.toFloat64Array === Array(8d, 32d))

    arrPower2 = NDArray.power(arr, 2d)
    assert(arrPower2.shape === Shape(2, 1))
    assert(arrPower2.dtype === DType.Float64)
    assert(arrPower2.toFloat64Array === Array(9d, 25d))

    arrPower3 = NDArray.power(arr, arr)
    assert(arrPower3.shape === Shape(2, 1))
    assert(arrPower3.dtype === DType.Float64)
    assert(arrPower3.toFloat64Array === Array(27d, 3125d))

    arrPower4 = arr ** 2f
    assert(arrPower4.shape === Shape(2, 1))
    assert(arrPower4.dtype === DType.Float64)
    assert(arrPower4.toFloat64Array === Array(9d, 25d))

    arrPower5 = arr ** arr
    assert(arrPower5.shape === Shape(2, 1))
    assert(arrPower5.dtype === DType.Float64)
    assert(arrPower5.toFloat64Array === Array(27d, 3125d))

    arr **= 2d
    assert(arr.shape === Shape(2, 1))
    assert(arr.dtype === DType.Float64)
    assert(arr.toFloat64Array === Array(9d, 25d))

    arr.set(Array(3d, 5d))
    arr **= arr
    assert(arr.shape === Shape(2, 1))
    assert(arr.dtype === DType.Float64)
    assert(arr.toFloat64Array === Array(27d, 3125d))
  }

  test("equal") {
    var arr1 = NDArray.array(Array(1f, 2f, 3f, 5f), shape = Shape(2, 2))
    var arr2 = NDArray.array(Array(1f, 4f, 3f, 6f), shape = Shape(2, 2))

    var arrEqual1 = NDArray.equal(arr1, arr2)
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.toArray === Array(1f, 0f, 1f, 0f))

    var arrEqual2 = NDArray.equal(arr1, 3f)
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.toArray === Array(0f, 0f, 1f, 0f))


    // Float64 methods test
    arr1 = NDArray.array(Array(1d, 2d, 3d, 5d), shape = Shape(2, 2))
    arr2 = NDArray.array(Array(1d, 4d, 3d, 6d), shape = Shape(2, 2))

    arrEqual1 = NDArray.equal(arr1, arr2)
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.dtype === DType.Float64)
    assert(arrEqual1.toFloat64Array === Array(1d, 0d, 1d, 0d))

    arrEqual2 = NDArray.equal(arr1, 3d)
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.dtype === DType.Float64)
    assert(arrEqual2.toFloat64Array === Array(0d, 0d, 1d, 0d))
  }

  test("not_equal") {
    var arr1 = NDArray.array(Array(1f, 2f, 3f, 5f), shape = Shape(2, 2))
    var arr2 = NDArray.array(Array(1f, 4f, 3f, 6f), shape = Shape(2, 2))

    var arrEqual1 = NDArray.notEqual(arr1, arr2)
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.toArray === Array(0f, 1f, 0f, 1f))

    var arrEqual2 = NDArray.notEqual(arr1, 3f)
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.toArray === Array(1f, 1f, 0f, 1f))

    // Float64 methods test

    arr1 = NDArray.array(Array(1d, 2d, 3d, 5d), shape = Shape(2, 2))
    arr2 = NDArray.array(Array(1d, 4d, 3d, 6d), shape = Shape(2, 2))

    arrEqual1 = NDArray.notEqual(arr1, arr2)
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.dtype === DType.Float64)
    assert(arrEqual1.toFloat64Array === Array(0d, 1d, 0d, 1d))

    arrEqual2 = NDArray.notEqual(arr1, 3d)
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.dtype === DType.Float64)
    assert(arrEqual2.toFloat64Array === Array(1d, 1d, 0d, 1d))

  }

  test("greater") {
    var arr1 = NDArray.array(Array(1f, 2f, 4f, 5f), shape = Shape(2, 2))
    var arr2 = NDArray.array(Array(1f, 4f, 3f, 6f), shape = Shape(2, 2))

    var arrEqual1 = arr1 > arr2
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.toArray === Array(0f, 0f, 1f, 0f))

    var arrEqual2 = arr1 > 2f
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.toArray === Array(0f, 0f, 1f, 1f))

    // Float64 methods test
    arr1 = NDArray.array(Array(1d, 2d, 4d, 5d), shape = Shape(2, 2))
    arr2 = NDArray.array(Array(1d, 4d, 3d, 6d), shape = Shape(2, 2))

    arrEqual1 = arr1 > arr2
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.dtype === DType.Float64)
    assert(arrEqual1.toFloat64Array === Array(0d, 0d, 1d, 0d))

    arrEqual2 = arr1 > 2d
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.dtype === DType.Float64)
    assert(arrEqual2.toFloat64Array === Array(0d, 0d, 1d, 1d))
  }

  test("greater_equal") {
    var arr1 = NDArray.array(Array(1f, 2f, 4f, 5f), shape = Shape(2, 2))
    var arr2 = NDArray.array(Array(1f, 4f, 3f, 6f), shape = Shape(2, 2))

    var arrEqual1 = arr1 >= arr2
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.toArray === Array(1f, 0f, 1f, 0f))

    var arrEqual2 = arr1 >= 2f
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.toArray === Array(0f, 1f, 1f, 1f))

    // Float64 methods test
    arr1 = NDArray.array(Array(1d, 2d, 4d, 5d), shape = Shape(2, 2))
    arr2 = NDArray.array(Array(1d, 4d, 3d, 6d), shape = Shape(2, 2))

    arrEqual1 = arr1 >= arr2
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.dtype === DType.Float64)
    assert(arrEqual1.toFloat64Array === Array(1d, 0d, 1d, 0d))

    arrEqual2 = arr1 >= 2d
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.dtype === DType.Float64)
    assert(arrEqual2.toFloat64Array === Array(0d, 1d, 1d, 1d))
  }

  test("lesser") {
    var arr1 = NDArray.array(Array(1f, 2f, 4f, 5f), shape = Shape(2, 2))
    var arr2 = NDArray.array(Array(1f, 4f, 3f, 6f), shape = Shape(2, 2))

    var arrEqual1 = arr1 < arr2
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.toArray === Array(0f, 1f, 0f, 1f))

    var arrEqual2 = arr1 < 2f
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.toArray === Array(1f, 0f, 0f, 0f))

    // Float64 methods test
    arr1 = NDArray.array(Array(1d, 2d, 4d, 5d), shape = Shape(2, 2))
    arr2 = NDArray.array(Array(1d, 4d, 3d, 6d), shape = Shape(2, 2))

    arrEqual1 = arr1 < arr2
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.dtype === DType.Float64)
    assert(arrEqual1.toFloat64Array === Array(0d, 1d, 0d, 1d))

    arrEqual2 = arr1 < 2d
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.dtype === DType.Float64)
    assert(arrEqual2.toFloat64Array === Array(1d, 0d, 0d, 0d))

  }

  test("lesser_equal") {
    var arr1 = NDArray.array(Array(1f, 2f, 4f, 5f), shape = Shape(2, 2))
    var arr2 = NDArray.array(Array(1f, 4f, 3f, 6f), shape = Shape(2, 2))

    var arrEqual1 = arr1 <= arr2
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.toArray === Array(1f, 1f, 0f, 1f))

    var arrEqual2 = arr1 <= 2f
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.toArray === Array(1f, 1f, 0f, 0f))

    // Float64 methods test
    arr1 = NDArray.array(Array(1d, 2d, 4d, 5d), shape = Shape(2, 2))
    arr2 = NDArray.array(Array(1d, 4d, 3d, 6d), shape = Shape(2, 2))

    arrEqual1 = arr1 <= arr2
    assert(arrEqual1.shape === Shape(2, 2))
    assert(arrEqual1.dtype === DType.Float64)
    assert(arrEqual1.toFloat64Array === Array(1d, 1d, 0d, 1d))

    arrEqual2 = arr1 <= 2d
    assert(arrEqual2.shape === Shape(2, 2))
    assert(arrEqual2.dtype === DType.Float64)
    assert(arrEqual2.toFloat64Array === Array(1d, 1d, 0d, 0d))
  }

  test("choose_element_0index") {
    val arr = NDArray.array(Array(1f, 2f, 3f, 4f, 6f, 5f), shape = Shape(2, 3))
    val indices = NDArray.array(Array(0f, 1f), shape = Shape(2))
    val res = NDArray.choose_element_0index(arr, indices)
    assert(res.toArray === Array(1f, 6f))
  }

  test("copy to") {
    var source = NDArray.array(Array(1f, 2f, 3f), shape = Shape(1, 3))
    var dest = NDArray.empty(1, 3)
    source.copyTo(dest)
    assert(dest.shape === Shape(1, 3))
    assert(dest.toArray === Array(1f, 2f, 3f))

    // Float64 methods test
    source = NDArray.array(Array(1d, 2d, 3d), shape = Shape(1, 3))
    dest = NDArray.empty(shape = Shape(1, 3), dtype = DType.Float64)
    source.copyTo(dest)
    assert(dest.dtype === DType.Float64)
    assert(dest.toFloat64Array === Array(1d, 2d, 3d))
  }

  test("abs") {
    val arr = NDArray.array(Array(-1f, -2f, 3f), shape = Shape(3, 1))
    assert(NDArray.abs(arr).toArray === Array(1f, 2f, 3f))
  }

  test("sign") {
    val arr = NDArray.array(Array(-1f, -2f, 3f), shape = Shape(3, 1))
    assert(NDArray.sign(arr).toArray === Array(-1f, -1f, 1f))
  }

  test("round") {
    val arr = NDArray.array(Array(1.5f, 2.1f, 3.7f), shape = Shape(3, 1))
    assert(NDArray.round(arr).toArray === Array(2f, 2f, 4f))
  }

  test("ceil") {
    val arr = NDArray.array(Array(1.5f, 2.1f, 3.7f), shape = Shape(3, 1))
    assert(NDArray.ceil(arr).toArray === Array(2f, 3f, 4f))
  }

  test("floor") {
    val arr = NDArray.array(Array(1.5f, 2.1f, 3.7f), shape = Shape(3, 1))
    assert(NDArray.floor(arr).toArray === Array(1f, 2f, 3f))
  }

  test("square") {
    val arr = NDArray.array(Array(1f, 2f, 3f), shape = Shape(3, 1))
    assert(NDArray.square(arr).toArray === Array(1f, 4f, 9f))
  }

  test("exp") {
    val arr = NDArray.ones(1)
    assert(NDArray.exp(arr).toScalar === 2.71828f +- 1e-3f)
  }

  test("log") {
    val arr = NDArray.empty(1)
    arr.set(10f)
    assert(NDArray.log(arr).toScalar === 2.302585f +- 1e-5f)
  }

  test("cos") {
    val arr = NDArray.empty(1)
    arr.set(12f)
    assert(NDArray.cos(arr).toScalar === 0.8438539f +- 1e-5f)
  }

  test("sin") {
    val arr = NDArray.empty(1)
    arr.set(12f)
    assert(NDArray.sin(arr).toScalar === -0.536572918f +- 1e-5f)
  }

  test("max") {
    val arr = NDArray.array(Array(1.5f, 2.1f, 3.7f), shape = Shape(3, 1))
    assert(NDArray.max(arr).toScalar === 3.7f +- 1e-3f)
  }

  test("maximum") {
    val arr1 = NDArray.array(Array(1.5f, 2.1f, 3.7f), shape = Shape(3, 1))
    val arr2 = NDArray.array(Array(4f, 1f, 3.5f), shape = Shape(3, 1))
    val arr = NDArray.maximum(arr1, arr2)
    assert(arr.shape === Shape(3, 1))
    assert(arr.toArray === Array(4f, 2.1f, 3.7f))

    // Float64 methods test
    val arr3 = NDArray.array(Array(1d, 2d, 3d), shape = Shape(3, 1))
    val maxArr = NDArray.maximum(arr3, 10d)
    assert(maxArr.shape === Shape(3, 1))
    assert(maxArr.toArray === Array(10d, 10d, 10d))
  }

  test("min") {
    val arr = NDArray.array(Array(1.5f, 2.1f, 3.7f), shape = Shape(3, 1))
    assert(NDArray.min(arr).toScalar === 1.5f +- 1e-3f)
  }

  test("minimum") {
    val arr1 = NDArray.array(Array(1.5f, 2.1f, 3.7f), shape = Shape(3, 1))
    val arr2 = NDArray.array(Array(4f, 1f, 3.5f), shape = Shape(3, 1))
    val arr = NDArray.minimum(arr1, arr2)
    assert(arr.shape === Shape(3, 1))
    assert(arr.toArray === Array(1.5f, 1f, 3.5f))

    // Float64 methods test
    val arr3 = NDArray.array(Array(4d, 5d, 6d), shape = Shape(3, 1))
    val minArr = NDArray.minimum(arr3, 5d)
    assert(minArr.shape === Shape(3, 1))
    assert(minArr.toFloat64Array === Array(4d, 5d, 5d))
  }

  test("sum") {
    var arr = NDArray.array(Array(1f, 2f, 3f, 4f), shape = Shape(2, 2))
    assert(NDArray.sum(arr).toScalar === 10f +- 1e-3f)

  }

  test("argmaxChannel") {
    val arr = NDArray.array(Array(1f, 2f, 4f, 3f), shape = Shape(2, 2))
    val argmax = NDArray.argmax_channel(arr)
    assert(argmax.shape === Shape(2))
    assert(argmax.toArray === Array(1f, 0f))
  }

  test("concatenate axis-0") {
    val arr1 = NDArray.array(Array(1f, 2f, 4f, 3f, 3f, 3f), shape = Shape(2, 3))
    val arr2 = NDArray.array(Array(8f, 7f, 6f), shape = Shape(1, 3))
    val arr = NDArray.concatenate(arr1, arr2)
    assert(arr.shape === Shape(3, 3))
    assert(arr.toArray === Array(1f, 2f, 4f, 3f, 3f, 3f, 8f, 7f, 6f))

    // Try concatenating float32 arr with float64 arr. Should get exception
    intercept[Exception] {
      val arr3 = NDArray.array(Array (5d, 6d, 7d), shape = Shape(1, 3))
      NDArray.concatenate(Array(arr1, arr3))
    }
  }

  test("concatenate axis-1") {
    val arr1 = NDArray.array(Array(1f, 2f, 3f, 4f), shape = Shape(2, 2))
    val arr2 = NDArray.array(Array(5f, 6f), shape = Shape(2, 1))
    val arr = NDArray.concatenate(Array(arr1, arr2), axis = 1)
    assert(arr.shape === Shape(2, 3))
    assert(arr.toArray === Array(1f, 2f, 5f, 3f, 4f, 6f))

    // Try concatenating float32 arr with float64 arr. Should get exception
    intercept[Exception] {
      val arr3 = NDArray.array(Array (5d, 6d), shape = Shape(2, 1))
      NDArray.concatenate(Array(arr1, arr3), axis = 1)
    }
  }

  test("transpose") {
    val arr = NDArray.array(Array(1f, 2f, 4f, 3f, 3f, 3f), shape = Shape(2, 3))
    assert(arr.toArray === Array(1f, 2f, 4f, 3f, 3f, 3f))
    assert(arr.T.shape === Shape(3, 2))
    assert(arr.T.toArray === Array(1f, 3f, 2f, 3f, 4f, 3f))
  }

  test("save and load with names") {
    val filename
      = s"${System.getProperty("java.io.tmpdir")}/ndarray-${sequence.getAndIncrement}.bin"
    try {
      val ndarray = NDArray.array(Array(1f, 2f, 3f), shape = Shape(3, 1))
      NDArray.save(filename, Map("local" -> ndarray))
      val (keys, arrays) = NDArray.load(filename)
      assert(keys.length === 1)
      assert(keys(0) === "local")
      assert(arrays.length === 1)
      val loadedArray = arrays(0)
      assert(loadedArray.shape === Shape(3, 1))
      assert(loadedArray.toArray === Array(1f, 2f, 3f))
      assert(loadedArray.dtype === DType.Float32)
    } finally {
      val file = new File(filename)
      file.delete()
    }

    // Try the same for Float64 array
    try {
      val ndarray = NDArray.array(Array(1d, 2d, 3d), shape = Shape(3, 1), ctx = Context.cpu())
      NDArray.save(filename, Map("local" -> ndarray))
      val (keys, arrays) = NDArray.load(filename)
      assert(keys.length === 1)
      assert(keys(0) === "local")
      assert(arrays.length === 1)
      val loadedArray = arrays(0)
      assert(loadedArray.shape === Shape(3, 1))
      assert(loadedArray.toArray === Array(1d, 2d, 3d))
      assert(loadedArray.dtype === DType.Float64)
    } finally {
      val file = new File(filename)
      file.delete()
    }
  }

  test("save and load without names") {
    val filename
      = s"${System.getProperty("java.io.tmpdir")}/ndarray-${sequence.getAndIncrement}.bin"
    try {
      val ndarray = NDArray.array(Array(1f, 2f, 3f), shape = Shape(3, 1))
      NDArray.save(filename, Array(ndarray))
      val (keys, arrays) = NDArray.load(filename)
      assert(keys.length === 0)
      assert(arrays.length === 1)
      val loadedArray = arrays(0)
      assert(loadedArray.shape === Shape(3, 1))
      assert(loadedArray.toArray === Array(1f, 2f, 3f))
      assert(loadedArray.dtype === DType.Float32)
    } finally {
      val file = new File(filename)
      file.delete()
    }

    // Try the same thing for Float64 array :

    try {
      val ndarray = NDArray.array(Array(1d, 2d, 3d), shape = Shape(3, 1), ctx = Context.cpu())
      NDArray.save(filename, Array(ndarray))
      val (keys, arrays) = NDArray.load(filename)
      assert(keys.length === 0)
      assert(arrays.length === 1)
      val loadedArray = arrays(0)
      assert(loadedArray.shape === Shape(3, 1))
      assert(loadedArray.toArray === Array(1d, 2d, 3d))
      assert(loadedArray.dtype === DType.Float64)
    } finally {
      val file = new File(filename)
      file.delete()
    }
  }

  test("get context") {
    val ndarray = NDArray.ones(3, 2)
    val ctx = ndarray.context
    assert(ctx.deviceType === "cpu")
    assert(ctx.deviceId === 0)
  }

  test("equals") {
    val ndarray1 = NDArray.array(Array(1f, 2f, 3f), shape = Shape(3, 1))
    val ndarray2 = NDArray.array(Array(1f, 2f, 3f), shape = Shape(3, 1))
    val ndarray3 = NDArray.array(Array(1f, 2f, 3f), shape = Shape(1, 3))
    val ndarray4 = NDArray.array(Array(3f, 2f, 3f), shape = Shape(3, 1))
    val ndarray5 = NDArray.array(Array(3d, 2d, 3d), shape = Shape(3, 1), ctx = Context.cpu())
    ndarray1 shouldEqual ndarray2
    ndarray1 shouldNot equal(ndarray3)
    ndarray1 shouldNot equal(ndarray4)
    ndarray5 shouldNot equal(ndarray3)
  }

  test("slice") {
    val arr = NDArray.array(Array(1f, 2f, 3f, 4f, 5f, 6f), shape = Shape(3, 2))

    val arr1 = arr.slice(1)
    assert(arr1.shape === Shape(1, 2))
    assert(arr1.toArray === Array(3f, 4f))

    val arr2 = arr.slice(1, 3)
    assert(arr2.shape === Shape(2, 2))
    assert(arr2.toArray === Array(3f, 4f, 5f, 6f))
  }

  test("at") {
    val arr = NDArray.array(Array(1f, 2f, 3f, 4f, 5f, 6f), shape = Shape(3, 2))

    val arr1 = arr.at(1)
    assert(arr1.shape === Shape(2))
    assert(arr1.toArray === Array(3f, 4f))
  }

  test("reshape") {
    val arr = NDArray.array(Array(1f, 2f, 3f, 4f, 5f, 6f), shape = Shape(3, 2))

    val arr1 = arr.reshape(Array(2, 3))
    assert(arr1.shape === Shape(2, 3))
    assert(arr1.toArray === Array(1f, 2f, 3f, 4f, 5f, 6f))

    arr.set(1f)
    assert(arr1.toArray === Array(1f, 1f, 1f, 1f, 1f, 1f))
  }

  test("dispose deps") {
    val arr1 = NDArray.ones(1, 2)
    val arr2 = NDArray.ones(1, 2)
    val arr3 = NDArray.ones(1, 2)

    val arrWithDeps = (arr1 + arr2) + arr3
    assert(arrWithDeps.dependencies.size === 4) // arr1 + arr2
    assert(arrWithDeps.dependencies.contains(arr1.handle))
    assert(arrWithDeps.dependencies.contains(arr2.handle))
    assert(arrWithDeps.dependencies.contains(arr3.handle))
    assert(!arr1.isDisposed)
    assert(!arr2.isDisposed)
    assert(!arr3.isDisposed)

    val arrNoDeps = (arr1 + arr2 + arr3).disposeDeps()
    assert(arrNoDeps.dependencies.isEmpty)
    assert(arr1.isDisposed)
    assert(arr2.isDisposed)
    assert(arr3.isDisposed)
  }

  test("dispose deps except") {
    val arr1 = NDArray.ones(1, 2)
    val arr2 = NDArray.ones(1, 2)
    val arr3 = NDArray.ones(1, 2)
    val arr1_2 = arr1 + arr2

    val arr = (arr1 + arr2 + arr1_2 + arr3).disposeDepsExcept(arr1_2)
    // since arr1_2 depends on arr1 & arr2
    // arr1 & arr2 will not be disposed either
    assert(arr.dependencies.size === 3)
    assert(arr.dependencies.contains(arr1.handle))
    assert(arr.dependencies.contains(arr2.handle))
    assert(arr.dependencies.contains(arr1_2.handle))
    assert(!arr1.isDisposed)
    assert(!arr2.isDisposed)
    assert(!arr1_2.isDisposed)
    assert(arr3.isDisposed)
  }

  test("serialize and deserialize") {
    val arr = NDArray.ones(1, 2) * 3
    val bytes = arr.serialize()
    val arrCopy = NDArray.deserialize(bytes)
    assert(arr === arrCopy)
    assert(arrCopy.dtype === DType.Float32)
  }

  test("dtype int32") {
    val arr = NDArray.ones(Shape(1, 2), dtype = DType.Int32) * 2
    assert(arr.dtype === DType.Int32)
    assert(arr.internal.getRaw.length === 8)
    assert(arr.internal.toFloatArray === Array(2f, 2f))
    assert(arr.internal.toIntArray === Array(2, 2))
    assert(arr.internal.toDoubleArray === Array(2d, 2d))
    assert(arr.internal.toByteArray === Array(2.toByte, 2.toByte))
  }

  test("dtype uint8") {
    val arr = NDArray.ones(Shape(1, 2), dtype = DType.UInt8) * 2
    assert(arr.dtype === DType.UInt8)
    assert(arr.internal.getRaw.length === 2)
    assert(arr.internal.toFloatArray === Array(2f, 2f))
    assert(arr.internal.toIntArray === Array(2, 2))
    assert(arr.internal.toDoubleArray === Array(2d, 2d))
    assert(arr.internal.toByteArray === Array(2.toByte, 2.toByte))
  }

  test("dtype float64") {
    val arr = NDArray.ones(Shape(1, 2), dtype = DType.Float64) * 2
    assert(arr.dtype === DType.Float64)
    assert(arr.internal.getRaw.length === 16)
    assert(arr.internal.toFloatArray === Array(2f, 2f))
    assert(arr.internal.toIntArray === Array(2, 2))
    assert(arr.internal.toDoubleArray === Array(2d, 2d))
    assert(arr.internal.toByteArray === Array(2.toByte, 2.toByte))
  }

  test("NDArray random module is generated properly") {
    val lam = NDArray.ones(1, 2)
    val rnd = NDArray.random.poisson(lam = Some(lam), shape = Some(Shape(3, 4)))
    val rnd2 = NDArray.random.poisson(lam = Some(1f), shape = Some(Shape(3, 4)),
      dtype = Some("float64"))
    assert(rnd.shape === Shape(1, 2, 3, 4))
    assert(rnd2.shape === Shape(3, 4))
    assert(rnd2.head.dtype === DType.Float64)
  }

  test("NDArray random module is generated properly - special case of 'normal'") {
    val mu = NDArray.ones(1, 2)
    val sigma = NDArray.ones(1, 2) * 2
    val rnd = NDArray.random.normal(mu = Some(mu), sigma = Some(sigma), shape = Some(Shape(3, 4)))
    val rnd2 = NDArray.random.normal(mu = Some(1f), sigma = Some(2f), shape = Some(Shape(3, 4)),
      dtype = Some("float64"))
    assert(rnd.shape === Shape(1, 2, 3, 4))
    assert(rnd2.shape === Shape(3, 4))
    assert(rnd2.head.dtype === DType.Float64)
  }

  test("Generated api") {
    // Without SomeConversion
    val arr3 = NDArray.ones(Shape(1, 2), dtype = DType.Float64)
    val arr4 = NDArray.ones(Shape(1), dtype = DType.Float64)
    val arr5 = NDArray.api.norm(arr3, ord = Some(1), out = Some(arr4))
    // With SomeConversion
    import org.apache.mxnet.util.OptionConversion._
    val arr = NDArray.ones(Shape(1, 2), dtype = DType.Float64)
    val arr2 = NDArray.ones(Shape(1), dtype = DType.Float64)
    NDArray.api.norm(arr, ord = 1, out = arr2)
    val result = NDArray.api.dot(arr2, arr2)
  }
}
