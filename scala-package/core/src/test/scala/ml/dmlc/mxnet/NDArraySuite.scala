package ml.dmlc.mxnet

import java.io.File
import java.util.concurrent.atomic.AtomicInteger

import ml.dmlc.mxnet.NDArrayConversions._
import org.scalatest.{Matchers, BeforeAndAfterAll, FunSuite}

class NDArraySuite extends FunSuite with BeforeAndAfterAll with Matchers {
  private val sequence: AtomicInteger = new AtomicInteger(0)

  test("to java array") {
    val ndarray = NDArray.zeros(2, 2)
    assert(ndarray.toArray === Array(0f, 0f, 0f, 0f))
  }

  test("to scalar") {
    val ndzeros = NDArray.zeros(1)
    assert(ndzeros.toScalar === 0f)
    val ndones = NDArray.ones(1)
    assert(ndones.toScalar === 1f)
  }

  test ("call toScalar on an ndarray which is not a scalar") {
    intercept[Exception] { NDArray.zeros(1, 1).toScalar }
  }

  test("size and shape") {
    val ndzeros = NDArray.zeros(4, 1)
    assert(ndzeros.shape === Shape(4, 1))
    assert(ndzeros.size === 4)
  }

  test("set scalar value") {
    val ndarray = NDArray.empty(2, 1)
    ndarray.set(10f)
    assert(ndarray.toArray === Array(10f, 10f))
  }

  test("copy from java array") {
    val ndarray = NDArray.empty(4, 1)
    ndarray.set(Array(1f, 2f, 3f, 4f))
    assert(ndarray.toArray === Array(1f, 2f, 3f, 4f))
  }

  test("plus") {
    val ndzeros = NDArray.zeros(2, 1)
    val ndones = ndzeros + 1f
    assert(ndones.toArray === Array(1f, 1f))
    assert((ndones + ndzeros).toArray === Array(1f, 1f))
    assert((1 + ndones).toArray === Array(2f, 2f))
    // in-place
    ndones += ndones
    assert(ndones.toArray === Array(2f, 2f))
  }

  test("minus") {
    val ndones = NDArray.ones(2, 1)
    val ndzeros = ndones - 1f
    assert(ndzeros.toArray === Array(0f, 0f))
    assert((ndones - ndzeros).toArray === Array(1f, 1f))
    assert((ndzeros - ndones).toArray === Array(-1f, -1f))
    assert((ndones - 1).toArray === Array(0f, 0f))
    // in-place
    ndones -= ndones
    assert(ndones.toArray === Array(0f, 0f))
  }

  test("multiplication") {
    val ndones = NDArray.ones(2, 1)
    val ndtwos = ndones * 2
    assert(ndtwos.toArray === Array(2f, 2f))
    assert((ndones * ndones).toArray === Array(1f, 1f))
    assert((ndtwos * ndtwos).toArray === Array(4f, 4f))
    ndtwos *= ndtwos
    // in-place
    assert(ndtwos.toArray === Array(4f, 4f))
  }

  test("division") {
    val ndones = NDArray.ones(2, 1)
    val ndzeros = ndones - 1f
    val ndhalves = ndones / 2
    assert(ndhalves.toArray === Array(0.5f, 0.5f))
    assert((ndhalves / ndhalves).toArray === Array(1f, 1f))
    assert((ndones / ndones).toArray === Array(1f, 1f))
    assert((ndzeros / ndones).toArray === Array(0f, 0f))
    ndhalves /= ndhalves
    // in-place
    assert(ndhalves.toArray === Array(1f, 1f))
  }

  test("clip") {
    val ndarray = NDArray.empty(3, 2)
    ndarray.set(Array(1f, 2f, 3f, 4f, 5f, 6f))
    assert(NDArray.clip(ndarray, 2f, 5f).toArray === Array(2f, 2f, 3f, 4f, 5f, 5f))
  }

  test("sqrt") {
    val ndarray = NDArray.empty(4, 1)
    ndarray.set(Array(0f, 1f, 4f, 9f))
    assert(NDArray.sqrt(ndarray).toArray === Array(0f, 1f, 2f, 3f))
  }

  test("rsqrt") {
    val ndarray = NDArray.array(Array(1f, 4f), shape = Shape(2, 1))
    assert(NDArray.rsqrt(ndarray).toArray === Array(1f, 0.5f))
  }

  test("norm") {
    val ndarray = NDArray.empty(3, 1)
    ndarray.set(Array(1f, 2f, 3f))
    val normed = NDArray.norm(ndarray)
    assert(normed.shape === Shape(1))
    assert(normed.toScalar === math.sqrt(14.0).toFloat +- 1e-3f)
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

  test("choose_element_0index") {
    val arr = NDArray.array(Array(1f, 2f, 3f, 4f, 6f, 5f), shape = Shape(2, 3))
    val indices = NDArray.array(Array(0f, 1f), shape = Shape(2))
    val res = NDArray.chooseElement0Index(arr, indices)
    assert(res.toArray === Array(1f, 6f))
  }

  test("copy to") {
    val source = NDArray.array(Array(1f, 2f, 3f), shape = Shape(1, 3))
    val dest = NDArray.empty(1, 3)
    source.copyTo(dest)
    assert(dest.shape === Shape(1, 3))
    assert(dest.toArray === Array(1f, 2f, 3f))
  }

  test("random uniform") {
    val matrix = NDArray.empty(3, 2)
    NDArray.randomUniform(0f, 1f, matrix)
    assert(matrix.shape === Shape(3, 2))
    val arr = matrix.toArray
    // scalastyle:off println
    println(s"Random Uniform: [${arr.mkString(",")}]")
    // scalastyle:on println
    arr.foreach { elem =>
      assert(elem > 0f && elem < 1f)
    }
  }

  test("random gaussian") {
    val matrix = NDArray.empty(3, 2)
    NDArray.randomGaussian(0f, 1f, matrix)
    assert(matrix.shape === Shape(3, 2))
    val arr = matrix.toArray
    // scalastyle:off println
    println(s"Random Gaussian: [${arr.mkString(",")}]")
    // scalastyle:on println
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

  test("min") {
    val arr = NDArray.array(Array(1.5f, 2.1f, 3.7f), shape = Shape(3, 1))
    assert(NDArray.min(arr).toScalar === 1.5f +- 1e-3f)
  }

  test("sum") {
    val arr = NDArray.array(Array(1f, 2f, 3f, 4f), shape = Shape(2, 2))
    assert(NDArray.sum(arr).toScalar === 10f +- 1e-3f)
  }

  test("argmaxChannel") {
    val arr = NDArray.array(Array(1f, 2f, 4f, 3f), shape = Shape(2, 2))
    val argmax = NDArray.argmaxChannel(arr)
    assert(argmax.shape === Shape(2))
    assert(argmax.toArray === Array(1f, 0f))
  }

  test("concatenate") {
    val arr1 = NDArray.array(Array(1f, 2f, 4f, 3f, 3f, 3f), shape = Shape(2, 3))
    val arr2 = NDArray.array(Array(8f, 7f, 6f), shape = Shape(1, 3))
    val arr = NDArray.concatenate(arr1, arr2)
    assert(arr.shape === Shape(3, 3))
    assert(arr.toArray === Array(1f, 2f, 4f, 3f, 3f, 3f, 8f, 7f, 6f))
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
    ndarray1 shouldEqual ndarray2
    ndarray1 shouldNot equal(ndarray3)
    ndarray1 shouldNot equal(ndarray4)
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
  }
}
