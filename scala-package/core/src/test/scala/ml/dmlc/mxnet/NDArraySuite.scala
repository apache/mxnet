package ml.dmlc.mxnet

import org.scalatest.{FunSuite, BeforeAndAfterAll}
import ml.dmlc.mxnet.NDArrayConversions._

class NDArraySuite extends FunSuite with BeforeAndAfterAll {
  test("to java array") {
    val ndarray = NDArray.zeros(Array(2, 2))
    assert(ndarray.toArray === Array(0f, 0f, 0f, 0f))
  }

  test("plus") {
    val ndzeros = NDArray.zeros(Array(2, 1))
    val ndones = ndzeros + 1f
    assert(ndones.toArray === Array(1f, 1f))
    assert((ndones + ndzeros).toArray === Array(1f, 1f))
    assert((1 + ndones).toArray === Array(2f, 2f))
    // in-place
    ndones += ndones
    assert(ndones.toArray === Array(2f, 2f))
  }
}
