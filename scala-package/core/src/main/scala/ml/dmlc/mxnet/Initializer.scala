package ml.dmlc.mxnet

import ml.dmlc.mxnet.NDArray.{array, zeros, ones}

/**
 * Created by yuantang on 12/27/15.
 */
abstract class Initializer(name: String, protected var arr: NDArray) {

  def _initBilinear() = {
    var weight = Array.fill[Float](arr.size)(0.0f)
    val shape = arr.shape
    val f = shape(3) / 2.0f
    val c = (2 * f - 1 - f % 2) / (2.0f * f)

    (0 to (arr.size)).foreach { i =>
      var x = i % shape(3)
      var y = (i % shape(3)) % shape(2)
      weight(i) = (1 - math.abs(x / f - c)) * (1 - math.abs(y / f - c))
    }

    arr = array(weight)

  }

  def _initZero() = {
    arr = zeros(arr.size)
  }

  def _initBias() = {
    arr = zeros(arr.size)
  }

  def _initGamma() = {
    arr = ones(arr.size)
  }

  def _initBeta() = {
    arr = zeros(arr.size)
  }

  def _initWeight()

  def _initDefault()
}
