package ml.dmlc.mxnet

import ml.dmlc.mxnet.NDArray.{array, zeros, ones}


/**
 *
 * Base class for Initializer.
 *
 * @author Yuan Tang
 *
 * @param name name of corrosponding ndarray
 * @param arr ndarray to be Initialized
 */
abstract class Initializer(protected val name: String, protected var arr: NDArray) {

  if (name.startsWith("upsampling")) {
    _initBilinear()
  } else if (name.endsWith("bias")) {
    _initBias()
  } else if (name.endsWith("gamma")) {
    _initGamma()
  } else if (name.endsWith("beta")) {
    _initBeta()
  } else if (name.endsWith("weight")) {
    _initWeight()
  } else if (name.endsWith("moving_mean")) {
    _initZero()
  } else if (name.endsWith("moving_var")) {
    _initZero()
  } else if (name.endsWith("moving_avg")) {
    _initZero()
  } else {
    throw new IllegalArgumentException(s"Unknown initialization pattern for ${name}.")
  }

  def _initBilinear() = {
    val weight = Array.fill[Float](arr.size)(0.0f)
    val shape = arr.shape
    val f = shape(3) / 2.0f
    val c = (2 * f - 1 - f % 2) / (2.0f * f)

    (0 to (arr.size)).foreach { i =>
      val x = i % shape(3)
      val y = (i / shape(3)) % shape(2)
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
}


/**
 * Initialize the weight with uniform [-scale, scale]
 *
 * @param name name of corrosponding ndarray
 * @param arr ndarray to be Initialized
 * @param scale The scale of uniform distribution
 */
class Uniform(name: String, arr: NDArray, protected val scale: Float=0.07f) extends Initializer(name: String, arr: NDArray) {
  def _initWeight() = {
    Random.uniform(-scale, scale, out=arr)
  }
}


/**
 * Initialize the weight with normal(0, sigma)
 *
 * @param name name of corrosponding ndarray
 * @param arr ndarray to be Initialized
 * @param sigma Standard deviation for gaussian distribution.
 */
class Normal(name: String, arr: NDArray, protected val sigma: Float=0.01f) extends Initializer(name: String, arr: NDArray) {
  def _initWeight() = {
    Random.normal(0, sigma, out=arr)
  }
}


/**
 * Initialize the weight with Xavier or similar initialization scheme.
 *
 * @param name name of corrosponding ndarray
 * @param arr ndarray to be Initialized
 * @param rndType Options are: "gaussian" or "uniform"
 * @param factorType Options are: "avg", "in", "out"
 * @param magnitude scale of random number range
 */
class Xavier(name: String, arr: NDArray, protected val rndType: String ="uniform",
             protected val factorType: String ="avg", protected val magnitude: Int = 3)
  extends Initializer(name: String, arr: NDArray) {

  def _initWeight() = {
    val shape = arr.shape
    val fanIn = shape.slice(1, shape.length).product
    val fanOut = shape(0)
    var factor = 1

    factor = factorType match {
      case "avg" => (fanIn + fanOut) / 2
      case "in" => fanIn
      case "out" => fanOut
      case _ => throw new IllegalArgumentException("Incorrect factor type")
    }
    val scale = math.sqrt(magnitude / factor).toFloat

    rndType match {
      case "uniform" => Random.uniform(-scale, scale, out=arr)
      case "normal" => Random.normal(0, scale, out=arr)
      case _ => throw new IllegalArgumentException("Unknown random type")
    }
  }
}