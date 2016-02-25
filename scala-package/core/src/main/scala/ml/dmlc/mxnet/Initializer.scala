package ml.dmlc.mxnet

/**
 *
 * Base class for Initializer.
 *
 * @author Yuan Tang
 */
abstract class Initializer {

  /**
   * Initialize an Initializer
   *
   * @param name name of corrosponding ndarray
   * @param arr ndarray to be Initialized
   */
  def apply(name: String, arr: NDArray): Unit = {

    if (name.startsWith("upsampling")) {
      initBilinear(name, arr)
    } else if (name.endsWith("bias")) {
      initBias(name, arr)
    } else if (name.endsWith("gamma")) {
      initGamma(name, arr)
    } else if (name.endsWith("beta")) {
      initBeta(name, arr)
    } else if (name.endsWith("weight")) {
      initWeight(name, arr)
    } else if (name.endsWith("moving_mean")) {
      initZero(name, arr)
    } else if (name.endsWith("moving_var")) {
      initZero(name, arr)
    } else if (name.endsWith("moving_avg")) {
      initZero(name, arr)
    } else {
      initDefault(name, arr)
    }
  }

  protected def initBilinear(name: String, arr: NDArray): Unit = {
    val weight = Array.fill[Float](arr.size)(0.0f)
    val shape = arr.shape
    val f = shape(3) / 2.0f
    val c = (2 * f - 1 - f % 2) / (2.0f * f)

    (0 until arr.size).foreach { i =>
      val x = i % shape(3)
      val y = (i / shape(3)) % shape(2)
      weight(i) = (1 - math.abs(x / f - c)) * (1 - math.abs(y / f - c))
    }

    arr.set(NDArray.array(weight, shape))
  }

  protected def initZero(name: String, arr: NDArray): Unit = {
    arr.set(0f)
  }

  protected def initBias(name: String, arr: NDArray): Unit = {
    arr.set(0f)
  }

  protected def initGamma(name: String, arr: NDArray): Unit = {
    arr.set(1f)
  }

  protected def initBeta(name: String, arr: NDArray): Unit = {
    arr.set(0f)
  }

  protected def initWeight(name: String, arr: NDArray): Unit

  protected def initDefault(name: String, arr: NDArray): Unit = {
    throw new IllegalArgumentException(s"Unknown initialization pattern for $name.")
  }
}


/**
 * Initialize the weight with uniform [-scale, scale]
 *
 * @param scale The scale of uniform distribution
 */
class Uniform(protected val scale: Float = 0.07f) extends Initializer {
  override def initWeight(name: String, arr: NDArray): Unit = {
    Random.uniform(-scale, scale, out = arr)
  }
}


/**
 * Initialize the weight with normal(0, sigma)
 *
 * @param sigma Standard deviation for gaussian distribution.
 */
class Normal(protected val sigma: Float = 0.01f) extends Initializer {
  override def initWeight(name: String, arr: NDArray): Unit = {
    Random.normal(0, sigma, out = arr)
  }
}


/**
 * Initialize the weight with Xavier or similar initialization scheme.
 *
 * @param rndType Options are: "gaussian" or "uniform"
 * @param factorType Options are: "avg", "in", "out"
 * @param magnitude scale of random number range
 */
class Xavier(protected val rndType: String = "uniform",
             protected val factorType: String = "avg",
             protected val magnitude: Float = 3) extends Initializer {

  override def initWeight(name: String, arr: NDArray): Unit = {
    val shape = arr.shape
    val fanIn = shape.slice(1, shape.length).product
    val fanOut = shape(0)
    var factor = 1f

    factor = factorType match {
      case "avg" => (fanIn + fanOut) / 2f
      case "in" => fanIn
      case "out" => fanOut
      case _ => throw new IllegalArgumentException("Incorrect factor type")
    }
    val scale = math.sqrt(magnitude / factor).toFloat

    rndType match {
      case "uniform" => Random.uniform(-scale, scale, out = arr)
      case "gaussian" => Random.normal(0, scale, out = arr)
      case _ => throw new IllegalArgumentException("Unknown random type")
    }
  }
}
