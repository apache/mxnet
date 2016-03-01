package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.{NDArray, Optimizer, LRScheduler}
import ml.dmlc.mxnet.NDArrayConversions._

/**
 * Adam optimizer as described in [King2014]
 *
 * [King2014] Diederik Kingma, Jimmy Ba,
 * Adam: A Method for Stochastic Optimization,
 * http://arxiv.org/abs/1412.6980
 *
 * <b>WARNING</b>
 * TODO: This class has NOT been tested yet.
 * And there exists potential <b>memory leak</b> in the implementation
 *
 * @author Yuan Tang, Yizhi Liu
 *
 * @param learningRate Float, Step size.
 * @param beta1 Float, Exponential decay rate for the first moment estimates.
 * @param beta2 Float, Exponential decay rate for the second moment estimates.
 * @param epsilon Float
 * @param decayFactor Float
 * @param wd Float, L2 regularization coefficient add to all the weights
 * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
 * @param lrScheduler The learning rate scheduler
 */
class Adam(var learningRate: Float = 0.002f, val beta1: Float = 0.9f, val beta2: Float = 0.999f,
           val epsilon: Float = 1e-8f, val decayFactor: Float = 1-1e-8f, val wd: Float = 0.0f,
           val clipGradient: Float = 0f, val lrScheduler: LRScheduler = null) extends Optimizer {

  protected var time: Int = 0
  protected var timeFirstIndex: Option[Int] = None

  if (lrScheduler != null) {
    lrScheduler.baseLR = learningRate
  }

  /**
   * Update the parameters.
   * @param index An unique integer key used to index the parameters
   * @param weight weight ndarray
   * @param grad grad ndarray
   * @param state NDArray or other objects returned by initState
   *              The auxiliary state used in optimization.
   */
  override def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit = {
    val lr =
      (if (lrScheduler != null) {
        val scheduledLr = lrScheduler(numUpdate)
        updateCount(index)
        scheduledLr
      } else {
        this.learningRate
      }) * lrScale.getOrElse(index, 1f)

    val (mean, variance) = state.asInstanceOf[(NDArray, NDArray)]

    // increment time only when the first parameters is called
    if (timeFirstIndex == None) {
      timeFirstIndex = Option(index)
      time = 0
    } else if (timeFirstIndex.get == index) {
      time += 1
    }

    val t1: Int = time + 1
    learningRate = (lr * math.sqrt(1.0 - math.pow(beta2, t1)) / (1.0 - math.pow(beta1, t1))).toFloat
    val beta1t = beta1 * math.pow(decayFactor, t1 - 1).toFloat

    var resdGrad = grad * rescaleGrad
    if (clipGradient != 0f) {
      resdGrad = NDArray.clip(resdGrad, -clipGradient, clipGradient)
    }

    val meanT = beta1t * mean + (1.0 - beta1t) * resdGrad
    val varianceT = beta2 * variance + (1.0f - beta2) * resdGrad * resdGrad

    var step = learningRate * meanT / (NDArray.sqrt(varianceT) + epsilon)

    if (wd > 0.0f) {
      step += lr * wd * weight
    }

    weight += -step
    mean.set(meanT)
    variance.set(varianceT)
  }

  // Create additional optimizer state: mean, variance
  override def createState(index: Int, weight: NDArray): (NDArray, NDArray) = {
    timeFirstIndex = None // time is incremented only on the first index
    (NDArray.zeros(weight.shape, weight.context), // mean
      NDArray.zeros(weight.shape, weight.context)) // variance
  }
}
