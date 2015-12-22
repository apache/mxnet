import ml.dmlc.mxnet.{NDArray, Optimizer, LRScheduler}
import scala.math

/**
 * Adam optimizer as described in [King2014]
 *
 * [King2014] Diederik Kingma, Jimmy Ba,
 * Adam: A Method for Stochastic Optimization,
 * http://arxiv.org/abs/1412.6980
 *
 * @param learningRate Float, Step size.
 * @param beta1 Float, Exponential decay rate for the first moment estimates.
 * @param beta2 Float, Exponential decay rate for the second moment estimates.
 * @param epsilon Float
 * @param decayFactor Float
 * @param wd Float, L2 regularization coefficient add to all the weights
 * @param rescaleGrad Float, rescaling factor of gradient.
 * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
 * @param lrScheduler The learning rate scheduler
 */
class Adam(val learningRate: Float = 0.002f, val beta1: Float = 0.9f, val beta2: Float = 0.999f,
          val epsilon: Float = 0.00000001f, val decayFactor: Float = 1-0.00000001f, val wd: Float = 0.0f,
           rescaleGrad: Float = 1f, val clipGradient: Float = 0f,
          val lrScheduler: LRScheduler = null) extends Optimizer(rescaleGrad: Float) {

  protected var time: Int = 0
  protected var timeFirstIndex: Int
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

    var mean, variance  = state

    if (timeFirstIndex == null) {
      timeFirstIndex = index
      time = 0
    } else if (timeFirstIndex == index) {
      time += 1
    }

    val t1: Int = time + 1
    learningRate = (lr * math.sqrt(1.0 - math.pow(beta2, t1))/(1.0 - math.pow(beta1, t1)))
    val beta1t = beta1 * math.pow(decayFactor, t1 - 1)


    var grad = grad * rescaleGrad
    if (clipGradient != 0f) {
      grad = NDArray.clip(grad, -clipGradient, clipGradient)
    }

    // mean_t
    if (state != null) {
      val mom = state.asInstanceOf[NDArray]
      mom *= momentum
      mom += -lr * (grad + wd * weight)
      weight += mom
    } else {
      require(momentum == 0f)
      weight += -lr * (grad + wd * weight)
    }
  }

  // Create additional optimizer state: mean, variance
  override def createState(index: Int, weight: NDArray): AnyRef = {
    timeFirstIndex = null
    (NDArray.zeros(weight.shape, weight.context), // mean
      NDArray.zeros(weight.shape, weight.context)) // variance
  }
}
