package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.{Optimizer, LRScheduler, NDArray}
import ml.dmlc.mxnet.NDArrayConversions._

/**
 * A very simple SGD optimizer with momentum and weight regularization.
 * @author Yizhi Liu
 */
class SGD(val learningRate: Float = 0.01f, val momentum: Float = 0.0f,
          val wd: Float = 0.0001f, rescaleGrad: Float = 1f, val clipGradient: Float = 0f,
          val lrScheduler: LRScheduler = null) extends Optimizer(rescaleGrad: Float) {
  /**
   * Update the parameters.
   * @param index An unique integer key used to index the parameters
   * @param weight weight ndarray
   * @param grad grad ndarray
   * @param state NDArray or other objects returned by initState
   *              The auxiliary state used in optimization.
   */
  override def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit = {
    // TODO(bing) implement wd_bias, wd_gamma, wd_beta (copy from python package)
    val lr =
      (if (lrScheduler != null) {
        val scheduledLr = lrScheduler(numUpdate)
        updateCount(index)
        scheduledLr
      } else {
        this.learningRate
      }) * lrScale.getOrElse(index, 1f)

    var resdGrad = grad * rescaleGrad
    if (clipGradient != 0f) {
      resdGrad = NDArray._genericNDArrayFunction(
        "clip", Array(resdGrad, -clipGradient, clipGradient))(0)
    }
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

  // Create additional optimizer state such as momentum.
  override def createState(index: Int, weight: NDArray): AnyRef = {
    if (momentum == 0.0f) {
      null
    } else {
      NDArray.zeros(weight.shape, weight.context)
    }
  }
}
