package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.{Optimizer, LRScheduler, NDArray}
import ml.dmlc.mxnet.NDArrayConversions._

/**
 * SGD with nesterov.
 * It is implemented according to
 * https://github.com/torch/optim/blob/master/sgd.lua
 *
 * @author Depeng Liang
 *
 * @param learningRate Float, Step size.
 * @param momentum Float, momentum value.
 * @param wd Float, L2 regularization coefficient add to all the weights
 * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
 * @param lrScheduler The learning rate scheduler
 */
class NAG(val learningRate: Float = 0.01f, val momentum: Float = 0.0f,
          val wd: Float = 0.0001f, val clipGradient: Float = 0f,
          val lrScheduler: LRScheduler = null) extends Optimizer {

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
    // TODO(bing) implement wd_bias, wd_gamma, wd_beta (copy from python package)
    val lr =
      (if (lrScheduler != null) {
        val scheduledLr = lrScheduler(numUpdate)
        updateCount(index)
        scheduledLr
      } else {
        this.learningRate
      }) * lrScale.getOrElse(index, 1f)

    val wd = getWd(index, this.wd)
    var resdGrad = grad * this.rescaleGrad
    if (clipGradient != 0f) {
      // to get rid of memory leak
      val oldResdGrad = resdGrad
      resdGrad = NDArray.clip(resdGrad, -clipGradient, clipGradient)
      oldResdGrad.dispose()
    }

    if (state != null) {
      val mom = state.asInstanceOf[NDArray]
      mom *= momentum
      resdGrad += wd * weight
      mom += resdGrad
      resdGrad += momentum * mom
      weight += -lr * resdGrad
    } else {
      require(momentum == 0f)
      // adder = -lr * (resdGrad + this.wd * weight)
      // we write in this way to get rid of memory leak
      val adder = this.wd * weight
      adder += resdGrad
      adder *= (-lr)
      weight += adder
      adder.dispose()
    }

    resdGrad.dispose()
  }

  // Create additional optimizer state such as momentum.
  override def createState(index: Int, weight: NDArray): AnyRef = {
    if (momentum == 0.0f) {
      null
    } else {
      NDArray.zeros(weight.shape, weight.context)
    }
  }

  // Dispose the state it created
  override def disposeState(state: AnyRef): Unit = {
    if (state != null) {
      state.asInstanceOf[NDArray].dispose()
    }
  }
}
