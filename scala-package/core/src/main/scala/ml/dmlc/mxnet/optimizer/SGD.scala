package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.{Optimizer, LRScheduler, NDArray}
import ml.dmlc.mxnet.NDArrayConversions._

/**
 * A very simple SGD optimizer with momentum and weight regularization.
 * @author Yizhi Liu
 */
class SGD(private val learningRate: Float = 0.01f, private val momentum: Float = 0.0f,
          private val wd: Float = 0.0001f, private val clipGradient: Float = 0f,
          private val lrScheduler: LRScheduler = null) extends Optimizer {
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
      // adder = -lr * (resdGrad + wd * weight)
      // we write in this way to get rid of memory leak
      val adder = wd * weight
      adder += resdGrad
      adder *= (-lr)
      mom += adder
      weight += mom
      adder.dispose()
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
