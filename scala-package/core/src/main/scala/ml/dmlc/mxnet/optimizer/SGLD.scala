package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.{Optimizer, LRScheduler, NDArray}
import ml.dmlc.mxnet.NDArrayConversions._
import ml.dmlc.mxnet.Random

/**
 * Stochastic Langevin Dynamics Updater to sample from a distribution.
 *
 * @author Depeng Liang
 *
 * @param learningRate Float, Step size.
 * @param rescaleGradient Float, rescaling factor of gradient.
 * @param wd Float, L2 regularization coefficient add to all the weights
 * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
 * @param lrScheduler The learning rate scheduler
 */
class SGLD(val learningRate: Float = 0.01f, val rescaleGradient: Float = 1.0f,
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

    val adder = this.wd * weight
    adder += resdGrad
    adder *= -(lr / 2)
    val norm = Random.normal(0f, Math.sqrt(lr).toFloat, weight.shape, weight.context)
    adder += norm
    weight += adder
    adder.dispose()
    norm.dispose()
  }

  // Create additional optimizer state such as momentum.
  override def createState(index: Int, weight: NDArray): AnyRef = {
    null
  }

  // Dispose the state it created
  override def disposeState(state: AnyRef): Unit = {}
}
