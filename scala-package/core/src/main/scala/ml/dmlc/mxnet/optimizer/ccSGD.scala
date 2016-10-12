package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.{Optimizer, LRScheduler, NDArray}
import ml.dmlc.mxnet.NDArrayConversions._
import ml.dmlc.mxnet.Base._


/**
 * A very simple SGD optimizer with momentum and weight regularization.
 * Implemented in C++.
 *
 * @author Depeng Liang
 *
 * @param learningRate Float, Step size.
 * @param momentum Float, momentum value.
 * @param rescaleGradient Float, rescaling factor of gradient.
 * @param wd Float, L2 regularization coefficient add to all the weights
 * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
 * @param lrScheduler The learning rate scheduler
 */
class ccSGD(val learningRate: Float = 0.01f, val momentum: Float = 0.0f,
            val wd: Float = 0.0001f, val rescaleGradient: Float = 1.0f,
            val clipGradient: Float = -1f, val lrScheduler: LRScheduler = null
    ) extends Optimizer {

  if (lrScheduler != null) {
    lrScheduler.baseLR = learningRate
  }

  private val optCreator = new OptimizerCreatorRef
  private val optHandle = new OptimizerHandleRef

  checkCall(_LIB.mxOptimizerFindCreator("ccsgd", optCreator))
  private val paramKeys = Array("momentum", "rescale_grad", "clip_gradient")
  private val paramvals = Array(s"$momentum", s"$rescaleGradient", s"$clipGradient")
  checkCall(_LIB.mxOptimizerCreateOptimizer(
    optCreator.value, paramKeys.length, paramKeys, paramvals, optHandle))

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
    checkCall(_LIB.mxOptimizerUpdate(optHandle.value, index, weight.handle, grad.handle, lr, wd))
  }

  // Create additional optimizer state such as momentum.
  override def createState(index: Int, weight: NDArray): AnyRef = {
    null
  }

  // Dispose the state it created
  override def disposeState(state: AnyRef): Unit = {}

  /**
   * Free the optimizer handle.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    checkCall(_LIB.mxOptimizerFree(optHandle.value))
  }
}
