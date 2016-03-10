package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.{NDArray, Optimizer, LRScheduler}
import ml.dmlc.mxnet.NDArrayConversions._

/**
  * AdaGrad optimizer as described in Matthew D. Zeiler, 2012.
  * http://arxiv.org/pdf/1212.5701v1.pdf
  *
  * <b>WARNING</b>
  * TODO: This class has NOT been tested yet.
  * And there exists potential <b>memory leak</b> in the implementation
  *
  * @author Yuan Tang
  *
  * @param learningRate Float, Step size.
  * @param epsilon Float
  * @param rescaleGradient Float, rescaling factor of gradient.
  * @param wd Float, L2 regularization coefficient add to all the weights
  */
class AdaGrad(var learningRate: Float = 0.05f, val rescaleGradient: Float = 1.0f,
           val epsilon: Float = 1e-8f, val wd: Float = 0.0f) extends Optimizer {

  /**
    * Update the parameters.
    * @param index An unique integer key used to index the parameters
    * @param weight weight ndarray
    * @param grad grad ndarray
    * @param state NDArray or other objects returned by initState
    *              The auxiliary state used in optimization.
    */
  override def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit = {
    val lr = this.learningRate

    val resdGrad = rescaleGradient * grad
    val history = state.asInstanceOf[NDArray]
    history.set(history + resdGrad * resdGrad)
    weight.set(-lr * (resdGrad / NDArray.sqrt(history + this.epsilon) + this.wd * weight))
  }

  override def createState(index: Int, weight: NDArray): NDArray = {
    NDArray.zeros(weight.shape, weight.context)
  }
}
