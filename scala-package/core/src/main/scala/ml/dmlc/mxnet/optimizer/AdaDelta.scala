package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.{NDArray, Optimizer}
import ml.dmlc.mxnet.NDArrayConversions._

/**
  * AdaDelta optimizer as described in Matthew D. Zeiler, 2012.
  * http://arxiv.org/abs/1212.5701
  *
  * <b>WARNING</b>
  * TODO: This class has NOT been tested yet.
  * And there exists potential <b>memory leak</b> in the implementation
  *
  * @author Yuan Tang
  *
  * @param rho Float, Decay rate for both squared gradients and delta x.
  * @param epsilon Float
  * @param rescaleGradient Float, rescaling factor of gradient.
  * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
  * @param wd Float, L2 regularization coefficient add to all the weights
  */
class AdaDelta(var rho: Float = 0.05f, val rescaleGradient: Float = 1.0f,
               val epsilon: Float = 1e-8f, val wd: Float = 0.0f,
               val clipGradient: Float = 0f) extends Optimizer {

  /**
    * Update the parameters.
    * @param index An unique integer key used to index the parameters
    * @param weight weight ndarray
    * @param grad grad ndarray
    * @param state NDArray or other objects returned by initState
    *              The auxiliary state used in optimization.
    */
  override def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit = {

    var resdGrad = grad * this.rescaleGrad

    if (clipGradient != 0f) {
      val oldResdGrad = resdGrad
      resdGrad = NDArray.clip(resdGrad, -clipGradient, clipGradient)
      oldResdGrad.dispose()
    }

    val (acc_g, acc_delta) = state.asInstanceOf[(NDArray, NDArray)]

    acc_g.set(this.rho * acc_g + (1.0f - this.rho) * resdGrad * resdGrad)
    val current_delta = NDArray.sqrt(acc_delta + this.epsilon) /
      NDArray.sqrt(acc_g + this.epsilon) * resdGrad
    acc_delta.set(this.rho * acc_delta + (1.0f - this.rho) * current_delta * current_delta)

    weight -= current_delta + this.wd * weight

  }

  override def createState(index: Int, weight: NDArray): (NDArray, NDArray) = {
    (NDArray.zeros(weight.shape, weight.context), // accumulated g
      NDArray.zeros(weight.shape, weight.context)) // accumulated delta
  }
}

