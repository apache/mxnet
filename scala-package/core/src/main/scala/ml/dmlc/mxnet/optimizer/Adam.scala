/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ml.dmlc.mxnet.optimizer

import ml.dmlc.mxnet.NDArrayConversions._
import ml.dmlc.mxnet.util.SerializerUtils
import ml.dmlc.mxnet.{LRScheduler, NDArray, Optimizer}

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
 * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
 * @param lrScheduler The learning rate scheduler
 */
class Adam(val learningRate: Float = 0.002f, beta1: Float = 0.9f, beta2: Float = 0.999f,
           epsilon: Float = 1e-8f, decayFactor: Float = 1-1e-8f, wd: Float = 0.0f,
           clipGradient: Float = 0f, lrScheduler: LRScheduler = null) extends Optimizer {

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
    var lr =
      (if (lrScheduler != null) {
        val scheduledLr = lrScheduler(numUpdate)
        updateCount(index)
        scheduledLr
      } else {
        this.learningRate
      })
    lr = getLr(index, lr)

    val (mean, variance) = state.asInstanceOf[(NDArray, NDArray)]

    // increment time only when the first parameters is called
    timeFirstIndex match {
      case Some(idx) =>
        if (idx == index) time += 1
      case None =>
        timeFirstIndex = Option(index)
        time = 0 // all parameters share the same time
    }

    val t1: Int = time + 1
    val learningRate = (lr *
      math.sqrt(1.0 - math.pow(beta2, t1)) /
      (1.0 - math.pow(beta1, t1))).toFloat
    val beta1t = beta1 * math.pow(decayFactor, t1 - 1).toFloat

    var resdGrad = grad * rescaleGrad
    if (clipGradient != 0f) {
      val oldResdGrad = resdGrad
      resdGrad = NDArray.clip(resdGrad, -clipGradient, clipGradient)
      oldResdGrad.dispose()
    }

    val meanT = (beta1t * mean + (1.0 - beta1t) * resdGrad)
      .disposeDepsExcept(mean, resdGrad)
    val varianceT = (beta2 * variance + (1.0f - beta2) * resdGrad * resdGrad)
      .disposeDepsExcept(variance, resdGrad)

    val step = (learningRate * meanT / (NDArray.sqrt(varianceT) + epsilon))
      .disposeDepsExcept(meanT, varianceT)

    val wd = this.getWd(index, this.wd)
    if (wd > 0.0f) {
      val stepDelta = lr * wd * weight
      step += stepDelta
      stepDelta.dispose()
    }

    weight -= step
    mean.set(meanT)
    variance.set(varianceT)

    meanT.dispose()
    varianceT.dispose()
    step.dispose()
    resdGrad.dispose()
  }

  // Create additional optimizer state: mean, variance
  override def createState(index: Int, weight: NDArray): (NDArray, NDArray) = {
    timeFirstIndex = None // time is incremented only on the first index
    (NDArray.zeros(weight.shape, weight.context), // mean
      NDArray.zeros(weight.shape, weight.context)) // variance
  }

  // Dispose the state it created
  override def disposeState(state: AnyRef): Unit = {
    if (state != null) {
      val (mean, variance) = state.asInstanceOf[(NDArray, NDArray)]
      mean.dispose()
      variance.dispose()
    }
  }

  override def serializeState(state: AnyRef): Array[Byte] = {
    if (state != null) {
      val (mean, variance) = state.asInstanceOf[(NDArray, NDArray)]
      SerializerUtils.serializeNDArrays(mean, variance)
    } else {
      null
    }
  }

  override def deserializeState(bytes: Array[Byte]): AnyRef = {
    if (bytes != null) {
      val ndArrays = SerializerUtils.deserializeNDArrays(bytes)
      require(ndArrays.size == 2, s"Got ${ndArrays.size} arrays, expected 2.")
      val state = (ndArrays(0), ndArrays(1))
      state.asInstanceOf[AnyRef]
    } else {
      null
    }
  }
}
