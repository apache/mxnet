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

import ml.dmlc.mxnet.util.SerializerUtils
import ml.dmlc.mxnet.{NDArray, Optimizer, LRScheduler}
import ml.dmlc.mxnet.NDArrayConversions._

/**
 * RMSProp optimizer as described in Tieleman & Hinton, 2012.
 * http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
 *
 * @param learningRate Float, Step size.
 * @param gamma1 Float, decay factor of moving average for gradient, gradient^^2.
 * @param gamma2 Float, momentum factor of moving average for gradient.
 * @param rescaleGradient Float, rescaling factor of gradient.
 * @param wd Float, L2 regularization coefficient add to all the weights
 * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
 * @param lrScheduler The learning rate scheduler
 */
class RMSProp(val learningRate: Float = 0.002f, rescaleGradient: Float = 1.0f,
              gamma1: Float = 0.95f, gamma2: Float = 0.9f, wd: Float = 0.0f,
              lrScheduler: LRScheduler = null, clipGradient: Float = 0f) extends Optimizer {

  /**
   * Update the parameters.
   * @param index An unique integer key used to index the parameters
   * @param weight weight ndarray
   * @param grad grad ndarray
   * @param state NDArray or other objects returned by initState
   *              The auxiliary state used in optimization.
   */
  override def update(index: Int, weight: NDArray, grad: NDArray, state: AnyRef): Unit = {
    val lr = getLr(index, this.learningRate)
    val (n, g, delta) = state.asInstanceOf[(NDArray, NDArray, NDArray)]
    val wd = getWd(index, this.wd)

    var resdGrad = grad * this.rescaleGrad
    if (clipGradient != 0f) {
      val oldResdGrad = resdGrad
      resdGrad = NDArray.clip(resdGrad, -clipGradient, clipGradient)
      oldResdGrad.dispose()
    }

    val nUpdated = ((1 - this.gamma1) * (resdGrad * resdGrad) + this.gamma1 * n)
      .disposeDepsExcept(resdGrad, n)
    n.set(nUpdated)
    nUpdated.dispose()

    val gUpdated = ((1 - this.gamma1) * resdGrad + this.gamma1 * g)
      .disposeDepsExcept(resdGrad, g)
    g.set(gUpdated)
    gUpdated.dispose()

    val deltaUpdated =
      (this.gamma2 * delta - lr * (resdGrad / NDArray.sqrt(n - g * g + 1e-4f) + wd * weight))
      .disposeDepsExcept(delta, resdGrad, n, g, weight)
    delta.set(deltaUpdated)
    deltaUpdated.dispose()

    weight += delta
    resdGrad.dispose()
  }

  override def createState(index: Int, weight: NDArray): (NDArray, NDArray, NDArray) = {
    (NDArray.zeros(weight.shape, weight.context), // n
      NDArray.zeros(weight.shape, weight.context), // g
      NDArray.zeros(weight.shape, weight.context)) // delta
  }

  // Dispose the state it created
  override def disposeState(state: AnyRef): Unit = {
    if (state != null) {
      val (n, g, delta) = state.asInstanceOf[(NDArray, NDArray, NDArray)]
      n.dispose()
      g.dispose()
      delta.dispose()
    }
  }

  override def serializeState(state: AnyRef): Array[Byte] = {
    if (state != null) {
      val (n, g, delta) = state.asInstanceOf[(NDArray, NDArray, NDArray)]
      SerializerUtils.serializeNDArrays(n, g, delta)
    } else {
      null
    }
  }

  override def deserializeState(bytes: Array[Byte]): AnyRef = {
    if (bytes != null) {
      val ndArrays = SerializerUtils.deserializeNDArrays(bytes)
      require(ndArrays.size == 3, s"Got ${ndArrays.size} arrays, expected 3.")
      val state = (ndArrays(0), ndArrays(1), ndArrays(2))
      state.asInstanceOf[AnyRef]
    } else {
      null
    }
  }
}

