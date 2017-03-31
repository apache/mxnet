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
import ml.dmlc.mxnet.{NDArray, Optimizer}

/**
 * AdaDelta optimizer as described in Matthew D. Zeiler, 2012.
 * http://arxiv.org/abs/1212.5701
 *
 * @param rho Decay rate for both squared gradients and delta x.
 * @param epsilon The constant as described in the thesis
 * @param rescaleGradient rescaling factor of gradient.
 * @param clipGradient clip gradient in range [-clip_gradient, clip_gradient]
 * @param wd L2 regularization coefficient add to all the weights
 */
class AdaDelta(rho: Float = 0.05f, rescaleGradient: Float = 1.0f,
               epsilon: Float = 1e-8f, wd: Float = 0.0f,
               clipGradient: Float = 0f) extends Optimizer {

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

    val (accG, accDelta) = state.asInstanceOf[(NDArray, NDArray)]

    val newAccG = (this.rho * accG + (1.0f - this.rho) *
      resdGrad * resdGrad).disposeDepsExcept(accG, resdGrad)
    accG.set(newAccG)
    val currentDelta = (
      NDArray.sqrt(accDelta + this.epsilon) /
      NDArray.sqrt(accG + this.epsilon) * resdGrad).disposeDepsExcept(accDelta, accG, resdGrad)
    val newAccDelta = (this.rho * accDelta +
      (1.0f - this.rho) * currentDelta * currentDelta).disposeDepsExcept(accDelta, currentDelta)
    accDelta.set(newAccDelta)

    weight *= (1 - this.wd)
    weight -= currentDelta

    newAccG.dispose()
    newAccDelta.dispose()
    resdGrad.dispose()
    currentDelta.dispose()
  }

  override def createState(index: Int, weight: NDArray): (NDArray, NDArray) = {
    (NDArray.zeros(weight.shape, weight.context), // accumulated g
      NDArray.zeros(weight.shape, weight.context)) // accumulated delta
  }

  // Dispose the state it created
  override def disposeState(state: AnyRef): Unit = {
    if (state != null) {
      val (g, delta) = state.asInstanceOf[(NDArray, NDArray)]
      g.dispose()
      delta.dispose()
    }
  }

  override def serializeState(state: AnyRef): Array[Byte] = {
    if (state != null) {
      val (g, delta) = state.asInstanceOf[(NDArray, NDArray)]
      SerializerUtils.serializeNDArrays(g, delta)
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

