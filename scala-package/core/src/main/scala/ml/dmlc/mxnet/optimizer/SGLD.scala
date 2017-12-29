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

import ml.dmlc.mxnet.{Optimizer, LRScheduler, NDArray}
import ml.dmlc.mxnet.NDArrayConversions._
import ml.dmlc.mxnet.Random

/**
 * Stochastic Langevin Dynamics Updater to sample from a distribution.
 *
 * @param learningRate Float, Step size.
 * @param rescaleGradient Float, rescaling factor of gradient.
 * @param wd Float, L2 regularization coefficient add to all the weights
 * @param clipGradient Float, clip gradient in range [-clip_gradient, clip_gradient]
 * @param lrScheduler The learning rate scheduler
 */
class SGLD(val learningRate: Float = 0.01f, rescaleGradient: Float = 1.0f,
           wd: Float = 0.0001f, clipGradient: Float = 0f,
           lrScheduler: LRScheduler = null) extends Optimizer {

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

  override def serializeState(state: AnyRef): Array[Byte] = {
    throw new UnsupportedOperationException("SGLD does not have states")
  }

  override def deserializeState(bytes: Array[Byte]): AnyRef = {
    throw new UnsupportedOperationException("SGLD does not have states")
  }
}
