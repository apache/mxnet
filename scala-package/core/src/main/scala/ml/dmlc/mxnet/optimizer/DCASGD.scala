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
import ml.dmlc.mxnet.util.SerializerUtils

/**
 * DCASGD optimizer with momentum and weight regularization.
 * Implementation of paper "Asynchronous Stochastic Gradient Descent with
 * Delay Compensation for Distributed Deep Learning"
 */
class DCASGD(val learningRate: Float = 0.01f, momentum: Float = 0.0f,
      lamda: Float = 0.04f, wd: Float = 0.0f, clipGradient: Float = 0f,
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

    var (mon, previousWeight) = state.asInstanceOf[(NDArray, NDArray)]

    val monUpdated = -lr * (resdGrad + wd * weight + this.lamda *
        resdGrad * resdGrad * (weight - previousWeight))
    monUpdated.disposeDepsExcept(resdGrad, weight, previousWeight)
    if (mon != null) {
      mon *= this.momentum
      mon += monUpdated
    } else {
      require(this.momentum == 0)
      mon = monUpdated
    }
    previousWeight.set(weight)
    weight += mon
    resdGrad.dispose()
  }

  // Create additional optimizer state such as momentum.
  override def createState(index: Int, weight: NDArray): (NDArray, NDArray) = {
    if (momentum == 0.0f) {
      (null, weight.copy())
    } else {
      (NDArray.zeros(weight.shape, weight.context, weight.dtype), weight.copy())
    }
  }

  // Dispose the state it created
  override def disposeState(state: AnyRef): Unit = {
    if (state != null) {
      val (mon, preWeight) = state.asInstanceOf[(NDArray, NDArray)]
      if (mon != null) mon.dispose()
      preWeight.dispose()
    }
  }

  override def serializeState(state: AnyRef): Array[Byte] = {
    if (state != null) {
      val (mon, preWeight) = state.asInstanceOf[(NDArray, NDArray)]
      if (mon != null) SerializerUtils.serializeNDArrays(mon, preWeight)
      else preWeight.serialize()
    } else {
      null
    }
  }

  override def deserializeState(bytes: Array[Byte]): AnyRef = {
    if (bytes != null) {
      val ndArrays = SerializerUtils.deserializeNDArrays(bytes)
      require(ndArrays.size <= 2, s"Got ${ndArrays.size} arrays, expected <= 2.")
      val state = {
        if (ndArrays.length == 1) (null, ndArrays(0))
        else (ndArrays(0), ndArrays(1))
      }
      state.asInstanceOf[AnyRef]
    } else {
      null
    }
  }
}
