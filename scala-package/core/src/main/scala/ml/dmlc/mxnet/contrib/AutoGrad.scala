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

package ml.dmlc.mxnet.contrib

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet._
import scala.collection.mutable.ArrayBuffer

object AutoGrad {

  /**
   * Turn on or turn off operator recording.
   *
   * @param recording Boolean
   */
  def setRecording(recording: Boolean): Unit = {
    val flag = if (recording) 1 else 0
    checkCall(_LIB.mxAutogradSetRecording(flag))
  }

  /**
  * Mark NDArrays as variables to compute gradient for autograd.
  *
  * @param variables array of NDArray
  */
  def markVariables(variables: Array[NDArray]): Unit = {
    val variableHandles = variables.map(_.handle)
    checkCall(_LIB.mxAutogradMarkVariables(variableHandles))
  }

  /**
  * Compute the gradients of outputs w.r.t variables.
  *
  * @param outputs array of NDArray
  * @return gradients array of NDArray
  */
  def computeGradient(outputs: Array[NDArray]): Array[NDArray] = {
    val outputHandles = outputs.map(_.handle)

    val gradHandles = ArrayBuffer.empty[NDArrayHandle]
    checkCall(_LIB.mxAutogradComputeGradient(outputHandles, gradHandles))
    gradHandles.map(new NDArray(_)).toArray
  }

  /**
  * Return function that computes both gradient of arguments and loss value.
  *
  * @param func a Scala function
  *                 The forward (loss) function.
  * @return gradAndLossFunc a Scala function
  *                A function that would compute both the gradient of arguments and loss value.
  */
  def gradAndLoss(func: Array[NDArray] => Array[NDArray]):
    Array[NDArray] => (Array[NDArray], Array[NDArray]) = {
    // Wrapped function
    def wrapped(args: Array[NDArray]): (Array[NDArray], Array[NDArray]) = {
      markVariables(args)
      setRecording(true)
      val outputs = func(args)
      setRecording(false)
      val gradVals = computeGradient(outputs)
      (gradVals, outputs)
    }
    wrapped
  }

  /**
  * Return function that computes gradient of arguments.
  *
  * @param func a Scala function
  *                 The forward (loss) function.
  * @return gradFunc a Scala function
  *                A function that would compute the gradient of arguments.
  */
  def grad(func: Array[NDArray] => Array[NDArray]): Array[NDArray] => Array[NDArray] = {
    def wrapped(args: Array[NDArray]): Array[NDArray] = {
      gradAndLoss(func)(args)._1
    }
    wrapped
  }
}
