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

package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

/**
 * Random Number interface of mxnet.
 */
object Random {
  /**
   * Generate uniform distribution in [low, high) with shape.
   *
   * @param low The lower bound of distribution.
   * @param high The upper bound of distribution.
   * @param shape Output shape of the NDArray generated.
   * @param ctx Context of output NDArray, will use default context if not specified.
   * @param out Output place holder
   * @return The result NDArray with generated result.
   */
  def uniform(low: Float,
              high: Float,
              shape: Shape = null,
              ctx: Context = null,
              out: NDArray = null): NDArray = {
    var outCopy = out
    if (outCopy != null) {
      require(shape == null && ctx == null, "shape and ctx is not needed when out is specified.")
    } else {
      require(shape != null, "shape is required when out is not specified")
      outCopy = NDArray.empty(shape, ctx)
    }
    NDArray.genericNDArrayFunctionInvoke("_random_uniform", Seq(low, high),
      Map("shape" -> outCopy.shape, "out" -> outCopy))
  }


  /**
   * Generate normal(Gaussian) distribution N(mean, stdvar^^2) with shape.
   *
   * @param loc The mean of the normal distribution.
   * @param scale The standard deviation of normal distribution.
   * @param shape Output shape of the NDArray generated.
   * @param ctx Context of output NDArray, will use default context if not specified.
   * @param out Output place holder
   * @return The result NDArray with generated result.
   */
  def normal(loc: Float,
             scale: Float,
             shape: Shape = null,
             ctx: Context = null,
             out: NDArray = null): NDArray = {
    var outCopy = out
    if (outCopy != null) {
      require(shape == null & ctx == null, "shape and ctx is not needed when out is specified.")
    } else {
      require(shape != null, "shape is required when out is not specified")
      outCopy = NDArray.empty(shape, ctx)
    }
    NDArray.genericNDArrayFunctionInvoke("_random_normal", Seq.empty[NDArray],
      Map("loc" -> loc, "scale" -> scale, "shape" -> outCopy.shape, "out" -> outCopy))
  }


  /**
   * Seed the random number generators in mxnet.
   *
   * This seed will affect behavior of functions in this module,
   * as well as results from executors that contains Random number
   * such as Dropout operators.
   *
   * @param seedState The random number seed to set to all devices.
   * @note The random number generator of mxnet is by default device specific.
   *       This means if you set the same seed, the random number sequence
   *       generated from GPU0 can be different from CPU.
   */
  def seed(seedState: Int): Unit = {
    checkCall(_LIB.mxRandomSeed(seedState))
  }
}
