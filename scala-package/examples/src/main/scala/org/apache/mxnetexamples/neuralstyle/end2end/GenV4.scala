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

package org.apache.mxnetexamples.neuralstyle.end2end

import org.apache.mxnet.{Context, Shape, Symbol, Xavier}


object GenV4 {

  def Conv(data: Symbol, numFilter: Int, workspace : Long, kernel: (Int, Int) = (5, 5),
           pad: (Int, Int) = (2, 2)): Symbol = {
    val sym1 = Symbol.api.Convolution(data = Some(data), num_filter = numFilter,
      kernel = Shape(kernel._1, kernel._2), workspace = Some(workspace),
      pad = Some(Shape(pad._1, pad._2)), no_bias = Some(false))
    val sym2 = Symbol.api.BatchNorm(data = Some(sym1), fix_gamma = Some(false))
    val sym3 = Symbol.api.LeakyReLU(data = Some(sym2), act_type = Some("leaky"))
    sym2.dispose()
    sym1.dispose()
    sym3
  }

  def getGenerator(prefix: String, imHw: (Int, Int)): Symbol = {
    val data = Symbol.Variable(s"${prefix}_data")

    var conv1_1 = Conv(data, 48, 4096)
    val conv2_1 = Conv(conv1_1, 32, 4096)
    var conv3_1 = Conv(conv2_1, 64, 4096, (3, 3), (1, 1))
    var conv4_1 = Conv(conv3_1, 32, 4096)
    var conv5_1 = Conv(conv4_1, 48, 4096)
    var conv6_1 = Conv(conv5_1, 32, 4096)
    var out = Symbol.api.Convolution(data = Some(conv6_1), num_filter = 3, kernel = Shape(3, 3),
      pad = Some(Shape(1, 1)), no_bias = Some(true), workspace = Some(4096))
    out = Symbol.api.BatchNorm(data = Some(out), fix_gamma = Some(false))
    out = Symbol.api.Activation(data = Some(out), act_type = "tanh")
    val rawOut = (out * 128) + 128
    val norm = Symbol.api.SliceChannel(data = Some(rawOut), num_outputs = 3)
    val rCh = norm.get(0) - 123.68f
    val gCh = norm.get(1) - 116.779f
    val bCh = norm.get(2) - 103.939f
    val normOut = Symbol.api.Concat(data = Array(rCh, gCh, bCh), num_args = 3)
    normOut * 0.4f + data * 0.6f
  }

  def getModule(prefix: String, dShape: Shape, ctx: Context, isTrain: Boolean = true): Module = {
    val sym = getGenerator(prefix, (dShape(2), dShape(3)))
    val (dataShapes, forTraining, inputsNeedGrad) = {
      val dataShape = Map(s"${prefix}_data" -> dShape)
      if (isTrain) (dataShape, true, true)
      else (dataShape, false, false)
    }
    val mod = new Module(symbol = sym, context = ctx,
      dataShapes = dataShapes,
      initializer = new Xavier(magnitude = 2f),
      forTraining = forTraining, inputsNeedGrad = inputsNeedGrad)
    mod
  }
}
