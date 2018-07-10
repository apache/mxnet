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


object GenV3 {
  def Conv(data: Symbol, numFilter: Int, kernel: (Int, Int) = (5, 5),
           pad: (Int, Int) = (2, 2), stride: (Int, Int) = (2, 2)): Symbol = {
    val sym1 = Symbol.api.Convolution(data = Some(data), num_filter = numFilter,
      kernel = Shape(kernel._1, kernel._2), stride = Some(Shape(stride._1, stride._2)),
      pad = Some(Shape(pad._1, pad._2)), no_bias = Some(false))
    val sym2 = Symbol.api.BatchNorm(data = Some(sym1), fix_gamma = Some(false))
    val sym3 = Symbol.api.LeakyReLU(data = Some(sym2), act_type = Some("leaky"))
    sym2.dispose()
    sym1.dispose()
    sym3
  }

  def Deconv(data: Symbol, numFilter: Int, imHw: (Int, Int),
             kernel: (Int, Int) = (7, 7), pad: (Int, Int) = (2, 2), stride: (Int, Int) = (2, 2),
             crop: Boolean = true, out: Boolean = false): Symbol = {
    var sym = Symbol.api.Deconvolution(data = Some(data), num_filter = numFilter,
      kernel = Shape(kernel._1, kernel._2), stride = Some(Shape(stride._1, stride._2)),
      pad = Some(Shape(pad._1, pad._2)), no_bias = Some(true))
    if (crop) sym = Symbol.api.Crop(data = Array(sym), offset = Some(Shape(1, 1)),
      h_w = Some(Shape(imHw._1, imHw._2)), num_args = 1)
    sym = Symbol.api.BatchNorm(data = Some(sym), fix_gamma = Some(false))
    if (out == false) Symbol.api.LeakyReLU(data = Some(sym), act_type = Some("leaky"))
    else Symbol.api.Activation(data = Some(sym), act_type = "tanh")
  }

  def getGenerator(prefix: String, imHw: (Int, Int)): Symbol = {
    val data = Symbol.Variable(s"${prefix}_data")
    val conv1 = Conv(data, 64) // 192
    val conv1_1 = Conv(conv1, 48, kernel = (3, 3), pad = (1, 1), stride = (1, 1))
    val conv2 = Conv(conv1_1, 128) // 96
    val conv2_1 = Conv(conv2, 96, kernel = (3, 3), pad = (1, 1), stride = (1, 1))
    val conv3 = Conv(conv2_1, 256) // 48
    val conv3_1 = Conv(conv3, 192, kernel = (3, 3), pad = (1, 1), stride = (1, 1))
    val deconv1 = Deconv(conv3_1, 128, (imHw._1 / 4, imHw._2 / 4)) + conv2
    val conv4_1 = Conv(deconv1, 160, kernel = (3, 3), pad = (1, 1), stride = (1, 1))
    val deconv2 = Deconv(conv4_1, 64, (imHw._1 / 2, imHw._2 / 2)) + conv1
    val conv5_1 = Conv(deconv2, 96, kernel = (3, 3), pad = (1, 1), stride = (1, 1))
    val deconv3 = Deconv(conv5_1, 3, imHw, kernel = (8, 8), pad = (3, 3), out = true, crop = false)
    val rawOut = (deconv3 * 128) + 128
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
