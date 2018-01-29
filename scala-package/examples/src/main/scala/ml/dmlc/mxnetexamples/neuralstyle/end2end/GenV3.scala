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

package ml.dmlc.mxnetexamples.neuralstyle.end2end

import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Xavier

/**
 * @author Depeng Liang
 */
object GenV3 {
  def Conv(data: Symbol, numFilter: Int, kernel: (Int, Int) = (5, 5),
      pad: (Int, Int) = (2, 2), stride: (Int, Int) = (2, 2)): Symbol = {
    var sym = Symbol.Convolution()()(Map("data" -> data, "num_filter" -> numFilter,
        "kernel" -> s"$kernel", "stride" -> s"$stride", "pad" -> s"$pad", "no_bias" -> false))
    sym = Symbol.BatchNorm()()(Map("data" -> sym, "fix_gamma" -> false))
    sym = Symbol.LeakyReLU()()(Map("data" -> sym, "act_type" -> "leaky"))
    sym
  }

  def Deconv(data: Symbol, numFilter: Int, imHw: (Int, Int),
      kernel: (Int, Int) = (7, 7), pad: (Int, Int) = (2, 2), stride: (Int, Int) = (2, 2),
      crop: Boolean = true, out: Boolean = false): Symbol = {
    var sym = Symbol.Deconvolution()()(Map("data" -> data, "num_filter" -> numFilter,
        "kernel" -> s"$kernel", "stride" -> s"$stride", "pad" -> s"$pad", "no_bias" -> true))
    if (crop) sym = Symbol.Crop()(sym)(
        Map("offset" -> "(1, 1)", "h_w" -> s"$imHw", "num_args" -> 1))
    sym = Symbol.BatchNorm()()(Map("data" -> sym, "fix_gamma" -> false))
    if (out == false) Symbol.LeakyReLU()()(Map("data" -> sym, "act_type" -> "leaky"))
    else Symbol.Activation()()(Map("data" -> sym, "act_type" -> "tanh"))
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
    val norm = Symbol.SliceChannel()(rawOut)(Map("num_outputs" -> 3))
    val rCh = norm.get(0) - 123.68f
    val gCh = norm.get(1) - 116.779f
    val bCh = norm.get(2) - 103.939f
    val normOut = Symbol.Concat()(rCh, gCh, bCh)() * 0.4f + data * 0.6f
    normOut
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
