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
object GenV4 {

  def Conv(data: Symbol, numFilter: Int, kernel: (Int, Int) = (5, 5),
      pad: (Int, Int) = (2, 2), stride: (Int, Int) = (2, 2)): Symbol = {
    var sym = Symbol.Convolution()()(Map("data" -> data, "num_filter" -> numFilter,
        "kernel" -> s"$kernel", "stride" -> s"$stride", "pad" -> s"$pad", "no_bias" -> false))
    sym = Symbol.BatchNorm()()(Map("data" -> sym, "fix_gamma" -> false))
    sym = Symbol.LeakyReLU()()(Map("data" -> sym, "act_type" -> "leaky"))
    sym
  }

  def Deconv(data: Symbol, numFilter: Int, imHw: (Int, Int), kernel: (Int, Int) = (6, 6),
      pad: (Int, Int) = (2, 2), stride: (Int, Int) = (2, 2), out: Boolean = false): Symbol = {
    var sym = Symbol.Deconvolution()()(Map("data" -> data, "num_filter" -> numFilter,
        "kernel" -> s"$kernel", "stride" -> s"$stride", "pad" -> s"$pad", "no_bias" -> true))
    sym = Symbol.BatchNorm()()(Map("data" -> sym, "fix_gamma" -> false))
    if (out == false) Symbol.LeakyReLU()()(Map("data" -> sym, "act_type" -> "leaky"))
    else Symbol.Activation()()(Map("data" -> sym, "act_type" -> "tanh"))
  }

  def getGenerator(prefix: String, imHw: (Int, Int)): Symbol = {
    val data = Symbol.Variable(s"${prefix}_data")

    var conv1_1 = Symbol.Convolution()()(Map("data" -> data, "num_filter" -> 48,
        "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "no_bias" -> false, "workspace" -> 4096))
    conv1_1 = Symbol.BatchNorm()()(Map("data" -> conv1_1, "fix_gamma" -> false))
    conv1_1 = Symbol.LeakyReLU()()(Map("data" -> conv1_1, "act_type" -> "leaky"))

    var conv2_1 = Symbol.Convolution()()(Map("data" -> conv1_1, "num_filter" -> 32,
        "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "no_bias" -> false, "workspace" -> 4096))
    conv2_1 = Symbol.BatchNorm()()(Map("data" -> conv2_1, "fix_gamma" -> false))
    conv2_1 = Symbol.LeakyReLU()()(Map("data" -> conv2_1, "act_type" -> "leaky"))

    var conv3_1 = Symbol.Convolution()()(Map("data" -> conv2_1, "num_filter" -> 64,
        "kernel" -> "(3, 3)", "pad" -> "(1, 1)", "no_bias" -> false, "workspace" -> 4096))
    conv3_1 = Symbol.BatchNorm()()(Map("data" -> conv3_1, "fix_gamma" -> false))
    conv3_1 = Symbol.LeakyReLU()()(Map("data" -> conv3_1, "act_type" -> "leaky"))

    var conv4_1 = Symbol.Convolution()()(Map("data" -> conv3_1, "num_filter" -> 32,
        "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "no_bias" -> false, "workspace" -> 4096))
    conv4_1 = Symbol.BatchNorm()()(Map("data" -> conv4_1, "fix_gamma" -> false))
    conv4_1 = Symbol.LeakyReLU()()(Map("data" -> conv4_1, "act_type" -> "leaky"))

    var conv5_1 = Symbol.Convolution()()(Map("data" -> conv4_1, "num_filter" -> 48,
        "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "no_bias" -> false, "workspace" -> 4096))
    conv5_1 = Symbol.BatchNorm()()(Map("data" -> conv5_1, "fix_gamma" -> false))
    conv5_1 = Symbol.LeakyReLU()()(Map("data" -> conv5_1, "act_type" -> "leaky"))

    var conv6_1 = Symbol.Convolution()()(Map("data" -> conv5_1, "num_filter" -> 32,
        "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "no_bias" -> true, "workspace" -> 4096))
    conv6_1 = Symbol.BatchNorm()()(Map("data" -> conv6_1, "fix_gamma" -> false))
    conv6_1 = Symbol.LeakyReLU()()(Map("data" -> conv6_1, "act_type" -> "leaky"))

    var out = Symbol.Convolution()()(Map("data" -> conv6_1, "num_filter" -> 3, "kernel" -> "(3, 3)",
        "pad" -> "(1, 1)", "no_bias" -> true, "workspace" -> 4096))
    out = Symbol.BatchNorm()()(Map("data" -> out, "fix_gamma" -> false))
    out = Symbol.Activation()()(Map("data" -> out, "act_type" -> "tanh"))
    val rawOut = (out * 128) + 128
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
