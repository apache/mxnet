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

package ml.dmlc.mxnetexamples.visualization

import ml.dmlc.mxnet.Symbol

/**
 * @author Depeng Liang
 */
object Inception_V3 {

  def Conv(data: Symbol, numFilter: Int, kernel: (Int, Int) = (1, 1), stride: (Int, Int) = (1, 1),
      pad: (Int, Int) = (0, 0), name: String = "", suffix: String = ""): Symbol = {
    val conv = Symbol.Convolution(s"${name}${suffix}_conv2d")()(
        Map("data" -> data,
        "num_filter" -> numFilter, "kernel" -> s"$kernel", "stride" -> s"$stride"
        , "pad" -> s"$pad", "no_bias" -> true))
    val bn = Symbol.BatchNorm(s"${name}${suffix}_batchnorm")()(
        Map("data" -> conv, "fix_gamma" -> true))
    val act = Symbol.Activation(s"${name}${suffix}_relu")()(
        Map("data" -> bn, "act_type" -> "relu"))
    act
  }

  def Inception7A(
      data: Symbol,
      num_1x1: Int,
      num_3x3_red: Int, num_3x3_1: Int, num_3x3_2: Int,
      num_5x5_red: Int, num_5x5: Int,
      pool: String, proj: Int,
      name: String): Symbol = {
    val tower_1x1 = Conv(data, num_1x1, name = s"${name}_conv")
    var tower_5x5 = Conv(data, num_5x5_red,
        name = s"${name}_tower", suffix = "_conv")
    tower_5x5 = Conv(tower_5x5, num_5x5, kernel = (5, 5),
        pad = (2, 2), name = s"${name}_tower", suffix = "_conv_1")
    var tower_3x3 = Conv(data, num_3x3_red,
        name = s"${name}_tower_1", suffix = "_conv")
    tower_3x3 = Conv(tower_3x3, num_3x3_1, kernel = (3, 3),
        pad = (1, 1), name = s"${name}_tower_1", suffix = "_conv_1")
    tower_3x3 = Conv(tower_3x3, num_3x3_2, kernel = (3, 3),
        pad = (1, 1), name = s"${name}_tower_1", suffix = "_conv_2")
    val pooling = Symbol.Pooling(s"${pool}_pool_${name}_pool")()(
        Map("data" -> data, "kernel" -> "(3, 3)",
        "stride" -> "(1, 1)", "pad" -> "(1, 1)", "pool_type" -> pool))
    val cproj = Conv(pooling, proj, name = s"${name}_tower_2", suffix = "_conv")
    val concat = Symbol.Concat(s"ch_concat_${name}_chconcat")(
        tower_1x1, tower_5x5, tower_3x3, cproj)()
    concat
  }

  // First Downsample
  def Inception7B(
      data: Symbol,
      num_3x3: Int,
      num_d3x3_red: Int, num_d3x3_1: Int, num_d3x3_2: Int,
      pool: String,
      name: String): Symbol = {
    val tower_3x3 = Conv(data, num_3x3, kernel = (3, 3), pad = (0, 0),
        stride = (2, 2), name = s"${name}_conv")
    var tower_d3x3 = Conv(data, num_d3x3_red,
        name = s"${name}_tower", suffix = "_conv")
    tower_d3x3 = Conv(tower_d3x3, num_d3x3_1, kernel = (3, 3),
        pad = (1, 1), stride = (1, 1), name = s"${name}_tower", suffix = "_conv_1")
    tower_d3x3 = Conv(tower_d3x3, num_d3x3_2, kernel = (3, 3),
        pad = (0, 0), stride = (2, 2), name = s"${name}_tower", suffix = "_conv_2")
    val pooling = Symbol.Pooling(s"max_pool_${name}_pool")()(Map("data" -> data,
        "kernel" -> "(3, 3)", "stride" -> "(2, 2)", "pad" -> "(0,0)", "pool_type" -> "max"))
    val concat = Symbol.Concat(s"ch_concat_${name}_chconcat")(
        tower_3x3, tower_d3x3, pooling)()
    concat
  }

  // scalastyle:off parameterNum
  def Inception7C(
      data: Symbol,
      num_1x1: Int,
      num_d7_red: Int, num_d7_1: Int, num_d7_2: Int,
      num_q7_red: Int, num_q7_1: Int, num_q7_2: Int, num_q7_3: Int, num_q7_4: Int,
      pool: String, proj: Int,
      name: String): Symbol = {
    val tower_1x1 = Conv(data = data, numFilter = num_1x1,
        kernel = (1, 1), name = s"${name}_conv")
    var tower_d7 = Conv(data = data, numFilter = num_d7_red,
        name = s"${name}_tower", suffix = "_conv")
    tower_d7 = Conv(data = tower_d7, numFilter = num_d7_1, kernel = (1, 7),
        pad = (0, 3), name = s"${name}_tower", suffix = "_conv_1")
    tower_d7 = Conv(data = tower_d7, numFilter = num_d7_2, kernel = (7, 1),
        pad = (3, 0), name = s"${name}_tower", suffix = "_conv_2")
    var tower_q7 = Conv(data = data, numFilter = num_q7_red,
        name = s"${name}_tower_1", suffix = "_conv")
    tower_q7 = Conv(data = tower_q7, numFilter = num_q7_1, kernel = (7, 1),
        pad = (3, 0), name = s"${name}_tower_1", suffix = "_conv_1")
    tower_q7 = Conv(data = tower_q7, numFilter = num_q7_2, kernel = (1, 7),
        pad = (0, 3), name = s"${name}_tower_1", suffix = "_conv_2")
    tower_q7 = Conv(data = tower_q7, numFilter = num_q7_3, kernel = (7, 1),
        pad = (3, 0), name = s"${name}_tower_1", suffix = "_conv_3")
    tower_q7 = Conv(data = tower_q7, numFilter = num_q7_4, kernel = (1, 7),
        pad = (0, 3), name = s"${name}_tower_1", suffix = "_conv_4")
    val pooling = Symbol.Pooling(s"${pool}_pool_${name}_pool")()(
        Map("data" -> data, "kernel" -> "(3, 3)",
        "stride" -> "(1, 1)", "pad" -> "(1, 1)", "pool_type" -> pool))
    val cproj = Conv(data = pooling, numFilter = proj, kernel = (1, 1),
        name = s"${name}_tower_2", suffix = "_conv")
    // concat
    val concat = Symbol.Concat(s"ch_concat_${name}_chconcat")(
        tower_1x1, tower_d7, tower_q7, cproj)()
    concat
  }

  def Inception7D(
      data: Symbol,
      num_3x3_red: Int, num_3x3: Int,
      num_d7_3x3_red: Int, num_d7_1: Int, num_d7_2: Int, num_d7_3x3: Int,
      pool: String,
      name: String): Symbol = {
    var tower_3x3 = Conv(data = data, numFilter = num_3x3_red,
        name = s"${name}_tower", suffix = "_conv")
    tower_3x3 = Conv(data = tower_3x3, numFilter = num_3x3, kernel = (3, 3),
        pad = (0, 0), stride = (2, 2), name = s"${name}_tower", suffix = "_conv_1")
    var tower_d7_3x3 = Conv(data = data, numFilter = num_d7_3x3_red,
        name = s"${name}_tower_1", suffix = "_conv")
    tower_d7_3x3 = Conv(data = tower_d7_3x3, numFilter = num_d7_1,
        kernel = (1, 7), pad = (0, 3), name = s"${name}_tower_1", suffix = "_conv_1")
    tower_d7_3x3 = Conv(data = tower_d7_3x3, numFilter = num_d7_2,
        kernel = (7, 1), pad = (3, 0), name = s"${name}_tower_1", suffix = "_conv_2")
    tower_d7_3x3 = Conv(data = tower_d7_3x3, numFilter = num_d7_3x3,
        kernel = (3, 3), stride = (2, 2), name = s"${name}_tower_1", suffix = "_conv_3")
    val pooling = Symbol.Pooling(s"${pool}_pool_${name}_pool")()(
        Map("data" -> data, "kernel" -> "(3, 3)", "stride" -> "(2, 2)", "pool_type" -> pool))
    // concat
    val concat = Symbol.Concat(s"ch_concat_${name}_chconcat")(
        tower_3x3, tower_d7_3x3, pooling)()
    concat
  }

  def Inception7E(
      data: Symbol,
      num_1x1: Int,
      num_d3_red: Int, num_d3_1: Int, num_d3_2: Int,
      num_3x3_d3_red: Int, num_3x3: Int, num_3x3_d3_1: Int, num_3x3_d3_2: Int,
      pool: String, proj: Int,
      name: String): Symbol = {
    val tower_1x1 = Conv(data = data, numFilter = num_1x1,
        kernel = (1, 1), name = s"${name}_conv")
    val tower_d3 = Conv(data = data, numFilter = num_d3_red,
        name = s"${name}_tower", suffix = "_conv")
    val tower_d3_a = Conv(data = tower_d3, numFilter = num_d3_1, kernel = (1, 3),
        pad = (0, 1), name = s"${name}_tower", suffix = "_mixed_conv")
    val tower_d3_b = Conv(data = tower_d3, numFilter = num_d3_2, kernel = (3, 1),
        pad = (1, 0), name = s"${name}_tower", suffix = "_mixed_conv_1")
    var tower_3x3_d3 = Conv(data = data, numFilter = num_3x3_d3_red,
        name = s"${name}_tower_1", suffix = "_conv")
    tower_3x3_d3 = Conv(data = tower_3x3_d3, numFilter = num_3x3, kernel = (3, 3),
        pad = (1, 1), name = s"${name}_tower_1", suffix = "_conv_1")
    val tower_3x3_d3_a = Conv(data = tower_3x3_d3, numFilter = num_3x3_d3_1,
        kernel = (1, 3), pad = (0, 1), name = s"${name}_tower_1", suffix = "_mixed_conv")
    val tower_3x3_d3_b = Conv(data = tower_3x3_d3, numFilter = num_3x3_d3_2,
        kernel = (3, 1), pad = (1, 0), name = s"${name}_tower_1", suffix = "_mixed_conv_1")
    val pooling = Symbol.Pooling(s"${pool}_pool_${name}_pool")()(Map("data" -> data,
        "kernel" -> "(3, 3)", "stride" -> "(1, 1)", "pad" -> "(1, 1)", "pool_type" -> pool))
    val cproj = Conv(data = pooling, numFilter = proj, kernel = (1, 1),
        name = s"${name}_tower_2", suffix = "_conv")
    // concat
    val concat = Symbol.Concat(s"ch_concat_${name}_chconcat")(
        tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj)()
    concat
  }
  // scalastyle:on parameterNum

  def getSymbol(numClasses: Int = 1000): Symbol = {
    val data = Symbol.Variable("data")
    // stage 1
    val conv = Conv(data, 32, kernel = (3, 3), stride = (2, 2), name = "conv")
    val conv_1 = Conv(conv, 32, kernel = (3, 3), name = "conv_1")
    val conv_2 = Conv(conv_1, 64, kernel = (3, 3), pad = (1, 1), name = "conv_2")
    var pool = Symbol.Pooling("pool")()(Map("data" -> conv_2, "kernel" -> "(3, 3)",
        "stride" -> "(2, 2)", "pool_type" -> "max"))
    // stage 2
    val conv_3 = Conv(pool, 80, kernel = (1, 1), name = "conv_3")
    val conv_4 = Conv(conv_3, 192, kernel = (3, 3), name = "conv_4")
    val pool1 = Symbol.Pooling("pool1")()(Map("data" -> conv_4, "kernel" -> "(3, 3)",
        "stride" -> "(2, 2)", "pool_type" -> "max"))
    // stage 3
    val in3a = Inception7A(pool1, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 32, "mixed")
    val in3b = Inception7A(in3a, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 64, "mixed_1")
    val in3c = Inception7A(in3b, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 64, "mixed_2")
    val in3d = Inception7B(in3c, 384,
                       64, 96, 96,
                       "max", "mixed_3")
    // stage 4
    val in4a = Inception7C(in3d, 192,
                       128, 128, 192,
                       128, 128, 128, 128, 192,
                       "avg", 192, "mixed_4")
    val in4b = Inception7C(in4a, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192,
                       "avg", 192, "mixed_5")
    val in4c = Inception7C(in4b, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192,
                       "avg", 192, "mixed_6")
    val in4d = Inception7C(in4c, 192,
                       192, 192, 192,
                       192, 192, 192, 192, 192,
                       "avg", 192, "mixed_7")
    val in4e = Inception7D(in4d, 192, 320,
                       192, 192, 192, 192,
                       "max", "mixed_8")
    // stage 5
    val in5a = Inception7E(in4e, 320,
                       384, 384, 384,
                       448, 384, 384, 384,
                       "avg", 192, "mixed_9")
    val in5b = Inception7E(in5a, 320,
                       384, 384, 384,
                       448, 384, 384, 384,
                       "max", 192, "mixed_10")
    // pool
    pool = Symbol.Pooling("global_pool")()(Map("data" -> in5b,
        "kernel" -> "(8, 8)", "stride" -> "(1, 1)", "pool_type" -> "avg"))
    val flatten = Symbol.Flatten("flatten")()(Map("data" -> pool))
    val fc1 = Symbol.FullyConnected("fc1")()(
        Map("data" -> flatten, "num_hidden" -> numClasses))
    val softmax = Symbol.SoftmaxOutput("softmax")()(Map("data" -> fc1))
    softmax
  }
}
