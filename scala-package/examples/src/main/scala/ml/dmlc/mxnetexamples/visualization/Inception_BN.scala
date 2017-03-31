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
object Inception_BN {

  def ConvFactory(data: Symbol, numFilter: Int, kernel: (Int, Int), stride: (Int, Int) = (1, 1),
      pad: (Int, Int) = (0, 0), name: String = "", suffix: String = ""): Symbol = {
    val conv = Symbol.Convolution(s"conv_${name}${suffix}")()(
        Map("data" -> data, "num_filter" -> numFilter, "kernel" -> s"$kernel",
            "stride" -> s"$stride", "pad" -> s"$pad"))
    val bn = Symbol.BatchNorm(s"bn_${name}${suffix}")()(Map("data" -> conv))
    val act = Symbol.Activation(s"relu_${name}${suffix}")()(
        Map("data" -> bn, "act_type" -> "relu"))
    act
  }

  def InceptionFactoryA(data: Symbol, num1x1: Int, num3x3red: Int, num3x3: Int,
      numd3x3red: Int, numd3x3: Int, pool: String, proj: Int, name: String): Symbol = {
    // 1x1
    val c1x1 = ConvFactory(data = data, numFilter = num1x1,
        kernel = (1, 1), name = s"${name}_1x1")
    // 3x3 reduce + 3x3
    val c3x3r = ConvFactory(data = data, numFilter = num3x3red,
        kernel = (1, 1), name = s"${name}_3x3", suffix = "_reduce")
    val c3x3 = ConvFactory(data = c3x3r, numFilter = num3x3,
        kernel = (3, 3), pad = (1, 1), name = s"${name}_3x3")
    // double 3x3 reduce + double 3x3
    val cd3x3r = ConvFactory(data = data, numFilter = numd3x3red,
        kernel = (1, 1), name = s"${name}_double_3x3", suffix = "_reduce")
    var cd3x3 = ConvFactory(data = cd3x3r, numFilter = numd3x3,
        kernel = (3, 3), pad = (1, 1), name = s"${name}_double_3x3_0")
    cd3x3 = ConvFactory(data = cd3x3, numFilter = numd3x3,
        kernel = (3, 3), pad = (1, 1), name = s"${name}_double_3x3_1")
    // pool + proj
    val pooling = Symbol.Pooling(s"${pool}_pool_${name}_pool")()(
        Map("data" -> data, "kernel" -> "(3, 3)", "stride" -> "(1, 1)",
            "pad" -> "(1, 1)", "pool_type" -> pool))
    val cproj = ConvFactory(data = pooling, numFilter = proj,
        kernel = (1, 1), name = s"${name}_proj")
    // concat
    val concat = Symbol.Concat(s"ch_concat_${name}_chconcat")(c1x1, c3x3, cd3x3, cproj)()
    concat
  }

  def InceptionFactoryB(data: Symbol, num3x3red : Int, num3x3 : Int,
      numd3x3red : Int, numd3x3 : Int, name: String): Symbol = {
    // 3x3 reduce + 3x3
    val c3x3r = ConvFactory(data = data, numFilter = num3x3red,
        kernel = (1, 1), name = s"${name}_3x3", suffix = "_reduce")
    val c3x3 = ConvFactory(data = c3x3r, numFilter = num3x3,
        kernel = (3, 3), pad = (1, 1), stride = (2, 2), name = s"${name}_3x3")
    // double 3x3 reduce + double 3x3
    val cd3x3r = ConvFactory(data = data, numFilter = numd3x3red,
        kernel = (1, 1), name = s"${name}_double_3x3", suffix = "_reduce")
    var cd3x3 = ConvFactory(data = cd3x3r, numFilter = numd3x3,
        kernel = (3, 3), pad = (1, 1), stride = (1, 1), name = s"${name}_double_3x3_0")
    cd3x3 = ConvFactory(data = cd3x3, numFilter = numd3x3,
        kernel = (3, 3), pad = (1, 1), stride = (2, 2), name = s"${name}_double_3x3_1")
    // pool + proj
    val pooling = Symbol.Pooling(s"max_pool_${name}_pool")()(
        Map("data" -> data, "kernel" -> "(3, 3)", "stride" -> "(2, 2)",
            "pad" -> "(1, 1)", "pool_type" -> "max"))
    // concat
    val concat = Symbol.Concat(s"ch_concat_${name}_chconcat")(c3x3, cd3x3, pooling)()
    concat
  }

  def getSymbol(numClasses: Int = 1000): Symbol = {
    // data
    val data = Symbol.Variable("data")
    // stage 1
    val conv1 = ConvFactory(data = data, numFilter = 64,
        kernel = (7, 7), stride = (2, 2), pad = (3, 3), name = "conv1")
    val pool1 = Symbol.Pooling("pool1")()(Map("data" -> conv1, "kernel" -> "(3, 3)",
        "stride" -> "(2, 2)", "pool_type" -> "max"))
    // stage 2
    val conv2red = ConvFactory(data = pool1, numFilter = 64,
        kernel = (1, 1), stride = (1, 1), name = "conv2red")
    val conv2 = ConvFactory(data = conv2red, numFilter = 192,
        kernel = (3, 3), stride = (1, 1), pad = (1, 1), name = "conv2")
    val pool2 = Symbol.Pooling("pool2")()(Map("data" -> conv2, "kernel" -> "(3, 3)",
        "stride" -> "(2, 2)", "pool_type" -> "max"))
    // stage 2
    val in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, "avg", 32, "3a")
    val in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, "3b")
    val in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, "3c")
    // stage 3
    val in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, "4a")
    val in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, "4b")
    val in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, "4c")
    val in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, "4d")
    val in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, "4e")
    // stage 4
    val in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, "5a")
    val in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, "5b")
    // global avg pooling
    val avg = Symbol.Pooling("global_pool")()(Map("data" -> in5b, "kernel" -> "(7, 7)",
        "stride" -> "(1, 1)", "pool_type" -> "avg"))
    // linear classifier
    val flatten = Symbol.Flatten("flatten")()(Map("data" -> avg))
    val fc1 = Symbol.FullyConnected("fc1")()(
        Map("data" -> flatten, "num_hidden" -> numClasses))
    val softmax = Symbol.SoftmaxOutput("softmax")()(Map("data" -> fc1))
    softmax
  }
}
