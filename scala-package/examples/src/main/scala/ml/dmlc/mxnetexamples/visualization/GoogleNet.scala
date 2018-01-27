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
object GoogleNet {

  def ConvFactory(data: Symbol, numFilter: Int, kernel: (Int, Int), stride: (Int, Int) = (1, 1),
      pad: (Int, Int) = (0, 0), name: String = "", suffix: String = ""): Symbol = {
    val conv = Symbol.Convolution(s"conv_${name}${suffix}")()(
      Map("data" -> data, "num_filter" -> numFilter, "kernel" -> s"$kernel",
        "stride" -> s"$stride", "pad" -> s"$pad"))
    val act = Symbol.Activation(s"relu_${name}${suffix}")()(
        Map("data" -> conv, "act_type" -> "relu"))
    act
  }

  def InceptionFactory(data: Symbol, num1x1: Int, num3x3red: Int, num3x3: Int,
      numd5x5red: Int, numd5x5: Int, pool: String, proj: Int, name: String): Symbol = {
      // 1x1
      val c1x1 = ConvFactory(data = data, numFilter = num1x1,
          kernel = (1, 1), name = s"${name}_1x1")
      // 3x3 reduce + 3x3
      val c3x3r = ConvFactory(data = data, numFilter = num3x3red,
          kernel = (1, 1), name = s"${name}_3x3", suffix = "_reduce")
      val c3x3 = ConvFactory(data = c3x3r, numFilter = num3x3,
          kernel = (3, 3), pad = (1, 1), name = s"${name}_3x3")
      // double 3x3 reduce + double 3x3
      val cd5x5r = ConvFactory(data = data, numFilter = numd5x5red,
          kernel = (1, 1), name = s"${name}_5x5", suffix = "_reduce")
      val cd5x5 = ConvFactory(data = cd5x5r, numFilter = numd5x5,
          kernel = (5, 5), pad = (2, 2), name = s"${name}_5x5")
      // pool + proj
      val pooling = Symbol.Pooling(s"${pool}_pool_${name}_pool")()(Map("data" -> data,
          "kernel" -> "(3, 3)", "stride" -> "(1, 1)", "pad" -> "(1, 1)", "pool_type" -> pool))
      val cproj = ConvFactory(data = pooling, numFilter = proj,
          kernel = (1, 1), name = s"${name}_proj")
      // concat
      val concat =
        Symbol.Concat(s"ch_concat_${name}_chconcat")(c1x1, c3x3, cd5x5, cproj)()
      concat
  }

  def getSymbol(numClasses: Int = 1000): Symbol = {
    val data = Symbol.Variable("data")
    val conv1 = ConvFactory(data, 64, kernel = (7, 7),
        stride = (2, 2), pad = (3, 3), name = "conv1")
    val pool1 = Symbol.Pooling()()(Map("data" -> conv1, "kernel" -> "(3, 3)",
        "stride" -> "(2, 2)", "pool_type" -> "max"))
    val conv2 = ConvFactory(pool1, 64, kernel = (1, 1), stride = (1, 1), name = "conv2")
    val conv3 = ConvFactory(conv2, 192, kernel = (3, 3),
        stride = (1, 1), pad = (1, 1), name = "conv3")
    val pool3 = Symbol.Pooling()()(Map("data" -> conv3,
        "kernel" -> "(3, 3)", "stride" -> "(2, 2)", "pool_type" -> "max"))
    val in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32, name = "in3a")
    val in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name = "in3b")
    val pool4 = Symbol.Pooling()()(Map("data" -> in3b, "kernel" -> "(3, 3)",
        "stride" -> "(2, 2)", "pool_type" -> "max"))
    val in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64, name = "in4a")
    val in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name = "in4b")
    val in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name = "in4c")
    val in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name = "in4d")
    val in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name = "in4e")
    val pool5 = Symbol.Pooling()()(Map("data" -> in4e, "kernel" -> "(3, 3)",
        "stride" -> "(2, 2)", "pool_type" -> "max"))
    val in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128, name = "in5a")
    val in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name = "in5b")
    val pool6 = Symbol.Pooling()()(Map("data" -> in5b, "kernel" -> "(7, 7)",
        "stride" -> "(1,1)", "pool_type" -> "avg"))
    val flatten = Symbol.Flatten()()(Map("data" -> pool6))
    val fc1 = Symbol.FullyConnected()()(Map("data" -> flatten, "num_hidden" -> numClasses))
    val softmax = Symbol.SoftmaxOutput("softmax")()(Map("data" -> fc1))
    softmax
  }
}
