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
object VGG {

  def getSymbol(numClasses: Int = 1000): Symbol = {
    // define alexnet
    val data = Symbol.Variable("data")
    // group 1
    val conv1_1 = Symbol.Convolution("conv1_1")()(
        Map("data" -> data, "num_filter" -> 64, "pad" -> "(1,1)", "kernel" -> "(3,3)"))
    val relu1_1 = Symbol.Activation("relu1_1")()(Map("data" -> conv1_1, "act_type" -> "relu"))
    val pool1 = Symbol.Pooling("pool1")()(
        Map("data" -> relu1_1, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2,2)"))
    // group 2
    val conv2_1 = Symbol.Convolution("conv2_1")()(
        Map("data" -> pool1, "num_filter" -> 128, "pad" -> "(1,1)", "kernel" -> "(3,3)"))
    val relu2_1 = Symbol.Activation("relu2_1")()(Map("data" -> conv2_1, "act_type" -> "relu"))
    val pool2 = Symbol.Pooling("pool2")()(
        Map("data" -> relu2_1, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2,2)"))
    // group 3
    val conv3_1 = Symbol.Convolution("conv3_1")()(
        Map("data" -> pool2, "num_filter" -> 256, "pad" -> "(1,1)", "kernel" -> "(3,3)"))
    val relu3_1 = Symbol.Activation("relu3_1")()(Map("data" -> conv3_1, "act_type" -> "relu"))
    val conv3_2 = Symbol.Convolution("conv3_2")()(
        Map("data" -> relu3_1, "num_filter" -> 256, "pad" -> "(1,1)", "kernel" -> "(3,3)"))
    val relu3_2 = Symbol.Activation("relu3_2")()(Map("data" -> conv3_2 , "act_type" -> "relu"))
    val pool3 = Symbol.Pooling("pool3")()(
        Map("data" -> relu3_2, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2,2)"))
    // group 4
    val conv4_1 = Symbol.Convolution("conv4_1")()(
        Map("data" -> pool3, "num_filter" -> 512, "pad" -> "(1,1)", "kernel" -> "(3,3)"))
    val relu4_1 = Symbol.Activation("relu4_1")()(Map("data" -> conv4_1 , "act_type" -> "relu"))
    val conv4_2 = Symbol.Convolution("conv4_2")()(
        Map("data" -> relu4_1, "num_filter" -> 512, "pad" -> "(1,1)", "kernel" -> "(3,3)"))
    val relu4_2 = Symbol.Activation("relu4_2")()(Map("data" -> conv4_2 , "act_type" -> "relu"))
    val pool4 = Symbol.Pooling("pool4")()(
        Map("data" -> relu4_2, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2,2)"))
    // group 5
    val conv5_1 = Symbol.Convolution("conv5_1")()(
        Map("data" -> pool4, "num_filter" -> 512, "pad" -> "(1,1)", "kernel" -> "(3,3)"))
    val relu5_1 = Symbol.Activation("relu5_1")()(Map("data" -> conv5_1, "act_type" -> "relu"))
    val conv5_2 = Symbol.Convolution("conv5_2")()(
        Map("data" -> relu5_1, "num_filter" -> 512, "pad" -> "(1,1)", "kernel" -> "(3,3)"))
    val relu5_2 = Symbol.Activation("relu5_2")()(Map("data" -> conv5_2, "act_type" -> "relu"))
    val pool5 = Symbol.Pooling("pool5")()(
        Map("data" -> relu5_2, "pool_type" -> "max", "kernel" -> "(2, 2)", "stride" -> "(2,2)"))
    // group 6
    val flatten = Symbol.Flatten("flatten")()(Map("data" -> pool5))
    val fc6 = Symbol.FullyConnected("fc6")()(Map("data" -> flatten, "num_hidden" -> 4096))
    val relu6 = Symbol.Activation("relu6")()(Map("data" -> fc6, "act_type" -> "relu"))
    val drop6 = Symbol.Dropout("drop6")()(Map("data" -> relu6, "p" -> 0.5f))
    // group 7
    val fc7 = Symbol.FullyConnected("fc7")()(Map("data" -> drop6, "num_hidden" -> 4096))
    val relu7 = Symbol.Activation("relu7")()(Map("data" -> fc7, "act_type" -> "relu"))
    val drop7 = Symbol.Dropout("drop7")()(Map("data" -> relu7, "p" -> 0.5f))
    // output
    val fc8 = Symbol.FullyConnected("fc8")()(
        Map("data" -> drop7, "num_hidden" -> numClasses))
    val softmax = Symbol.SoftmaxOutput("softmax")()(Map("data" -> fc8))
    softmax
  }
}
