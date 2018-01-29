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

import ml.dmlc.mxnet.Executor
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Symbol


object ModelVgg19 {

  def getVggSymbol(prefix: String, contentOnly: Boolean = false): (Symbol, Symbol) = {
    // declare symbol
    val data = Symbol.Variable(s"${prefix}_data")
    val conv1_1 = Symbol.Convolution(s"${prefix}_conv1_1")()(Map("data" -> data,
                            "num_filter" -> 64, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu1_1 = Symbol.Activation(s"${prefix}_relu1_1")()(Map("data" -> conv1_1,
                            "act_type" -> "relu"))
    val conv1_2 = Symbol.Convolution(s"${prefix}_conv1_2")()(Map("data" -> relu1_1,
                            "num_filter" -> 64, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu1_2 = Symbol.Activation(s"${prefix}_relu1_2")()(Map("data" -> conv1_2,
                            "act_type" -> "relu"))
    val pool1 = Symbol.Pooling(s"${prefix}_pool1")()(Map("data" -> relu1_2 , "pad" -> "(0,0)",
                            "kernel" -> "(2,2)", "stride" -> "(2,2)", "pool_type" -> "avg"))
    val conv2_1 = Symbol.Convolution(s"${prefix}_conv2_1")()(Map("data" -> pool1,
                            "num_filter" -> 128, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu2_1 = Symbol.Activation(s"${prefix}_relu2_1")()(Map("data" -> conv2_1,
                            "act_type" -> "relu"))
    val conv2_2 = Symbol.Convolution(s"${prefix}_conv2_2")()(Map("data" -> relu2_1,
                            "num_filter" -> 128, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu2_2 = Symbol.Activation(s"${prefix}_relu2_2")()(Map("data" -> conv2_2,
                            "act_type" -> "relu"))
    val pool2 = Symbol.Pooling("pool2")()(Map("data" -> relu2_2 , "pad" -> "(0,0)",
                            "kernel" -> "(2,2)", "stride" -> "(2,2)", "pool_type" -> "avg"))
    val conv3_1 = Symbol.Convolution(s"${prefix}_conv3_1")()(Map("data" -> pool2,
                            "num_filter" -> 256, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu3_1 = Symbol.Activation(s"${prefix}_relu3_1")()(Map("data" -> conv3_1,
                            "act_type" -> "relu"))
    val conv3_2 = Symbol.Convolution(s"${prefix}_conv3_2")()(Map("data" -> relu3_1,
                            "num_filter" -> 256, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu3_2 = Symbol.Activation(s"${prefix}_relu3_2")()(Map("data" -> conv3_2,
                            "act_type" -> "relu"))
    val conv3_3 = Symbol.Convolution(s"${prefix}_conv3_3")()(Map("data" -> relu3_2,
                            "num_filter" -> 256, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu3_3 = Symbol.Activation(s"${prefix}_relu3_3")()(Map("data" -> conv3_3,
                            "act_type" -> "relu"))
    val conv3_4 = Symbol.Convolution(s"${prefix}_conv3_4")()(Map("data" -> relu3_3,
                            "num_filter" -> 256, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu3_4 = Symbol.Activation(s"${prefix}_relu3_4")()(Map("data" -> conv3_4 ,
                            "act_type" -> "relu"))
    val pool3 = Symbol.Pooling(s"${prefix}_pool3")()(Map("data" -> relu3_4,
                            "pad" -> "(0,0)", "kernel" -> "(2,2)", "stride" -> "(2,2)",
                            "pool_type" -> "avg"))
    val conv4_1 = Symbol.Convolution(s"${prefix}_conv4_1")()(Map("data" -> pool3,
                            "num_filter" -> 512, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu4_1 = Symbol.Activation(s"${prefix}_relu4_1")()(Map("data" -> conv4_1,
                            "act_type" -> "relu"))
    val conv4_2 = Symbol.Convolution(s"${prefix}_conv4_2")()(Map("data" -> relu4_1,
                            "num_filter" -> 512, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu4_2 = Symbol.Activation(s"${prefix}_relu4_2")()(Map("data" -> conv4_2,
                            "act_type" -> "relu"))
    val conv4_3 = Symbol.Convolution(s"${prefix}_conv4_3")()(Map("data" -> relu4_2,
                            "num_filter" -> 512, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu4_3 = Symbol.Activation(s"${prefix}_relu4_3")()(Map("data" -> conv4_3,
                            "act_type" -> "relu"))
    val conv4_4 = Symbol.Convolution(s"${prefix}_conv4_4")()(Map("data" -> relu4_3,
                            "num_filter" -> 512, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu4_4 = Symbol.Activation(s"${prefix}_relu4_4")()(Map("data" -> conv4_4,
                            "act_type" -> "relu"))
    val pool4 = Symbol.Pooling(s"${prefix}_pool4")()(Map("data" -> relu4_4,
                            "pad" -> "(0,0)", "kernel" -> "(2,2)", "stride" -> "(2,2)",
                            "pool_type" -> "avg"))
    val conv5_1 = Symbol.Convolution(s"${prefix}_conv5_1")()(Map("data" -> pool4,
                            "num_filter" -> 512, "pad" -> "(1,1)", "kernel" -> "(3,3)",
                            "stride" -> "(1,1)", "no_bias" -> false, "workspace" -> 1024))
    val relu5_1 = Symbol.Activation(s"${prefix}_relu5_1")()(Map("data" -> conv5_1,
                            "act_type" -> "relu"))

    // style and content layers
    val style = if (contentOnly) null else Symbol.Group(relu1_1, relu2_1, relu3_1, relu4_1, relu5_1)
    val content = Symbol.Group(relu4_2)
    (style, content)
  }
}
