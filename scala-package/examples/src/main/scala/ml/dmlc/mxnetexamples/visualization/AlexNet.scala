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
object AlexNet {

  def getSymbol(numClasses: Int = 1000): Symbol = {
    val inputData = Symbol.Variable("data")
    // stage 1
    val conv1 = Symbol.Convolution()()(Map(
        "data" -> inputData, "kernel" -> "(11, 11)", "stride" -> "(4, 4)", "num_filter" -> 96))
    val relu1 = Symbol.Activation()()(Map("data" -> conv1, "act_type" -> "relu"))
    val pool1 = Symbol.Pooling()()(Map(
        "data" -> relu1, "pool_type" -> "max", "kernel" -> "(3, 3)", "stride" -> "(2,2)"))
    val lrn1 = Symbol.LRN()()(Map("data" -> pool1,
        "alpha" -> 0.0001f, "beta" -> 0.75f, "knorm" -> 1f, "nsize" -> 5))
    // stage 2
    val conv2 = Symbol.Convolution()()(Map(
        "data" -> lrn1, "kernel" -> "(5, 5)", "pad" -> "(2, 2)", "num_filter" -> 256))
    val relu2 = Symbol.Activation()()(Map("data" -> conv2, "act_type" -> "relu"))
    val pool2 = Symbol.Pooling()()(Map("data" -> relu2,
        "kernel" -> "(3, 3)", "stride" -> "(2, 2)", "pool_type" -> "max"))
    val lrn2 = Symbol.LRN()()(Map("data" -> pool2,
        "alpha" -> 0.0001f, "beta" -> 0.75f, "knorm" -> 1f, "nsize" -> 5))
    // stage 3
    val conv3 = Symbol.Convolution()()(Map(
        "data" -> lrn2, "kernel" -> "(3, 3)", "pad" -> "(1, 1)", "num_filter" -> 384))
    val relu3 = Symbol.Activation()()(Map("data" -> conv3, "act_type" -> "relu"))
    val conv4 = Symbol.Convolution()()(Map(
        "data" -> relu3, "kernel" -> "(3, 3)", "pad" -> "(1, 1)", "num_filter" -> 384))
    val relu4 = Symbol.Activation()()(Map("data" -> conv4, "act_type" -> "relu"))
    val conv5 = Symbol.Convolution()()(Map(
        "data" -> relu4, "kernel" -> "(3, 3)", "pad" -> "(1, 1)", "num_filter" -> 256))
    val relu5 = Symbol.Activation()()(Map("data" -> conv5, "act_type" -> "relu"))
    val pool3 = Symbol.Pooling()()(Map("data" -> relu5,
        "kernel" -> "(3, 3)", "stride" -> "(2, 2)", "pool_type" -> "max"))
    // stage 4
    val flatten = Symbol.Flatten()()(Map("data" -> pool3))
    val fc1 = Symbol.FullyConnected()()(Map("data" -> flatten, "num_hidden" -> 4096))
    val relu6 = Symbol.Activation()()(Map("data" -> fc1, "act_type" -> "relu"))
    val dropout1 = Symbol.Dropout()()(Map("data" -> relu6, "p" -> 0.5f))
    // stage 5
    val fc2 = Symbol.FullyConnected()()(Map("data" -> dropout1, "num_hidden" -> 4096))
    val relu7 = Symbol.Activation()()(Map("data" -> fc2, "act_type" -> "relu"))
    val dropout2 = Symbol.Dropout()()(Map("data" -> relu7, "p" -> 0.5f))
    // stage 6
    val fc3 = Symbol.FullyConnected()()(
        Map("data" -> dropout2, "num_hidden" -> numClasses))
    val softmax = Symbol.SoftmaxOutput("softmax")()(Map("data" -> fc3))
    softmax
  }
}
