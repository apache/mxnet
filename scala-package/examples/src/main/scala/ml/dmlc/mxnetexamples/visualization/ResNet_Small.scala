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
object ResNet_Small {

  sealed trait ConvType
  case object ConvWithoutAct extends ConvType
  case object ConvWitAct extends ConvType

  def convFactory(data: Symbol, numFilter: Int, kernel: (Int, Int),
      stride: (Int, Int), pad: (Int, Int), actType: String = "relu",
      convType: ConvType = ConvWitAct): Symbol = convType match {
    case ConvWitAct => {
      val conv = Symbol.Convolution()()(Map("data" -> data,
          "num_filter" -> numFilter, "kernel" -> s"$kernel",
          "stride" -> s"$stride", "pad" -> s"$pad"))
      val bn = Symbol.BatchNorm()()(Map("data" -> conv))
      val act = Symbol.Activation()()(Map("data" -> bn, "act_type" -> actType))
      act
    }
    case ConvWithoutAct => {
      val conv = Symbol.Convolution()()(Map("data" -> data,
          "num_filter" -> numFilter, "kernel" -> s"$kernel",
          "stride" -> s"$stride", "pad" -> s"$pad"))
      val bn = Symbol.BatchNorm()()(Map("data" -> conv))
      bn
    }
  }

  def residualFactory(data: Symbol, numFilter: Int, dimMatch: Boolean): Symbol = {
    if (dimMatch == true) {
        val identityData = data
        val conv1 = convFactory(data = data, numFilter = numFilter, kernel = (3, 3),
            stride = (1, 1), pad = (1, 1), actType = "relu", convType = ConvWitAct)

        val conv2 = convFactory(data = conv1, numFilter = numFilter, kernel = (3, 3),
            stride = (1, 1), pad = (1, 1), convType = ConvWithoutAct)
        val newData = identityData + conv2
        val act = Symbol.Activation()()(Map("data" -> newData, "act_type" -> "relu"))
        act
    } else {
        val conv1 = convFactory(data = data, numFilter = numFilter, kernel = (3, 3),
            stride = (2, 2), pad = (1, 1), actType = "relu", convType = ConvWitAct)
        val conv2 = convFactory(data = conv1, numFilter = numFilter, kernel = (3, 3),
            stride = (1, 1), pad = (1, 1), convType = ConvWithoutAct)

        // adopt project method in the paper when dimension increased
        val projectData = convFactory(data = data, numFilter = numFilter, kernel = (1, 1),
            stride = (2, 2), pad = (0, 0), convType = ConvWithoutAct)
        val newData = projectData + conv2
        val act = Symbol.Activation()()(Map("data" -> newData, "act_type" -> "relu"))
        act
    }
  }

  def residualNet(data: Symbol, n: Int): Symbol = {
    // fisrt 2n layers
    val data1 = (data /: (0 until n)) { (acc, elem) =>
      residualFactory(data = acc, numFilter = 16, dimMatch = true)
    }

    // second 2n layers
    val data2 = (data1 /: (0 until n)) { (acc, elem) =>
      if (elem == 0) residualFactory(data = acc, numFilter = 32, dimMatch = false)
      else residualFactory(data = acc, numFilter = 32, dimMatch = true)
    }

    // third 2n layers
    val data3 = (data2 /: (0 until n)) { (acc, elem) =>
      if (elem == 0) residualFactory(data = acc, numFilter = 64, dimMatch = false)
      else residualFactory(data = acc, numFilter = 64, dimMatch = true)
    }
     data3
  }

  def getSymbol(numClasses: Int = 1000): Symbol = {
    val conv = convFactory(data = Symbol.Variable("data"), numFilter = 16,
        kernel = (3, 3), stride = (1, 1), pad = (1, 1), actType = "relu", convType = ConvWitAct)
    // set n = 3 means get a model with 3*6+2=20 layers, set n = 9 means 9*6+2=56 layers
    val n = 3
    val resNet = residualNet(conv, n)
    val pool = Symbol.Pooling()()(Map("data" -> resNet,
        "kernel" -> "(7,7)", "pool_type" -> "avg"))
    val flatten = Symbol.Flatten("flatten")()(Map("data" -> pool))
    val fc = Symbol.FullyConnected("fc1")()(Map("data" -> flatten, "num_hidden" -> numClasses))
    val softmax = Symbol.SoftmaxOutput("softmax")()(Map("data" -> fc))
    softmax
  }
}
