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

package org.apache.mxnetexamples.imclassification.models

import org.apache.mxnet._

object Resnet {
  /**
    * Helper to produce individual residual unit
    */
  def residualUnit(data: Symbol, numFilter: Int, stride: Shape, dimMatch: Boolean,
                   name: String = "", bottleNeck: Boolean = true, bnMom: Float = 0.9f,
                   workspace: Int = 256, memonger: Boolean = false): Symbol = {
    val (act1, operated) = if (bottleNeck) {
      val bn1 = Symbol.api.BatchNorm(data = Some(data), fix_gamma = Some(false), eps = Some(2e-5),
        momentum = Some(bnMom), name = name + "_bn1")
      val act1: Symbol = Symbol.api.Activation(data = Some(bn1), act_type = "relu",
        name = name + "_relu1")
      val conv1 = Symbol.api.Convolution(data = Some(act1), num_filter = (numFilter * 0.25).toInt,
        kernel = Shape(1, 1), stride = Some(Shape(1, 1)), pad = Some(Shape(0, 0)),
        no_bias = Some(true), workspace = Some(workspace), name = name + "_conv1")
      val bn2 = Symbol.api.BatchNorm(data = Some(conv1), fix_gamma = Some(false),
        eps = Some(2e-5), momentum = Some(bnMom), name = name + "_bn2")
      val act2 = Symbol.api.Activation(data = Some(bn2), act_type = "relu", name = name + "_relu2")
      val conv2 = Symbol.api.Convolution(data = Some(act2), num_filter = (numFilter * 0.25).toInt,
        kernel = Shape(3, 3), stride = Some(stride), pad = Some(Shape(1, 1)),
        no_bias = Some(true), workspace = Some(workspace), name = name + "_conv2")
      val bn3 = Symbol.api.BatchNorm(data = Some(conv2), fix_gamma = Some(false),
        eps = Some(2e-5), momentum = Some(bnMom), name = name + "_bn3")
      val act3 = Symbol.api.Activation(data = Some(bn3), act_type = "relu", name = name + "_relu3")
      val conv3 = Symbol.api.Convolution(data = Some(act3), num_filter = numFilter,
        kernel = Shape(1, 1), stride = Some(Shape(1, 1)), pad = Some(Shape(0, 0)),
        no_bias = Some(true), workspace = Some(workspace), name = name + "_conv3")
      (act1, conv3)
    } else {
      val bn1 = Symbol.api.BatchNorm(data = Some(data), fix_gamma = Some(false),
        eps = Some(2e-5), momentum = Some(bnMom), name = name + "_bn1")
      val act1 = Symbol.api.Activation(data = Some(bn1), act_type = "relu", name = name + "_relu1")
      val conv1 = Symbol.api.Convolution(data = Some(act1), num_filter = numFilter,
        kernel = Shape(3, 3), stride = Some(stride), pad = Some(Shape(1, 1)),
        no_bias = Some(true), workspace = Some(workspace), name = name + "_conv1")
      val bn2 = Symbol.api.BatchNorm(data = Some(conv1), fix_gamma = Some(false),
        eps = Some(2e-5), momentum = Some(bnMom), name = name + "_bn2")
      val act2 = Symbol.api.Activation(data = Some(bn2), act_type = "relu", name = name + "_relu2")
      val conv2 = Symbol.api.Convolution(data = Some(act2), num_filter = numFilter,
        kernel = Shape(3, 3), stride = Some(Shape(1, 1)), pad = Some(Shape(1, 1)),
        no_bias = Some(true), workspace = Some(workspace), name = name + "_conv2")
      (act1, conv2)
    }
    val shortcut = if (dimMatch) {
      data
    } else {
      Symbol.api.Convolution(Some(act1), num_filter = numFilter, kernel = Shape(1, 1),
        stride = Some(stride), no_bias = Some(true), workspace = Some(workspace),
        name = name + "_sc")
    }
    operated + shortcut
  }

  /**
    * Helper for building the resnet Symbol
    */
  def resnet(units: List[Int], numStages: Int, filterList: List[Int], numClasses: Int,
             imageShape: List[Int], bottleNeck: Boolean = true, bnMom: Float = 0.9f,
             workspace: Int = 256, dtype: String = "float32", memonger: Boolean = false): Symbol = {
    assert(units.size == numStages)
    var data = Symbol.Variable("data", shape = Shape(List(4) ::: imageShape), dType = DType.Float32)
    if (dtype == "float32") {
      data = Symbol.api.identity(Some(data), "id")
    } else if (dtype == "float16") {
      data = Symbol.api.cast(Some(data), "float16")
    }
    data = Symbol.api.BatchNorm(Some(data), fix_gamma = Some(true), eps = Some(2e-5),
      momentum = Some(bnMom), name = "bn_data")
    val List(channels, height, width) = imageShape
    var body = if (height <= 32) {
      Symbol.api.Convolution(Some(data), num_filter = filterList.head, kernel = Shape(7, 7),
        stride = Some(Shape(1, 1)), pad = Some(Shape(1, 1)), no_bias = Some(true), name = "conv0",
        workspace = Some(workspace))
    } else {
      var body0 = Symbol.api.Convolution(Some(data), num_filter = filterList.head,
        kernel = Shape(3, 3), stride = Some(Shape(2, 2)), pad = Some(Shape(3, 3)),
        no_bias = Some(true), name = "conv0", workspace = Some(workspace))
      body0 = Symbol.api.BatchNorm(Some(body0), fix_gamma = Some(false), eps = Some(2e-5),
        momentum = Some(bnMom), name = "bn0")
      body0 = Symbol.api.Activation(Some(body0), act_type = "relu", name = "relu0")
      Symbol.api.Pooling(Some(body0), kernel = Some(Shape(3, 3)), stride = Some(Shape(2, 2)),
        pad = Some(Shape(1, 1)), pool_type = Some("max"))
    }
    for (((filter, i), unit) <- filterList.tail.zipWithIndex.zip(units)) {
      val stride = Shape(if (i == 0) 1 else 2, if (i == 0) 1 else 2)
      body = residualUnit(body, filter, stride, false, name = s"stage${i + 1}_unit${1}",
        bottleNeck = bottleNeck, workspace = workspace, memonger = memonger)
      for (j <- 0 until unit - 1) {
        body = residualUnit(body, filter, Shape(1, 1), true, s"stage${i + 1}_unit${j + 2}",
          bottleNeck, workspace = workspace, memonger = memonger)
      }
    }
    val bn1 = Symbol.api.BatchNorm(Some(body), fix_gamma = Some(false), eps = Some(2e-5),
      momentum = Some(bnMom), name = "bn1")
    val relu1 = Symbol.api.Activation(Some(bn1), act_type = "relu", name = "relu1")
    val pool1 = Symbol.api.Pooling(Some(relu1), global_pool = Some(true),
      kernel = Some(Shape(7, 7)), pool_type = Some("avg"), name = "pool1")
    val flat = Symbol.api.Flatten(Some(pool1))
    var fc1 = Symbol.api.FullyConnected(Some(flat), num_hidden = numClasses, name = "fc1")
    if (dtype == "float16") {
      fc1 = Symbol.api.cast(Some(fc1), "float32")
    }
    Symbol.api.SoftmaxOutput(Some(fc1), name = "softmax")
  }

  /**
    * Gets the resnet model symbol
    * @param numClasses Number of classes to classify into
    * @param numLayers Number of residual layers
    * @param imageShape The image shape as List(channels, height, width)
    * @param convWorkspace Maximum temporary workspace allowed (MB) in convolutions
    * @param dtype Type of data (float16, float32, etc) to use during computation
    * @return Model symbol
    */
  def getSymbol(numClasses: Int, numLayers: Int, imageShape: List[Int], convWorkspace: Int = 256,
                dtype: String = "float32"): Symbol = {
    val List(channels, height, width) = imageShape
    val (numStages, units, filterList, bottleNeck): (Int, List[Int], List[Int], Boolean) =
      if (height <= 28) {
        val (perUnit, filterList, bottleNeck) = if ((numLayers - 2) % 9 == 0 && numLayers > 165) {
          (List(Math.floor((numLayers - 2) / 9).toInt),
            List(16, 64, 128, 256),
            true)
        } else if ((numLayers - 2) % 6 == 0 && numLayers < 164) {
          (List(Math.floor((numLayers - 2) / 6).toInt),
            List(16, 16, 32, 64),
            false)
        } else {
          throw new Exception(s"Invalid number of layers: ${numLayers}")
        }
        val numStages = 3
        val units = (1 to numStages).map(_ => perUnit.head).toList
        (numStages, units, filterList, bottleNeck)
      } else {
        val (filterList, bottleNeck) = if (numLayers >= 50) {
          (List(64, 256, 512, 1024, 2048), true)
        } else {
          (List(64, 64, 128, 256, 512), false)
        }
        val units: List[Int] = Map(
          18 -> List(2, 2, 2, 2),
          34 -> List(3, 4, 6, 3),
          50 -> List(3, 4, 6, 3),
          101 -> List(3, 4, 23, 3),
          152 -> List(3, 8, 36, 3),
          200 -> List(3, 24, 36, 3),
          269 -> List(3, 30, 48, 8)
        ).get(numLayers) match {
          case Some(x) => x
          case None => throw new Exception(s"Invalid number of layers: ${numLayers}")
        }
        (4, units, filterList, bottleNeck)
      }
    resnet(units, numStages, filterList, numClasses, imageShape, bottleNeck,
      workspace = convWorkspace, dtype = dtype)
  }
}
