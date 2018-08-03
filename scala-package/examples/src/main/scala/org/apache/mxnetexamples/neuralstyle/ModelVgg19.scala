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

package org.apache.mxnetexamples.neuralstyle

import org.apache.mxnet.{Context, Executor, NDArray, Shape, Symbol}

/**
  * Definition for the neuralstyle network and initialize it with pretrained weight
  */
object ModelVgg19 {
  case class ConvExecutor(executor: Executor, data: NDArray, dataGrad: NDArray,
                          style: Array[NDArray], content: NDArray, argDict: Map[String, NDArray])

  def ConvRelu(data : Symbol, convName : String, reluName : String,
               numFilter : Int, kernel : (Int, Int) = (3, 3),
               stride : (Int, Int) = (1, 1)) : Symbol = {
    val conv = Symbol.api.Convolution(data = Some(data), num_filter = numFilter,
      pad = Some(Shape(1, 1)), kernel = Shape(kernel._1, kernel._2),
      stride = Some(Shape(stride._1, stride._2)), no_bias = Some(false),
      workspace = Some(1024), name = convName)
    val relu = Symbol.api.relu(data = Some(conv), name = reluName)
    conv.dispose()
    relu
  }

  def getSymbol: (Symbol, Symbol) = {
    getVggSymbol()
  }

  def getVggSymbol(prefix: String = "", contentOnly: Boolean = false): (Symbol, Symbol) = {
    // declare symbol
    val data = Symbol.Variable(s"${prefix}data")

    val relu1_1 = ConvRelu(data, s"${prefix}conv1_1", s"${prefix}relu1_1", 64)
    val relu1_2 = ConvRelu(relu1_1, s"${prefix}conv1_2", s"${prefix}relu1_2", 64)
    val pool1 = Symbol.api.Pooling(data = Some(relu1_2), pad = Some(Shape(0, 0)),
      kernel = Some(Shape(2, 2)), stride = Some(Shape(2, 2)), pool_type = Some("avg"),
      name = s"${prefix}pool1")

    val relu2_1 = ConvRelu(pool1, s"${prefix}conv2_1", s"${prefix}relu2_1", 128)
    val relu2_2 = ConvRelu(relu2_1, s"${prefix}conv2_2", s"${prefix}relu2_2", 128)
    val pool2 = Symbol.api.Pooling(data = Some(relu2_2), pad = Some(Shape(0, 0)),
      kernel = Some(Shape(2, 2)), stride = Some(Shape(2, 2)), pool_type = Some("avg"),
      name = s"${prefix}pool2")

    val relu3_1 = ConvRelu(pool2, s"${prefix}conv3_1", s"${prefix}relu3_1", 256)
    val relu3_2 = ConvRelu(relu3_1, s"${prefix}conv3_2", s"${prefix}relu3_2", 256)
    val relu3_3 = ConvRelu(relu3_2, s"${prefix}conv3_3", s"${prefix}relu3_3", 256)
    val relu3_4 = ConvRelu(relu3_3, s"${prefix}conv3_4", s"${prefix}relu3_4", 256)
    val pool3 = Symbol.api.Pooling(data = Some(relu3_4), pad = Some(Shape(0, 0)),
      kernel = Some(Shape(2, 2)), stride = Some(Shape(2, 2)), pool_type = Some("avg"),
      name = s"${prefix}pool3")

    val relu4_1 = ConvRelu(pool3, s"${prefix}conv4_1", s"${prefix}relu4_1", 512)
    val relu4_2 = ConvRelu(relu4_1, s"${prefix}conv4_2", s"${prefix}relu4_2", 512)
    val relu4_3 = ConvRelu(relu4_2, s"${prefix}conv4_3", s"${prefix}relu4_3", 512)
    val relu4_4 = ConvRelu(relu4_3, s"${prefix}conv4_4", s"${prefix}relu4_4", 512)
    val pool4 = Symbol.api.Pooling(data = Some(relu4_4), pad = Some(Shape(0, 0)),
      kernel = Some(Shape(2, 2)), stride = Some(Shape(2, 2)), pool_type = Some("avg"),
      name = s"${prefix}pool4")

    val relu5_1 = ConvRelu(pool4, s"${prefix}conv5_1", s"${prefix}relu5_1", 512)

    // style and content layers
    val style = if (contentOnly) null else Symbol.Group(relu1_1, relu2_1, relu3_1, relu4_1, relu5_1)
    val content = Symbol.Group(relu4_2)
    (style, content)
  }

  def getExecutor(style: Symbol, content: Symbol, modelPath: String,
                  inputSize: (Int, Int), ctx: Context): ConvExecutor = {
    val out = Symbol.Group(style, content)
    // make executor
    val (argShapes, outputShapes, auxShapes) = out.inferShape(
      Map("data" -> Shape(1, 3, inputSize._1, inputSize._2)))
    val argNames = out.listArguments()
    val argDict = argNames.zip(argShapes.map(NDArray.zeros(_, ctx))).toMap
    val gradDict = Map("data" -> argDict("data").copyTo(ctx))
    // init with pretrained weight
    val pretrained = NDArray.load2Map(modelPath)
    argNames.filter(_ != "data").foreach { name =>
      val key = s"arg:$name"
      if (pretrained.contains(key)) argDict(name).set(pretrained(key))
    }
    pretrained.foreach(ele => ele._2.dispose())
    val executor = out.bind(ctx, argDict, gradDict)
    out.dispose()
    val outArray = executor.outputs
    ConvExecutor(executor = executor,
      data = argDict("data"),
      dataGrad = gradDict("data"),
      style = outArray.take(outArray.length - 1),
      content = outArray(outArray.length - 1),
      argDict = argDict)
  }

  def getModel(modelPath: String, inputSize: (Int, Int), ctx: Context): ConvExecutor = {
    val (style, content) = getSymbol
    getExecutor(style, content, modelPath, inputSize, ctx)
  }
}
