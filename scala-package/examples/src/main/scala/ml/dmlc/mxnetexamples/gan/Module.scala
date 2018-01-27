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

package ml.dmlc.mxnetexamples.gan

import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Optimizer
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Initializer
import ml.dmlc.mxnet.DataBatch
import ml.dmlc.mxnet.Random

/**
 * @author Depeng Liang
 */
class GANModule(
              symbolGenerator: Symbol,
              symbolEncoder: Symbol,
              context: Context,
              dataShape: Shape,
              codeShape: Shape,
              posLabel: Float = 0.9f) {

  // generator
  private val gDataLabelShape = Map("rand" -> codeShape)
  private val (gArgShapes, gOutShapes, gAuxShapes) = symbolGenerator.inferShape(gDataLabelShape)

  private val gArgNames = symbolGenerator.listArguments()
  private val gArgDict = gArgNames.zip(gArgShapes.map(NDArray.empty(_, context))).toMap

  private val gGradDict = gArgNames.zip(gArgShapes).filter { case (name, shape) =>
    !gDataLabelShape.contains(name)
  }.map(x => x._1 -> NDArray.empty(x._2, context) ).toMap

  private val gData = gArgDict("rand")

  val gAuxNames = symbolGenerator.listAuxiliaryStates()
  val gAuxDict = gAuxNames.zip(gAuxShapes.map(NDArray.empty(_, context))).toMap
  private val gExecutor =
    symbolGenerator.bind(context, gArgDict, gGradDict, "write", gAuxDict, null, null)

  // discriminator
  private val batchSize = dataShape(0)

  private val dDataShape = Map("data" -> dataShape)
  private val dLabelShape = Map("dloss_label" -> Shape(batchSize))
  private val (dArgShapes, _, dAuxShapes) = symbolEncoder.inferShape(dDataShape ++ dLabelShape)

  private val dArgNames = symbolEncoder.listArguments()
  private val dArgDict = dArgNames.zip(dArgShapes.map(NDArray.empty(_, context))).toMap

  private val dGradDict = dArgNames.zip(dArgShapes).filter { case (name, shape) =>
    !dLabelShape.contains(name)
  }.map(x => x._1 -> NDArray.empty(x._2, context) ).toMap

  private val tempGradD = dArgNames.zip(dArgShapes).filter { case (name, shape) =>
    !dLabelShape.contains(name)
  }.map(x => x._1 -> NDArray.empty(x._2, context) ).toMap

  private val dData = dArgDict("data")
  val dLabel = dArgDict("dloss_label")

  val dAuxNames = symbolEncoder.listAuxiliaryStates()
  val dAuxDict = dAuxNames.zip(dAuxShapes.map(NDArray.empty(_, context))).toMap
  private val dExecutor =
    symbolEncoder.bind(context, dArgDict, dGradDict, "write", dAuxDict, null, null)

  val tempOutG = gOutShapes.map(NDArray.empty(_, context)).toArray
  val tempDiffD: NDArray = dGradDict("data")

  var outputsFake: Array[NDArray] = null
  var outputsReal: Array[NDArray] = null

  def initGParams(initializer: Initializer): Unit = {
    gArgDict.filter(x => !gDataLabelShape.contains(x._1))
                   .foreach { case (name, ndArray) => initializer(name, ndArray) }
  }

  def initDParams(initializer: Initializer): Unit = {
    dArgDict.filter(x => !dDataShape.contains(x._1) && !dLabelShape.contains(x._1))
                   .foreach { case (name, ndArray) => initializer(name, ndArray) }
  }

  private var gOpt: Optimizer = null
  private var gParamsGrads: List[(Int, String, NDArray, AnyRef)] = null
  private var dOpt: Optimizer = null
  private var dParamsGrads: List[(Int, String, NDArray, AnyRef)] = null

  def initOptimizer(opt: Optimizer): Unit = {
    gOpt = opt
    gParamsGrads = gGradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
      (idx, name, grad, gOpt.createState(idx, gArgDict(name)))
    }
    dOpt = opt
    dParamsGrads =
      dGradDict.filter(x => !dDataShape.contains(x._1))
      .toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, dOpt.createState(idx, dArgDict(name)))
    }
  }

  private def saveTempGradD(): Unit = {
    val keys = this.dGradDict.keys
    for (k <- keys) {
      this.dGradDict(k).copyTo(this.tempGradD(k))
    }
  }

  // add back saved gradient
  private def addTempGradD(): Unit = {
    val keys = this.dGradDict.keys
    for (k <- keys) {
      this.dGradDict(k).set(this.dGradDict(k) + this.tempGradD(k))
    }
  }

  // update the model for a single batch
  def update(dBatch: DataBatch): Unit = {
    // generate fake image
    this.gData.set(Random.normal(0, 1.0f, this.gData.shape, context))
    this.gExecutor.forward(isTrain = true)
    val outG = this.gExecutor.outputs(0)
    this.dLabel.set(0f)
    this.dData.set(outG)
    this.dExecutor.forward(isTrain = true)
    this.dExecutor.backward()
    this.saveTempGradD()
    // update generator
    this.dLabel.set(1f)
    this.dExecutor.forward(isTrain = true)
    this.dExecutor.backward()
    this.gExecutor.backward(tempDiffD)
    gParamsGrads.foreach { case (idx, name, grad, optimState) =>
      gOpt.update(idx, gArgDict(name), grad, optimState)
    }
    this.outputsFake = this.dExecutor.outputs.map(x => x.copy())
    // update discriminator
    this.dLabel.set(posLabel)
    this.dData.set(dBatch.data(0))
    this.dExecutor.forward(isTrain = true)
    this.dExecutor.backward()
    this.addTempGradD()
    dParamsGrads.foreach { case (idx, name, grad, optimState) =>
      dOpt.update(idx, dArgDict(name), grad, optimState)
    }
    this.outputsReal = this.dExecutor.outputs.map(x => x.copy())
    this.tempOutG.indices.foreach(i => this.tempOutG(i).set(this.gExecutor.outputs(i)))
  }
}
