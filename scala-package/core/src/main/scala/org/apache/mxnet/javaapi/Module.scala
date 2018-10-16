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

package org.apache.mxnet.javaapi
import collection.JavaConverters._
import scala.collection.mutable

class Module(val module : org.apache.mxnet.module.Module) {

  case class ArgAux(argParams: java.util.Map[String, NDArray], auxParams: java.util.Map[String, NDArray])

  def dataShape : Array[DataDesc] = module.dataShapes.map(DataDesc.fromDataDesc).toArray

  def labelShape : Array[DataDesc] = module.labelShapes.map(DataDesc.fromDataDesc).toArray

  def outputShapes : java.util.Map[String, Shape]
  = module.outputShapes.map(ele => (ele._1, Shape.fromShape(ele._2))).toMap.asJava

  def outputNames : Array[String] = module.outputNames.toArray

  def getParams : ArgAux = {
    val result = module.getParams
    val JavaArgParams = mutable.Map[String, NDArray]()
    val JavaAuxParams = mutable.Map[String, NDArray]()
    result._1.foreach(ele => JavaArgParams(ele._1) = ele._2)
    result._2.foreach(ele => JavaArgParams(ele._1) = ele._2)
    ArgAux(JavaArgParams.asJava, JavaAuxParams.asJava)
  }

  def setParams(argParams: java.util.Map[String, NDArray], auxParams: java.util.Map[String, NDArray],
                allowMissing: Boolean, forceInit : Boolean, allowExtra : Boolean) : Unit = {
    module.setParams(argParams.asScala.map(ele => (ele._1, NDArray.toNDArray(ele._2))).toMap,
      auxParams.asScala.map(ele => (ele._1, NDArray.toNDArray(ele._2))).toMap,
      allowMissing, forceInit, allowExtra)
  }

  // TODO: Extend the full bind functionalities
  def bind(dataShapes : Array[DataDesc], forTraining: Boolean, inputsNeedGrad: Boolean,
           forceRebind: Boolean) : Unit = {
    module.bind(forTraining, inputsNeedGrad, forceRebind, dataShapes.map(DataDesc.toDataDesc): _*)
  }

  def forward(dataBatch: DataBatch, isTrain : Boolean) : Unit = {
    module.forward(dataBatch, Some(isTrain))
  }

  def backward(outGrads : Array[NDArray]) : Unit = {
    module.backward(outGrads.map(NDArray.toNDArray))
  }

  // TODO: Extends with training features
}

object Module {
  implicit def toModule(module : org.apache.mxnet.module.Module) : Module = {
    new Module(module)
  }

  implicit def fromModule(module : Module) : org.apache.mxnet.module.Module = {
    module.module
  }

  def loadCheckpoint(prefix: String, epoch: Int, loadOptimizerStates: Boolean = false,
                     dataNames: Array[String],
                     labelNames: Array[String],
                     contexts: Array[Context],
                     workLoadList: Array[Float],
                     fixedParamNames: Set[String]) : Module = {
    org.apache.mxnet.module.Module.loadCheckpoint(prefix, epoch, loadOptimizerStates,
      dataNames, labelNames, contexts.map(Context.toContext), Some(workLoadList), Some(fixedParamNames))
  }

  class Builder(private val modelDef : Symbol) {
    private val builder : org.apache.mxnet.module.Module.Builder
    = new org.apache.mxnet.module.Module.Builder(modelDef)

    def setContext(ctx : Array[Context]) : Builder = {
      builder.setContext(ctx.map(Context.toContext) : _*)
      this
    }

    def setDataNames(name : Array[String]) : Builder = {
      builder.setDataNames(name : _*)
      this
    }

    def setLabelNames(name : Array[String]) : Builder = {
      builder.setLabelNames(name : _*)
      this
    }

    def setWorkLoadList(workloads : Array[Float]) : Builder = {
      builder.setWorkLoadList(workloads : _*)
      this
    }

    def setFixedParamNames(name : Array[String]) : Builder = {
      builder.setFixedParamNames(name : _*)
      this
    }

    def build() : Module = {
      builder.build()
    }
  }

}
