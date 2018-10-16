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

class DataDesc(val dataDesc: org.apache.mxnet.DataDesc) {

  def this(name: String, shape: Shape, dType: DType.DType, layout: String) =
    this(new org.apache.mxnet.DataDesc(name, shape, dType, layout))

  override def toString(): String = dataDesc.toString()
}

object DataDesc{
  implicit def fromDataDesc(dDesc: org.apache.mxnet.DataDesc): DataDesc = new DataDesc(dDesc)

  implicit def toDataDesc(dataDesc: DataDesc): org.apache.mxnet.DataDesc = dataDesc.dataDesc

  def getBatchAxis(layout: String): Int = org.apache.mxnet.DataDesc.getBatchAxis(Some(layout))
}

class DataBatch(val dataBatch: org.apache.mxnet.DataBatch) {
  def dispose() : Unit = dataBatch.dispose()

  def provideDataDesc : Array[DataDesc] = dataBatch.provideDataDesc.map(DataDesc.fromDataDesc).toArray

  def provideLabelDesc : Array[DataDesc] = dataBatch.provideLabelDesc.map(DataDesc.fromDataDesc).toArray
}

object DataBatch {
  implicit def FromDataBatch(dataBatch : DataBatch) : org.apache.mxnet.DataBatch = {
    dataBatch.dataBatch
  }

  implicit def toDataBatch(dataBatch: org.apache.mxnet.DataBatch) : DataBatch = {
    new DataBatch(dataBatch)
  }

  class Builder() {

    private val builder : org.apache.mxnet.DataBatch.Builder = new org.apache.mxnet.DataBatch.Builder

    def setData(data : Array[NDArray]) : Builder = {
      this.builder.setData(data.map(NDArray.toNDArray) : _*)
      this
    }

    def setLabel(label : Array[NDArray]) : Builder = {
      this.builder.setLabel(label.map(NDArray.toNDArray) : _*)
      this
    }

    def setIndex(index : Array[Long]) : Builder = {
      this.builder.setIndex(index : _*)
      this
    }

    def setPad(pad : Int) : Builder = {
      this.builder.setPad(pad)
      this
    }

    def setDataDesc(dataDesc : Array[DataDesc]) : Builder = {
      this.builder.provideDataDesc(dataDesc.map(DataDesc.toDataDesc))
      this
    }

    def setLabelDesc(labelDesc : Array[DataDesc]) : Builder = {
      this.builder.provideLabelDesc(labelDesc.map(DataDesc.toDataDesc))
      this
    }

    def build() : DataBatch = {
      this.builder.build()
    }

    // TODO: Extends for Bucketing support
  }
}
