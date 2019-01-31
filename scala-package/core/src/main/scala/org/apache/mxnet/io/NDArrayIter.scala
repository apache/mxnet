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

package org.apache.mxnet.io

import java.util.NoSuchElementException

import org.apache.mxnet.Base._
import org.apache.mxnet.DType.DType
import org.apache.mxnet._
import org.slf4j.LoggerFactory

import scala.collection.immutable.ListMap

/**
 * NDArrayIter object in mxnet. Taking NDArray to get dataiter.
 *
 * @param data Specify the data as well as the name.
 *             NDArrayIter supports single or multiple data and label.
 * @param label Same as data, but is not fed to the model during testing.
 * @param dataBatchSize Batch Size
 * @param shuffle Whether to shuffle the data
 * @param lastBatchHandle "pad", "discard" or "roll_over". How to handle the last batch
 *
 * This iterator will pad, discard or roll over the last batch if
 * the size of data does not match batch_size. Roll over is intended
 * for training and can cause problems if used for prediction.
 */
class NDArrayIter(data: IndexedSeq[(DataDesc, NDArray)],
                  label: IndexedSeq[(DataDesc, NDArray)],
                  private val dataBatchSize: Int, shuffle: Boolean,
                  lastBatchHandle: String) extends DataIter {

  /**
  * @param data Specify the data. Data names will be data_0, data_1, ..., etc.
  * @param label Same as data, but is not fed to the model during testing.
  *              Label names will be label_0, label_1, ..., etc.
  * @param dataBatchSize Batch Size
  * @param shuffle Whether to shuffle the data
  * @param lastBatchHandle "pad", "discard" or "roll_over". How to handle the last batch
  *
  * This iterator will pad, discard or roll over the last batch if
  * the size of data does not match batch_size. Roll over is intended
  * for training and can cause problems if used for prediction.
  */
  def this(data: IndexedSeq[NDArray], label: IndexedSeq[NDArray] = IndexedSeq.empty,
           dataBatchSize: Int = 1, shuffle: Boolean = false,
           lastBatchHandle: String = "pad",
           dataName: String = "data", labelName: String = "label") {
    this(IO.initDataDesc(data, allowEmpty = false, dataName,
      if (data == null || data.isEmpty)  MX_REAL_TYPE else data(0).dtype, Layout.UNDEFINED),
      IO.initDataDesc(label, allowEmpty = true, labelName, MX_REAL_TYPE, Layout.UNDEFINED),
      dataBatchSize, shuffle, lastBatchHandle)
  }

  private val logger = LoggerFactory.getLogger(classOf[NDArrayIter])

  val (initData: IndexedSeq[(DataDesc, NDArray)], initLabel: IndexedSeq[(DataDesc, NDArray)]) = {
    // data should not be null and size > 0
    require(data != null && data.size > 0,
      "data should not be null and data.size should not be zero")

    require(label != null,
      "label should not be null. Use IndexedSeq.empty if there are no labels")

    // shuffle is not supported currently
    require(!shuffle, "shuffle is not supported currently")

    // discard final part if lastBatchHandle equals discard
    if (lastBatchHandle.equals("discard")) {
      val dataSize = data(0)._2.shape(0)
      require(dataBatchSize <= dataSize,
        "batch_size need to be smaller than data size when not padding.")
      val keepSize = dataSize - dataSize % dataBatchSize
      val dataList = data.map { case (name, ndArray) => (name, ndArray.slice(0, keepSize)) }
      if (!label.isEmpty) {
        val labelList = label.map { case (name, ndArray) => (name, ndArray.slice(0, keepSize)) }
        (dataList, labelList)
      } else {
        (dataList, label)
      }
    } else {
      (data, label)
    }
  }

  val numData = initData(0)._2.shape(0)
  val numSource: MXUint = initData.size
  private var cursor = -dataBatchSize

  private val (_provideData: ListMap[String, Shape],
               _provideLabel: ListMap[String, Shape],
               _provideDataDesc: IndexedSeq[DataDesc],
               _provideLabelDesc: IndexedSeq[DataDesc]) = {
    val pData = ListMap.empty[String, Shape] ++ initData.map(getShape)
    val pLabel = ListMap.empty[String, Shape] ++ initLabel.map(getShape)
    val pDData = IndexedSeq.empty[DataDesc] ++ initData.map(ele => {
      val temp = getShape(ele)
      new DataDesc(temp._1, temp._2, ele._1.dtype, ele._1.layout)
    })
    val pDLabel = IndexedSeq.empty[DataDesc] ++ initLabel.map(ele => {
      val temp = getShape(ele)
      new DataDesc(temp._1, temp._2, ele._1.dtype, ele._1.layout)
    })
    (pData, pLabel, pDData, pDLabel)
  }

  /**
   * get shape via dataBatchSize
   * @param dataItem
   */
  private def getShape(dataItem: (DataDesc, NDArray)): (String, Shape) = {
    val len = dataItem._2.shape.size
    val newShape = dataItem._2.shape.slice(1, len)
    (dataItem._1.name, Shape(Array[Int](dataBatchSize)) ++ newShape)
  }


  /**
   * Igore roll over data and set to start
   */
  def hardReset(): Unit = {
    cursor = -dataBatchSize
  }

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    if (lastBatchHandle.equals("roll_over") && cursor > numData) {
      cursor = -dataBatchSize + (cursor%numData) % dataBatchSize
    } else {
      cursor = -dataBatchSize
    }
  }

  override def hasNext: Boolean = {
    if (cursor + dataBatchSize < numData) {
      true
    } else {
      false
    }
  }

  @throws(classOf[NoSuchElementException])
  override def next(): DataBatch = {
    if (hasNext) {
      cursor += dataBatchSize
      new DataBatch(getData(), getLabel(), getIndex(), getPad(),
        null, null, null)
    } else {
      throw new NoSuchElementException
    }
  }

  /**
   * handle the last batch
   * @param ndArray
   * @return
   */
  private def _padData(ndArray: NDArray): NDArray = {
    val padNum = cursor + dataBatchSize - numData
    val shape = Shape(dataBatchSize) ++ ndArray.shape.slice(1, ndArray.shape.size)
    val newArray = NDArray.zeros(shape)
    NDArrayCollector.auto().withScope {
      val batch = ndArray.slice(cursor, numData)
      val padding = ndArray.slice(0, padNum)
      newArray.slice(0, dataBatchSize - padNum).set(batch)
      newArray.slice(dataBatchSize - padNum, dataBatchSize).set(padding)
      newArray
    }
  }

  private def _getData(data: IndexedSeq[(DataDesc, NDArray)]): IndexedSeq[NDArray] = {
    require(cursor < numData, "DataIter needs reset.")
    if (data == null) {
      null
    } else {
      if (cursor + dataBatchSize <= numData) {
        data.map { case (_, ndArray) => ndArray.slice(cursor, cursor + dataBatchSize) }
      } else {
        // padding
        data.map { case (_, ndArray) => _padData(ndArray) }
      }
    }
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    _getData(initData)
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    _getData(initLabel)
  }

  /**
   * the index of current batch
   * @return
   */
  override def getIndex(): IndexedSeq[Long] = {
    cursor.toLong to (cursor + dataBatchSize).toLong
  }

  /**
   * get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): MXUint = {
    if (lastBatchHandle.equals("pad") && cursor + batchSize > numData) {
      cursor + batchSize - numData
    } else {
      0
    }
  }


  // The name and shape of data provided by this iterator
  @deprecated("Please use provideDataDesc instead", "1.3.0")
  override def provideData: ListMap[String, Shape] = _provideData

  // The name and shape of label provided by this iterator
  @deprecated("Please use provideLabelDesc instead", "1.3.0")
  override def provideLabel: ListMap[String, Shape] = _provideLabel

  // Provide type:DataDesc of the data
  override def provideDataDesc: IndexedSeq[DataDesc] = _provideDataDesc

  // Provide type:DataDesc of the label
  override def provideLabelDesc: IndexedSeq[DataDesc] = _provideLabelDesc

  override def batchSize: Int = dataBatchSize
}

object NDArrayIter {

  /**
   * Builder class for NDArrayIter.
   */
  class Builder() {
    private var data: IndexedSeq[(DataDesc, NDArray)] = IndexedSeq.empty
    private var label: IndexedSeq[(DataDesc, NDArray)] = IndexedSeq.empty
    private var dataBatchSize: Int = 1
    private var lastBatchHandle: String = "pad"

    /**
     * Add one data input with its name.
     * @param name Data name.
     * @param data Data nd-array.
     * @return The builder object itself.
     */
    def addData(name: String, data: NDArray): Builder = {
      this.data = this.data ++ IndexedSeq((new DataDesc(name,
        data.shape, data.dtype, Layout.UNDEFINED), data))
      this
    }

    /**
     * Add one label input with its name.
     * @param name Label name.
     * @param label Label nd-array.
     * @return The builder object itself.
     */
    def addLabel(name: String, label: NDArray): Builder = {
      this.label = this.label ++ IndexedSeq((new DataDesc(name,
        label.shape, label.dtype, Layout.UNDEFINED), label))
      this
    }

    /**
      * Add one data input with its DataDesc
     */
    def addDataWithDesc(dataDesc: DataDesc, data: NDArray): Builder = {
      this.data = this.data ++ IndexedSeq((dataDesc, data))
      this
    }

    /**
      * Add one label input with its DataDesc
      */
    def addLabelWithDesc(labelDesc: DataDesc, label: NDArray): Builder = {
      this.data = this.data ++ IndexedSeq((labelDesc, label))
      this
    }

    /**
     * Set the batch size of the iterator.
     * @param batchSize batch size.
     * @return The builder object itself.
     */
    def setBatchSize(batchSize: Int): Builder = {
      this.dataBatchSize = batchSize
      this
    }

    /**
     * How to handle the last batch.
     * @param lastBatchHandle Can be "pad", "discard" or "roll_over".
     * @return The builder object itself.
     */
    def setLastBatchHandle(lastBatchHandle: String): Builder = {
      this.lastBatchHandle = lastBatchHandle
      this
    }

    /**
     * Build the NDArrayIter object.
     * @return the built object.
     */
    def build(): NDArrayIter = {
      new NDArrayIter(data, label, dataBatchSize, false, lastBatchHandle)
    }
  }
}
