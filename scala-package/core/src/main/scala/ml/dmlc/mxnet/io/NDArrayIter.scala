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

package ml.dmlc.mxnet.io

import java.util.NoSuchElementException

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet._
import org.slf4j.LoggerFactory

import scala.collection.immutable.ListMap

/**
 * NDArrayIter object in mxnet. Taking NDArray to get dataiter.
 *
 * @param data NDArrayIter supports single or multiple data and label.
 * @param label Same as data, but is not fed to the model during testing.
 * @param dataBatchSize Batch Size
 * @param shuffle Whether to shuffle the data
 * @param lastBatchHandle "pad", "discard" or "roll_over". How to handle the last batch
 *
 * This iterator will pad, discard or roll over the last batch if
 * the size of data does not match batch_size. Roll over is intended
 * for training and can cause problems if used for prediction.
 */
class NDArrayIter (data: IndexedSeq[NDArray], label: IndexedSeq[NDArray] = IndexedSeq.empty,
                  private val dataBatchSize: Int = 1, shuffle: Boolean = false,
                  lastBatchHandle: String = "pad",
                  dataName: String = "data", labelName: String = "label") extends DataIter {
  private val logger = LoggerFactory.getLogger(classOf[NDArrayIter])


  private val (_dataList: IndexedSeq[NDArray],
  _labelList: IndexedSeq[NDArray]) = {
    // data should not be null and size > 0
    require(data != null && data.size > 0,
      "data should not be null and data.size should not be zero")

    require(label != null,
      "label should not be null. Use IndexedSeq.empty if there are no labels")

    // shuffle is not supported currently
    require(shuffle == false, "shuffle is not supported currently")

    // discard final part if lastBatchHandle equals discard
    if (lastBatchHandle.equals("discard")) {
      val dataSize = data(0).shape(0)
      require(dataBatchSize <= dataSize,
        "batch_size need to be smaller than data size when not padding.")
      val keepSize = dataSize - dataSize % dataBatchSize
      val dataList = data.map(ndArray => {ndArray.slice(0, keepSize)})
      if (!label.isEmpty) {
        val labelList = label.map(ndArray => {ndArray.slice(0, keepSize)})
        (dataList, labelList)
      } else {
        (dataList, label)
      }
    } else {
      (data, label)
    }
  }


  val initData: IndexedSeq[(String, NDArray)] = IO.initData(_dataList, false, dataName)
  val initLabel: IndexedSeq[(String, NDArray)] = IO.initData(_labelList, true, labelName)
  val numData = _dataList(0).shape(0)
  val numSource = initData.size
  var cursor = -dataBatchSize


  private val (_provideData: ListMap[String, Shape],
               _provideLabel: ListMap[String, Shape]) = {
    val pData = ListMap.empty[String, Shape] ++ initData.map(getShape)
    val pLabel = ListMap.empty[String, Shape] ++ initLabel.map(getShape)
    (pData, pLabel)
  }

  /**
   * get shape via dataBatchSize
   * @param dataItem
   */
  private def getShape(dataItem: (String, NDArray)): (String, Shape) = {
    val len = dataItem._2.shape.size
    val newShape = dataItem._2.shape.slice(1, len)
    (dataItem._1, Shape(Array[Int](dataBatchSize)) ++ newShape)
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
    if (lastBatchHandle.equals("roll_over") && cursor>numData) {
      cursor = -dataBatchSize + (cursor%numData)%dataBatchSize
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
      new DataBatch(getData(), getLabel(), getIndex(), getPad())
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
    val newArray = NDArray.zeros(ndArray.slice(0, dataBatchSize).shape)
    val batch = ndArray.slice(cursor, numData)
    val padding = ndArray.slice(0, padNum)
    newArray.slice(0, dataBatchSize - padNum).set(batch).dispose()
    newArray.slice(dataBatchSize - padNum, dataBatchSize).set(padding).dispose()
    batch.dispose()
    padding.dispose()
    newArray
  }

  private def _getData(data: IndexedSeq[NDArray]): IndexedSeq[NDArray] = {
    require(cursor < numData, "DataIter needs reset.")
    if (data == null) {
      null
    } else {
      if (cursor + dataBatchSize <= numData) {
        data.map(ndArray => {ndArray.slice(cursor, cursor + dataBatchSize)}).toIndexedSeq
      } else {
        // padding
        data.map(_padData).toIndexedSeq
      }
    }
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    _getData(_dataList)
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    _getData(_labelList)
  }

  /**
   * the index of current batch
   * @return
   */
  override def getIndex(): IndexedSeq[Long] = {
    (cursor.toLong to (cursor + dataBatchSize).toLong).toIndexedSeq
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
  override def provideData: ListMap[String, Shape] = _provideData

  // The name and shape of label provided by this iterator
  override def provideLabel: ListMap[String, Shape] = _provideLabel

  override def batchSize: Int = dataBatchSize
}
