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

import ml.dmlc.mxnet.Base._
import ml.dmlc.mxnet.{DataPack, DataBatch, DataIter, NDArray, Shape}
import ml.dmlc.mxnet.IO._
import org.slf4j.LoggerFactory

import scala.collection.immutable.ListMap
import scala.collection.mutable.ListBuffer

/**
 * DataIter built in MXNet.
 * @param handle the handle to the underlying C++ Data Iterator
 */
// scalastyle:off finalize
private[mxnet] class MXDataIter(private[mxnet] val handle: DataIterHandle,
                                dataName: String = "data",
                                labelName: String = "label") extends DataIter {
  private val logger = LoggerFactory.getLogger(classOf[MXDataIter])

  // use currentBatch to implement hasNext
  // (may be this is not the best way to do this work,
  // fix me if any better way found)
  private var currentBatch: DataBatch = null

  private val (_provideData: ListMap[String, Shape],
               _provideLabel: ListMap[String, Shape],
               _batchSize: Int) =
    if (hasNext) {
      iterNext()
      val data = currentBatch.data(0)
      val label = currentBatch.label(0)
      // properties
      val res = (ListMap(dataName -> data.shape), ListMap(labelName -> label.shape), data.shape(0))
      currentBatch.dispose()
      reset()
      res
    } else {
      (null, null, 0)
    }

  private var disposed = false
  override protected def finalize(): Unit = {
    dispose()
  }

  /**
   * Release the native memory.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (!disposed) {
      _LIB.mxDataIterFree(handle)
      disposed = true
    }
  }

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    currentBatch = null
    checkCall(_LIB.mxDataIterBeforeFirst(handle))
  }

  @throws(classOf[NoSuchElementException])
  override def next(): DataBatch = {
    if (currentBatch == null) {
      iterNext()
    }

    if (currentBatch != null) {
      val batch = currentBatch
      currentBatch = null
      batch
    } else {
      throw new NoSuchElementException
    }
  }

  /**
   * Iterate to next batch
   * @return whether the move is successful
   */
  private def iterNext(): Boolean = {
    val next = new RefInt
    checkCall(_LIB.mxDataIterNext(handle, next))
    currentBatch = null
    if (next.value > 0) {
      currentBatch = new DataBatch(data = getData(), label = getLabel(),
        index = getIndex(), pad = getPad())
    }
    next.value > 0
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    val out = new NDArrayHandleRef
    checkCall(_LIB.mxDataIterGetData(handle, out))
    IndexedSeq(new NDArray(out.value, writable = false))
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    val out = new NDArrayHandleRef
    checkCall(_LIB.mxDataIterGetLabel(handle, out))
    IndexedSeq(new NDArray(out.value, writable = false))
  }

  /**
   * Get the index of current batch
   * @return the index of current batch
   */
  override def getIndex(): IndexedSeq[Long] = {
    val outIndex = new ListBuffer[Long]
    val outSize = new RefLong
    checkCall(_LIB.mxDataIterGetIndex(handle, outIndex, outSize))
    outIndex.toIndexedSeq
  }

  /**
   * get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): MXUint = {
    val out = new MXUintRef
    checkCall(_LIB.mxDataIterGetPadNum(handle, out))
    out.value
  }

  // The name and shape of data provided by this iterator
  override def provideData: ListMap[String, Shape] = _provideData

  // The name and shape of label provided by this iterator
  override def provideLabel: ListMap[String, Shape] = _provideLabel

  override def hasNext: Boolean = {
    if (currentBatch != null) {
      true
    } else {
      iterNext()
    }
  }

  override def batchSize: Int = _batchSize
}

// scalastyle:on finalize
private[mxnet] class MXDataPack(iterName: String, params: Map[String, String]) extends DataPack {
  /**
    * get data iterator
    * @return DataIter
    */
  override def iterator: DataIter = {
    createIterator(iterName, params)
  }
}
