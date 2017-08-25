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

import ml.dmlc.mxnet.{DataBatch, DataIter, NDArray, Shape}
import org.slf4j.LoggerFactory

import scala.collection.immutable.ListMap


/**
 * Resize a DataIter to given number of batches per epoch.
 * May produce incomplete batch in the middle of an epoch due
 * to padding from internal iterator.
 *
 * @param dataIter Internal data iterator.
 * @param reSize number of batches per epoch to resize to.
 * @param resetInternal whether to reset internal iterator on ResizeIter.reset
 */
class ResizeIter(
    dataIter: DataIter,
    reSize: Int,
    resetInternal: Boolean = true) extends DataIter {

  private val logger = LoggerFactory.getLogger(classOf[ResizeIter])

  private var currentBatch: DataBatch = null
  private var cur = 0


  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    cur = 0
    if(resetInternal) {
      dataIter.reset()
    }
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

  private def iterNext(): Boolean = {
    if (cur == reSize) {
      false
    } else {
      try {
        currentBatch = dataIter.next()
      } catch {
        case ex: NoSuchElementException => {
          dataIter.reset()
          currentBatch = dataIter.next()
        }
      }
      cur+=1
      true
    }
  }

  override def hasNext: Boolean = {
    if (currentBatch != null) {
      true
    } else {
      iterNext()
    }
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    currentBatch.data
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    currentBatch.label
  }

  /**
   * Get the index of current batch
   * @return the index of current batch
   */
  override def getIndex(): IndexedSeq[Long] = {
    currentBatch.index
  }

  /**
   * Get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): Int = {
    currentBatch.pad
  }

  override def batchSize: Int = {
    dataIter.batchSize
  }

  // The name and shape of data provided by this iterator
  override def provideData: ListMap[String, Shape] = {
    dataIter.provideData
  }

  // The name and shape of label provided by this iterator
  override def provideLabel: ListMap[String, Shape] = {
    dataIter.provideLabel
  }
}
