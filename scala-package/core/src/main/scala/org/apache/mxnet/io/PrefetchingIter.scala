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

import org.apache.mxnet._
import org.slf4j.LoggerFactory
import java.util.concurrent.Semaphore

import org.apache.mxnet.DType.DType

import scala.collection.immutable.ListMap

/**
 * Base class for prefetching iterators. Takes one or more DataIters
 * and combine them with prefetching.
 *
 * @param iters list of DataIters
 * @param dataNames
 * @param labelNames
 */
class PrefetchingIter(
    iters: IndexedSeq[DataIter],
    dataNames: IndexedSeq[Map[String, String]] = null,
    labelNames: IndexedSeq[Map[String, String]] = null) extends DataIter {

  private val logger = LoggerFactory.getLogger(classOf[PrefetchingIter])

  require(iters.nonEmpty, "Iters length must be greater than 0")

  @deprecated("Please use provideDataDesc instead", "1.3.0")
  override def provideData: ListMap[String, Shape] = {
    if (dataNames == null) {
      iters.map(_.provideData).reduce(_ ++ _)
    } else {
      iters.map(_.provideData).zip(dataNames).map { case (providedData, names) =>
        providedData.map { case (oldName, shape) => names(oldName) -> shape }
      }.reduceLeft(_ ++ _)
    }
  }

  @deprecated("Please use provideDataDesc instead", "1.3.0")
  override def provideLabel: ListMap[String, Shape] = {
    if (labelNames == null) {
      iters.map(_.provideLabel).reduce(_ ++ _)
    } else {
      iters.map(_.provideLabel).zip(labelNames).map { case (providedLabel, names) =>
        providedLabel.map { case (oldName, shape) => names(oldName) -> shape }
      }.reduceLeft(_ ++ _)
    }
  }

  override def provideDataDesc: IndexedSeq[DataDesc] = {
    if (dataNames == null) {
      iters.flatMap(_.provideDataDesc)
    } else {
      iters.map(_.provideDataDesc).zip(dataNames).flatMap { case (providedDataDesc, names) =>
          providedDataDesc.map(desc =>
            new DataDesc(names(desc.name), desc.shape, desc.dtype, desc.layout))
      }
    }
  }

  override def provideLabelDesc: IndexedSeq[DataDesc] = {
    if (labelNames == null) {
      iters.flatMap(_.provideLabelDesc)
    } else {
      iters.map(_.provideLabelDesc).zip(labelNames).flatMap { case (providedLabelDesc, names) =>
        providedLabelDesc.map(desc =>
          new DataDesc(names(desc.name), desc.shape, desc.dtype, desc.layout))
      }
    }
  }

  private val _batchSize: Int = this.provideDataDesc.head.shape(0)
  private val dataReady: IndexedSeq[Semaphore] =
                                        (0 until iters.length).map(i => new Semaphore(0))
  private val dataTaken: IndexedSeq[Semaphore] =
                                        (0 until iters.length).map(i => new Semaphore(1))

  @volatile private var started: Boolean = true
  private var currentBatch: DataBatch = null
  private val nextBatch: Array[DataBatch] = (0 until iters.length).map { i =>
    new DataBatch(null, null, null, 0)
  }.toArray

  // thread entry
  def prefetchFunc(i: Int): Runnable = new Runnable {
    override def run(): Unit = {
      while (started) {
        dataTaken(i).acquire()
        if (started) {
          try {
            nextBatch(i) = iters(i).next()
          } catch {
            case ex: NoSuchElementException => nextBatch(i) = null
          }
        }
        dataReady(i).release()
      }
    }
  }

  private val prefetchThreads =
    for (i <- 0 until iters.length) yield new Thread(prefetchFunc(i))
  prefetchThreads.foreach(_.start())

  override def next(): DataBatch = currentBatch

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    for (e <- dataReady) e.acquire()
    for (i <- iters) i.reset()
    for (e <- dataTaken) e.release()
  }

  override def batchSize: Int = this._batchSize

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = currentBatch.data

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = currentBatch.label

  /**
   * the index of current batch
   * @return
   */
  override def getIndex(): IndexedSeq[Long] = currentBatch.index

  /**
    * get the number of padding examples
    * in current batch
    * @return number of padding examples in current batch
    */
  override def getPad(): Int = this.currentBatch.pad

  override def hasNext: Boolean = {
    for (e <- dataReady) e.acquire()
    if (nextBatch(0) == null) {
      for (i <- nextBatch) {
        assert(i == null, "Number of entry mismatches between iterators")
      }
      for (e <- dataReady) e.release()
      false
    } else {
      for (batch <- nextBatch) {
        assert(batch.pad == nextBatch(0).pad,
            "Number of entry mismatches between iterators")
      }
      val datas = for (batch <- nextBatch) yield batch.data
      val labels = for (batch <- nextBatch) yield batch.label
      currentBatch = new DataBatch(datas.toIndexedSeq.flatten,
        labels.toIndexedSeq.flatten,
        nextBatch(0).index,
        nextBatch(0).pad)
      for (e <- dataTaken) e.release()
      true
    }
  }

  /**
   * Stop all its internal prefetching threads.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    started = false
    for (e <- dataTaken) e.release()
    for (t <- prefetchThreads) t.join()
  }
}
