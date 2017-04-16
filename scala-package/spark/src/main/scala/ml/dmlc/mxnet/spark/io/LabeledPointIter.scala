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

package ml.dmlc.mxnet.spark.io

import ml.dmlc.mxnet.{DataBatch, NDArray, Shape, DataIter}
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.immutable.ListMap
import scala.collection.mutable.ArrayBuffer

/**
 * A helper converter for LabeledPoint
 * @author Yizhi Liu
 */
class LabeledPointIter private[mxnet](
  private val points: Iterator[LabeledPoint],
  private val dimension: Shape,
  private val _batchSize: Int,
  private val dataName: String = "data",
  private val labelName: String = "label") extends DataIter {

  private val cache: ArrayBuffer[DataBatch] = ArrayBuffer.empty[DataBatch]
  private var index: Int = -1
  private val dataShape = Shape(_batchSize) ++ dimension

  def dispose(): Unit = {
    cache.foreach(_.dispose())
  }

  /**
   * reset the iterator
   */
  override def reset(): Unit = {
    index = -1
  }

  @throws(classOf[NoSuchElementException])
  override def next(): DataBatch = {
    if (!hasNext) {
      throw new NoSuchElementException("No more data")
    }
    index += 1
    if (index >= 0 && index < cache.size) {
      cache(index)
    } else {
      val dataBuilder = NDArray.empty(dataShape)
      val labelBuilder = NDArray.empty(_batchSize)
      var instNum = 0
      while (instNum < batchSize && points.hasNext) {
        val point = points.next()
        val features = point.features.toArray.map(_.toFloat)
        require(features.length == dimension.product,
          s"Dimension mismatch: ${features.length} != $dimension")
        dataBuilder.slice(instNum).set(features)
        labelBuilder.slice(instNum).set(Array(point.label.toFloat))
        instNum += 1
      }
      val pad = batchSize - instNum
      val dataBatch = new LongLivingDataBatch(
        IndexedSeq(dataBuilder), IndexedSeq(labelBuilder), null, pad)
      cache += dataBatch
      dataBatch
    }
  }

  /**
   * get data of current batch
   * @return the data of current batch
   */
  override def getData(): IndexedSeq[NDArray] = {
    if (index >= 0 && index < cache.size) {
      cache(index).data
    } else {
      null
    }
  }

  /**
   * Get label of current batch
   * @return the label of current batch
   */
  override def getLabel(): IndexedSeq[NDArray] = {
    if (index >= 0 && index < cache.size) {
      cache(index).label
    } else {
      null
    }
  }

  /**
   * Get the index of current batch
   * @return the index of current batch
   */
  override def getIndex(): IndexedSeq[Long] = {
    if (index >= 0 && index < cache.size) {
      cache(index).index
    } else {
      null
    }
  }

  // The name and shape of label provided by this iterator
  override def provideLabel: ListMap[String, Shape] = {
    ListMap(labelName -> Shape(_batchSize))
  }

  // The name and shape of data provided by this iterator
  override def provideData: ListMap[String, Shape] = {
    ListMap(dataName -> dataShape)
  }

  /**
   * Get the number of padding examples
   * in current batch
   * @return number of padding examples in current batch
   */
  override def getPad(): Int = 0

  override def batchSize: Int = _batchSize

  override def hasNext: Boolean = {
    points.hasNext || (index < cache.size - 1 && cache.size > 0)
  }
}
