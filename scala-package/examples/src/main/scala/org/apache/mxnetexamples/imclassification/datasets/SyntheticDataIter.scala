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

package org.apache.mxnetexamples.imclassification.datasets

import org.apache.mxnet.DType.DType
import org.apache.mxnet._

import scala.collection.immutable.ListMap
import scala.util.Random

class SyntheticDataIter(numClasses: Int, val batchSize: Int, datumShape: List[Int],
                        labelShape: List[Int], maxIter: Int, dtype: DType = DType.Float32
                       ) extends DataIter {
  var curIter = 0
  val random = new Random()
  val shape = Shape(batchSize :: datumShape)
  val batchLabelShape = Shape(batchSize :: labelShape)

  val maxLabel = if (labelShape.isEmpty) numClasses.toFloat else 1f
  var label: IndexedSeq[NDArray] = IndexedSeq(
    NDArray.api.random_uniform(Some(0f), Some(maxLabel), shape = Some(batchLabelShape)))
  var data: IndexedSeq[NDArray] = IndexedSeq(
    NDArray.api.random_uniform(shape = Some(shape)))

  val provideDataDesc: IndexedSeq[DataDesc] = IndexedSeq(
    new DataDesc("data", shape, dtype, Layout.UNDEFINED))
  val provideLabelDesc: IndexedSeq[DataDesc] = IndexedSeq(
    new DataDesc("softmax_label", batchLabelShape, dtype, Layout.UNDEFINED))
  val getPad: Int = 0

  override def getData(): IndexedSeq[NDArray] = data

  override def getIndex: IndexedSeq[Long] = IndexedSeq(curIter)

  override def getLabel: IndexedSeq[NDArray] = label

  override def hasNext: Boolean = curIter < maxIter - 1

  override def next(): DataBatch = {
    if (hasNext) {
      curIter += batchSize
      new DataBatch(data, label, getIndex, getPad, null, null, null)
    } else {
      throw new NoSuchElementException
    }
  }

  override def reset(): Unit = {
    curIter = 0
  }

  override def provideData: ListMap[String, Shape] = ListMap("data" -> shape)

  override def provideLabel: ListMap[String, Shape] = ListMap("softmax_label" -> batchLabelShape)
}
