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

package ml.dmlc.mxnet.spark.utils

import scala.collection.Iterator

/**
 * Repeatable Iterator useful in mapPartitions
 * @author Yuance.Li
 */
class RepIterator[T](iteratorInternal: Iterator[T], repetition: Int = 1) extends Iterator[T] {
  assert(repetition > 0)
  var counter = repetition - 1
  var (currentIter, backupIter) = iteratorInternal.duplicate

  override def hasNext: Boolean = {
    currentIter.hasNext || counter > 0
  }

  override def next(): T = {
    assert(hasNext)
    if(currentIter.hasNext) {
      currentIter.next()
    } else if (counter > 0) {
      counter = counter - 1
      var iterTuple = backupIter.duplicate
      currentIter = iterTuple._1
      backupIter = iterTuple._2
      currentIter.next()
    } else {
      throw new NoSuchElementException("No element in this collection")
    }
  }
}
