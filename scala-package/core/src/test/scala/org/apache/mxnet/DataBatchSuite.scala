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

package org.apache.mxnet

import java.lang.ref.ReferenceQueue
import java.util.concurrent.ConcurrentHashMap

import org.apache.mxnet
import org.apache.mxnet.Base.CPtrAddress
import org.mockito.Mockito._
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers, TagAnnotation}

@TagAnnotation("resource")
class DataBatchSuite extends FunSuite with BeforeAndAfterAll with Matchers {

  object TestRef  {
    def getRefQueue: ReferenceQueue[NativeResource] = { NativeResourceRef.referenceQueue}
    def getRefMap: ConcurrentHashMap[NativeResourceRef, CPtrAddress]
    = {NativeResourceRef.phantomRefMap}
    def getCleaner: Thread = { NativeResourceRef.cleaner }
  }

  class TestRef(resource: NativeResource,
                          resourceDeAllocator: CPtrAddress => Int)
    extends NativeResourceRef(resource, resourceDeAllocator) {
  }

  test(testName = "test DataBatch dispose") {
    val dataArray: IndexedSeq[NDArray]
    = IndexedSeq.fill[NDArray](10)(NDArray.ones(Shape (3, 4)))
    val labelArray =
      IndexedSeq.fill[NDArray](10)(NDArray.ones(Shape (1, 2)))
    val index = IndexedSeq.fill[Long](10)(0L)
    val dBatch: DataBatch = new DataBatch(dataArray, labelArray, index, 0)
    val dBatchSpy = spy(dBatch)

    val aRefs = dataArray.map(_.phantomRef)
    val batchRef = dBatch.phantomRef

    aRefs.foreach(r => assert(TestRef.getRefMap.containsKey(r) == true))
    assert(TestRef.getRefMap.containsKey(batchRef))
    dBatchSpy.close()
    verify(dBatchSpy, times(1)).dispose()
    aRefs.foreach(r => assert(TestRef.getRefMap.containsKey(r) == false))
    assert(TestRef.getRefMap.containsKey(batchRef) == false)
  }
}

