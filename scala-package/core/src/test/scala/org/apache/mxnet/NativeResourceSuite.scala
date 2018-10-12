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

import org.apache.mxnet.Base.CPtrAddress
import org.mockito.Matchers.any
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers, TagAnnotation}
import org.mockito.Mockito._

@TagAnnotation("resource")
class NativeResourceSuite extends FunSuite with BeforeAndAfterAll with Matchers {

  object TestRef  {
    def getRefQueue: ReferenceQueue[NativeResource] = { NativeResourceRef.refQ}
    def getRefMap: ConcurrentHashMap[NativeResourceRef, CPtrAddress]
    = {NativeResourceRef.refMap}
    def getCleaner: Thread = { NativeResourceRef.cleaner }
  }

  class TestRef(resource: NativeResource,
                          resourceDeAllocator: CPtrAddress => Int)
    extends NativeResourceRef(resource, resourceDeAllocator) {
  }

  test(testName = "test native resource setup/teardown") {
    val a = spy(NDArray.ones(Shape(2, 3)))
    val aRef = a.ref
    val spyRef = spy(aRef)

    assert(TestRef.getRefMap.containsKey(aRef) == true)
    a.close()
    verify(a).dispose()
    verify(a).nativeDeAllocator
    // resourceDeAllocator does not get called when explicitly closing
    verify(spyRef, times(0)).resourceDeAllocator

    assert(TestRef.getRefMap.containsKey(aRef) == false)
    assert(a.isDisposed == true, "isDisposed should be set to true after calling close")
  }

  test(testName = "test dispose") {
    val a: NDArray = spy(NDArray.ones(Shape(3, 4)))
    val aRef = a.ref
    val spyRef = spy(aRef)
    a.dispose()
    verify(a).nativeDeAllocator
    assert(TestRef.getRefMap.containsKey(aRef) == false)
    assert(a.isDisposed == true, "isDisposed should be set to true after calling close")
  }
}

