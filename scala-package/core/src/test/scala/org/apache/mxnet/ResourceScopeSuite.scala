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
import org.apache.mxnet.ResourceScope.logger
import org.mockito.Matchers.any
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}
import org.mockito.Mockito._

class ResourceScopeSuite extends FunSuite with BeforeAndAfterAll with Matchers {

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

  test(testName = "testAutoReleasefromScope") {
    var a: NDArray = null
    var aRef: NativeResourceRef = null
    val b: NDArray = ResourceScope.using() {
        a = spy(NDArray.ones(Shape(3, 4)))
        print("testAutoReleasefromScope: a address %x\n".format(a.nativeAddress))
        aRef = a.ref
        val x = NDArray.ones(Shape(3, 4))
        print("testAutoReleasefromScope: x address %x\n".format(x.nativeAddress))
      x
    }
    val bRef: NativeResourceRef = b.ref
    assert(a.isDisposed == true, "objects created within scope should have isDisposed set to true")
    assert(b.isDisposed == false, "returned NativeResource should not be released")
    assert(TestRef.getRefMap.containsKey(aRef) == false,
      "reference of resource in Scope should be removed refMap")
    assert(TestRef.getRefMap.containsKey(bRef) == true,
      "reference of resource outside scope should be not removed refMap")

  }

  test("release from outerscope") {
    var a: NDArray = null
  }

}
