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
import scala.collection.mutable.HashMap

class ResourceScopeSuite extends FunSuite with BeforeAndAfterAll with Matchers {

  class TestNativeResource extends NativeResource {
    /**
      * native Address associated with this object
      */
    override def nativeAddress: CPtrAddress = hashCode()

    /**
      * Function Pointer to the NativeDeAllocator of nativeAddress
      */
    override def nativeDeAllocator: CPtrAddress => Int = TestNativeResource.deAllocator

    /** Call NativeResource.register to get the reference
      */
    override val ref: NativeResourceRef = super.register()
    /**
      * Off-Heap Bytes Allocated for this object
      */
    override val bytesAllocated: Long = 0
  }
  object TestNativeResource {
    def deAllocator(handle: CPtrAddress): Int = 0
  }

  object TestPhantomRef  {
    def getRefQueue: ReferenceQueue[NativeResource] = { NativeResourceRef.refQ}
    def getRefMap: ConcurrentHashMap[NativeResourceRef, CPtrAddress]
    = {NativeResourceRef.refMap}
    def getCleaner: Thread = { NativeResourceRef.cleaner }

  }

  class TestPhantomRef(resource: NativeResource,
                       resourceDeAllocator: CPtrAddress => Int)
    extends NativeResourceRef(resource, resourceDeAllocator) {
  }

  test(testName = "test NDArray Auto Release") {
    var a: NDArray = null
    var aRef: NativeResourceRef = null
    var b: NDArray = null

    ResourceScope.using() {
      b = ResourceScope.using() {
          a = NDArray.ones(Shape(3, 4))
          aRef = a.ref
          val x = NDArray.ones(Shape(3, 4))
        x
      }
      val bRef: NativeResourceRef = b.ref
      assert(a.isDisposed == true,
        "objects created within scope should have isDisposed set to true")
      assert(b.isDisposed == false,
        "returned NativeResource should not be released")
      assert(TestPhantomRef.getRefMap.containsKey(aRef) == false,
        "reference of resource in Scope should be removed refMap")
      assert(TestPhantomRef.getRefMap.containsKey(bRef) == true,
        "reference of resource outside scope should be not removed refMap")
    }
    assert(b.isDisposed, "resource returned from inner scope should be released in outer scope")
  }

  test("test return object release from outer scope") {
    var a: TestNativeResource = null
    ResourceScope.using() {
      a = ResourceScope.using() {
        new TestNativeResource()
      }
      assert(a.isDisposed == false, "returned object should not be disposed within Using")
    }
    assert(a.isDisposed == true, "returned object should be disposed in the outer scope")
  }

  test(testName = "test NativeResources in returned Lists are not disposed") {
    var ndListRet: IndexedSeq[TestNativeResource] = null
    ResourceScope.using() {
      ndListRet = ResourceScope.using() {
        val ndList: IndexedSeq[TestNativeResource] =
          IndexedSeq(new TestNativeResource(), new TestNativeResource())
        ndList
      }
      ndListRet.foreach(nd => assert(nd.isDisposed == false,
        "NativeResources within a returned collection should not be disposed"))
    }
    ndListRet.foreach(nd => assert(nd.isDisposed == true,
    "NativeResources returned from inner scope should be disposed in outer scope"))
  }

  test("test native resource inside a map") {
    var nRInKeyOfMap: HashMap[TestNativeResource, String] = null
    var nRInValOfMap: HashMap[String, TestNativeResource] = HashMap[String, TestNativeResource]()

    ResourceScope.using() {
      nRInKeyOfMap = ResourceScope.using() {
        val ret = HashMap[TestNativeResource, String]()
        ret.put(new TestNativeResource, "hello")
        ret
      }
      assert(!nRInKeyOfMap.isEmpty)

      nRInKeyOfMap.keysIterator.foreach(it => assert(it.isDisposed == false,
      "NativeResources returned in Traversable should not be disposed"))
    }

    nRInKeyOfMap.keysIterator.foreach(it => assert(it.isDisposed))

    ResourceScope.using() {

      nRInValOfMap = ResourceScope.using() {
        val ret = HashMap[String, TestNativeResource]()
        ret.put("world!", new TestNativeResource)
        ret
      }
      assert(!nRInValOfMap.isEmpty)
      nRInValOfMap.valuesIterator.foreach(it => assert(it.isDisposed == false,
        "NativeResources returned in Collection should not be disposed"))
    }
    nRInValOfMap.valuesIterator.foreach(it => assert(it.isDisposed))
  }

}
