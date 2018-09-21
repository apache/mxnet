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

import org.apache.mxnet.Base.CPtrAddress
import java.lang.ref.{PhantomReference, ReferenceQueue, WeakReference}
import java.util.concurrent._

import org.apache.mxnet.Base.checkCall
import java.util.concurrent.atomic.AtomicLong

import org.apache.mxnet.NativeResourceRef.phantomRefMap
import org.slf4j.{Logger, LoggerFactory}

/**
  * NativeResource trait is used to manage MXNet Objects
  * such as NDArray, Symbol, Executor, etc.,
  * The MXNet Object calls {@link NativeResource.register}
  * and assign the returned NativeResourceRef to {@link PhantomReference}
  * NativeResource also implements AutoCloseable so MXNetObjects
  * can be used like Resources in try-with-resources paradigm
  */
// scalastyle:off finalize
private[mxnet] trait NativeResource
  extends AutoCloseable with WarnIfNotDisposed {

  /**
    * native Address associated with this object
    */
  def nativeResource: CPtrAddress

  /**
    * Function Pointer to the NativeDeAllocator of {@link nativeAddress}
    */
  def nativeDeAllocator: (CPtrAddress => Int)

  /** Call {@link NativeResource.register} to get {@link NativeResourceRef}
    */
  val phantomRef: NativeResourceRef

  /**
    * Off-Heap Bytes Allocated for this object
    */
  // intentionally making it a val, so it gets evaluated when defined
  val bytesAllocated: Long

  private var scope: ResourceScope = null

  @volatile var disposed = false

  override def isDisposed: Boolean = disposed
  /**
    * Register this object for PhantomReference tracking and within
    * ResourceScope if used inside ResourceScope.
    * @return NativeResourceRef that tracks reachability of this object
    *         using PhantomReference
    */
  def register(): NativeResourceRef = {
    scope = ResourceScope.getScope()
    if (scope != null) scope.register(this)

    NativeResource.totalBytesAllocated.getAndAdd(bytesAllocated)
    // register with PhantomRef tracking to release incase the objects go
    // out of reference within scope but are held for long time
    NativeResourceRef.register(this, nativeDeAllocator)
 }

  /**
    * Removes this object from PhantomRef tracking and from ResourceScope
    * @param removeFromScope
    */
  private def deRegister(removeFromScope: Boolean = true): Unit = {
    NativeResourceRef.deRegister(phantomRef)
    if (scope != null && removeFromScope) scope.deRegister(this)
  }

  // Implements {@link AutoCloseable.close}
  override def close(): Unit = {
    dispose()
  }

  // Implements {@link WarnIfNotDisposed.dispose}
  def dispose(): Unit = {
    dispose(true)
  }

  def dispose(removeFromScope: Boolean): Unit = {
    if (!disposed) {
      print("NativeResource: Disposing NativeResource:%x\n".format(nativeResource))
      checkCall(nativeDeAllocator(this.nativeResource))
      deRegister(removeFromScope)
      NativeResource.totalBytesAllocated.getAndAdd(-1*bytesAllocated)
      disposed = true
    }
  }
}
// scalastyle:on finalize

private[mxnet] object NativeResource {
  var totalBytesAllocated : AtomicLong = new AtomicLong(0)
}
// do not make resource a member, this will hold reference and GC will not clear the object.
private[mxnet] class NativeResourceRef(resource: NativeResource,
                                       val resourceDeAllocator: CPtrAddress => Int)
        extends PhantomReference[NativeResource](resource, NativeResourceRef.referenceQueue) {}

private[mxnet] object NativeResourceRef {

  private[mxnet] val referenceQueue: ReferenceQueue[NativeResource]
                = new ReferenceQueue[NativeResource]

  private[mxnet] val phantomRefMap = new ConcurrentHashMap[NativeResourceRef, CPtrAddress]()

  private[mxnet] val cleaner = new ResourceCleanupThread()

  cleaner.start()

  def register(resource: NativeResource, nativeDeAllocator: (CPtrAddress => Int)):
  NativeResourceRef = {
    val resourceRef = new NativeResourceRef(resource, nativeDeAllocator)
    phantomRefMap.put(resourceRef, resource.nativeResource)
    resourceRef
  }

  def deRegister(resourceRef: NativeResourceRef): Unit = {
    if (phantomRefMap.containsKey(resourceRef)) {
      phantomRefMap.remove(resourceRef)
    }
  }

  protected class ResourceCleanupThread extends Thread {
    setPriority(Thread.MAX_PRIORITY)
    setName("NativeResourceDeAllocatorThread")
    setDaemon(true)

    def deAllocate(): Unit = {
      print("NativeResourceRef: cleanup\n")
      // remove is a blocking call
      val ref: NativeResourceRef = referenceQueue.remove().asInstanceOf[NativeResourceRef]
      print("NativeResourceRef: got a reference with deAlloc\n")
      // phantomRef will be removed from the map when NativeResource.close is called.
      val resource = phantomRefMap.get(ref)
      if (resource != 0L)  { // since CPtrAddress is Scala Long, it cannot be null
        print("NativeResourceRef: got a reference for resource\n")
        ref.resourceDeAllocator(resource)
        phantomRefMap.remove(ref)
      }
    }

    override def run(): Unit = {
      while (true) {
        try {
          deAllocate()
        }
        catch {
          case _: InterruptedException => Thread.currentThread().interrupt()
        }
      }
    }
  }
}