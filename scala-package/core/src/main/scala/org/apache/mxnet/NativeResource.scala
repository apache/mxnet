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


/**
  * NativeResource trait is used to manage MXNet Objects
  * such as NDArray, Symbol, Executor, etc.,
  * The MXNet Object calls NativeResource.register
  * and assign the returned NativeResourceRef to PhantomReference
  * NativeResource also implements AutoCloseable so MXNetObjects
  * can be used like Resources in try-with-resources paradigm
  */
private[mxnet] trait NativeResource
  extends AutoCloseable with WarnIfNotDisposed {

  /**
    * native Address associated with this object
    */
  def nativeAddress: CPtrAddress

  /**
    * Function Pointer to the NativeDeAllocator of nativeAddress
    */
  def nativeDeAllocator: (CPtrAddress => Int)

  /**
    * Call NativeResource.register to get the reference
    */
  val ref: NativeResourceRef

  /**
    * Off-Heap Bytes Allocated for this object
    */
  // intentionally making it a val, so it gets evaluated when defined
  val bytesAllocated: Long

  // this is set and unset by [[ResourceScope.add]] and [[ResourceScope.remove]]
  private[mxnet] var scope: Option[ResourceScope] = None

  @volatile private var disposed = false

  override def isDisposed: Boolean = disposed || isDeAllocated

  /**
    * Register this object for PhantomReference tracking and in
    * ResourceScope if used inside ResourceScope.
    * @return NativeResourceRef that tracks reachability of this object
    *         using PhantomReference
    */
  def register(): NativeResourceRef = {
    val scope = ResourceScope.getCurrentScope()
    if (scope.isDefined) scope.get.add(this)

    NativeResource.totalBytesAllocated.getAndAdd(bytesAllocated)
    // register with PhantomRef tracking to release in case the objects go
    // out of reference within scope but are held for long time
    NativeResourceRef.register(this, nativeDeAllocator)
 }

  // Implements [[@link AutoCloseable.close]]
  override def close(): Unit = {
    dispose()
  }

  // Implements [[@link WarnIfNotDisposed.dispose]]
  def dispose(): Unit = dispose(true)

  /**
    * This method deAllocates nativeResource and deRegisters
    * from PhantomRef and removes from Scope if
    * removeFromScope is set to true.
    * @param removeFromScope remove from the currentScope if true
    */
  // the parameter here controls whether to remove from current scope.
  // [[ResourceScope.close]] calls NativeResource.dispose
  // if we remove from the ResourceScope ie., from the container in ResourceScope.
  // while iterating on the container, calling iterator.next is undefined and not safe.
  // Note that ResourceScope automatically disposes all the resources within.
  private[mxnet] def dispose(removeFromScope: Boolean = true): Unit = {
    if (!disposed) {
      checkCall(nativeDeAllocator(this.nativeAddress))
      NativeResourceRef.deRegister(ref) // removes from PhantomRef tracking
      if (removeFromScope && scope.isDefined) scope.get.remove(this)
      NativeResource.totalBytesAllocated.getAndAdd(-1*bytesAllocated)
      disposed = true
    }
  }

  /*
  this is used by the WarnIfNotDisposed finalizer,
  the object could be disposed by the GC without the need for explicit disposal
  but the finalizer might not have run, then the WarnIfNotDisposed throws a warning
   */
  private[mxnet] def isDeAllocated(): Boolean = NativeResourceRef.isDeAllocated(ref)

}

private[mxnet] object NativeResource {
  var totalBytesAllocated : AtomicLong = new AtomicLong(0)
}

// Do not make [[NativeResource.resource]] a member of the class,
// this will hold reference and GC will not clear the object.
private[mxnet] class NativeResourceRef(resource: NativeResource,
                                       val resourceDeAllocator: CPtrAddress => Int)
        extends PhantomReference[NativeResource](resource, NativeResourceRef.refQ) {}

private[mxnet] object NativeResourceRef {

  private[mxnet] val refQ: ReferenceQueue[NativeResource]
                = new ReferenceQueue[NativeResource]

  private[mxnet] val refMap = new ConcurrentHashMap[NativeResourceRef, CPtrAddress]()

  private[mxnet] val cleaner = new ResourceCleanupThread()

  cleaner.start()

  def register(resource: NativeResource, nativeDeAllocator: (CPtrAddress => Int)):
  NativeResourceRef = {
    val ref = new NativeResourceRef(resource, nativeDeAllocator)
    refMap.put(ref, resource.nativeAddress)
    ref
  }

  // remove from PhantomRef tracking
  def deRegister(ref: NativeResourceRef): Unit = refMap.remove(ref)

  /**
    * This method will check if the cleaner ran and deAllocated the object
    * As a part of GC, when the object is unreachable GC inserts a phantomRef
    * to the ReferenceQueue which the cleaner thread will deallocate, however
    * the finalizer runs much later depending on the GC.
    * @param resource resource to verify if it has been deAllocated
    * @return true if already deAllocated
    */
  def isDeAllocated(ref: NativeResourceRef): Boolean = {
    !refMap.containsKey(ref)
  }

  def cleanup: Unit = {
    // remove is a blocking call
    val ref: NativeResourceRef = refQ.remove().asInstanceOf[NativeResourceRef]
    // phantomRef will be removed from the map when NativeResource.close is called.
    val resource = refMap.get(ref)
    if (resource != 0L)  { // since CPtrAddress is Scala a Long, it cannot be null
      ref.resourceDeAllocator(resource)
      refMap.remove(ref)
    }
  }

  protected class ResourceCleanupThread extends Thread {
    setPriority(Thread.MAX_PRIORITY)
    setName("NativeResourceDeAllocatorThread")
    setDaemon(true)

    override def run(): Unit = {
      while (true) {
        try {
          NativeResourceRef.cleanup
        }
        catch {
          case _: InterruptedException => Thread.currentThread().interrupt()
        }
      }
    }
  }
}
