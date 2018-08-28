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

import java.lang.ref.{PhantomReference, ReferenceQueue}
import java.util.concurrent.ConcurrentHashMap
import org.apache.mxnet.Base.checkCall

/**
  * Should be generic to All MXNet Objects
  * Should call the DeAlloc automatically
  * Should do it only if not disposed. ie., dispose removes from the refQ
  */

private[mxnet] trait MXNativeHandle extends AutoCloseable {

  def nativeAddress: CPtrAddress

  def nativeDeAllocAddress: (CPtrAddress => Int)

  val phantomRef: MXHandlePhantomRef

  def bytesAllocated: Long

  var isDisposed: Boolean = false

  def register(referent: MXNativeHandle): MXHandlePhantomRef = {
    MXHandlePhantomRef.register(this, nativeDeAllocAddress)
 }

  def deRegister(phantomRef: MXHandlePhantomRef): Unit = {
    MXHandlePhantomRef.deRegister(phantomRef)
  }

  /* call {@link deAllocFn} if !{@link isDispose} */
  def dispose(): Unit = {
    print("dispose called")
    if (!isDisposed) {
      checkCall(nativeDeAllocAddress(this.nativeAddress))
      deRegister(phantomRef)
      isDisposed = true
    }
  }

  override def close(): Unit = {
    print("close called")
    dispose()
  }
}

private[mxnet] class MXHandlePhantomRef(h: MXNativeHandle, val deAllocFn: CPtrAddress => Int)
        extends PhantomReference[MXNativeHandle](h, MXHandlePhantomRef.phantomRefQ) {
}

object MXHandlePhantomRef {
  private val phantomRefQ: ReferenceQueue[MXNativeHandle] = new ReferenceQueue[MXNativeHandle]

  private val phantomRefMap = new ConcurrentHashMap[MXHandlePhantomRef, CPtrAddress]()

  def register(referent: MXNativeHandle, deAllocNativeAddr: CPtrAddress => Int):
  MXHandlePhantomRef = {
    val ref = new MXHandlePhantomRef(referent, deAllocNativeAddr)
    phantomRefMap.put(ref, referent.nativeAddress)
    ref
  }

  def deRegister(phantomRef: MXHandlePhantomRef): Unit = {
    val r = phantomRefMap.get(phantomRef)
    if (r != null) {
      phantomRefMap.remove(phantomRef)
    }
  }

  def cleanUp(): Unit = {
    var ref: MXHandlePhantomRef = phantomRefQ.poll().asInstanceOf[MXHandlePhantomRef]

    while (ref != null) {
      val hdl = phantomRefMap.get(ref)
      // may be dispose or close was called on this
      if (hdl != null) {
        ref.deAllocFn(hdl)
        phantomRefMap.remove(ref)
      }
      ref = phantomRefQ.poll().asInstanceOf[MXHandlePhantomRef]
    }
  }
}