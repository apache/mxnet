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

import scala.ref.{PhantomReference, ReferenceQueue}
import java.util.concurrent.ConcurrentHashMap
import org.apache.mxnet.Base.checkCall

/**
  * Should be generic to All MXNet Objects
  * Should call the DeAlloc automatically
  * Should do it only if not disposed. ie., dispose removes from the refQ
  */

private[mxnet] trait MXNetHandle extends AutoCloseable {

  val nativeAddress: CPtrAddress

  val bytesAllocated: Long

  var isDisposed: Boolean = false

  val deAllocFn = (mxFreeHandleAddress: CPtrAddress => Int)

  def register(referent: MXNetHandle): Unit = {
    MXNetHandlePhantomRef.register(this, deAllocFn)
  }

  def deRegister(referent: MXNetHandle): Unit = {
    MXNetHandlePhantomRef.deRegister(referent)
  }

  /* call {@link deAllocFn} if !{@link isDispose} */
  def dispose(): Unit = {
    if (!isDisposed) {
      checkCall(checkdeAllocFn)
      isDisposed = true
      deRegister(this)
    }
  }

  override def close(): Unit = {
    dispose()
  }
}

/**
  * Fill me in
  * @param h
  * @param deAllocFn
  */
private[mxnet] class MXNetHandlePhantomRef(h: MXNetHandle, val deAllocFn: CPtrAddress => Int)
        extends PhantomReference[MXNetHandle](h, refQ) {
}

object MXNetHandlePhantomRef {
  private val refQ: ReferenceQueue[MXNetHandle] = new ReferenceQueue[MXNetHandle]

  private val refs = new ConcurrentHashMap[MXNetHandlePhantomRef, CPtrAddress]()

  def register(referent: MXNetHandle, deAllocFn: CPtrAddress => Int): MXNetHandlePhantomRef = {
    val ref = new MXNetHandlePhantomRef(referent, deAllocFn)
    refs.put(ref, referent.nativeAddress)
    ref
  }

  def deRegister(referent: MXNetHandlePhantomRef): Unit = {
    if ((r = refs.get(referent)) != null) {
      refs.remove(referent)
    }
  }

  def cleanUp(): Unit = {
    var ref: MXNetHandlePhantomRef = refQ.poll().asInstanceOf[MXNetHandlePhantomRef]
    while (ref != null) {
      // may be dispose or close was called on this
      if ((hdl = refs.get(ref)) != null) {
        ref.deAllocFn(hdl)
        refs.remove(ref)
      }
      ref = refQ.poll().asInstanceOf[MXNetHandlePhantomRef]
    }
  }
}