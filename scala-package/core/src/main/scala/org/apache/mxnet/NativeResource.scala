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
import java.util.concurrent._

import org.apache.mxnet.Base.checkCall
import java.lang.AutoCloseable

import scala.annotation.varargs
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Try

trait NativeResourceManager extends AutoCloseable{
  override def close(): Unit = {}
}

private[mxnet] object NativeResourceManager {

  // inspired from slide 21 of
  def using[T <:NativeResource, U](resource: T)(block: T => U): U = {
    try {
      block(resource)
    } finally {
      // TODO(nswamy@): handle exceptions
      if (resource != null) resource.close
    }
  }

  private val logger = LoggerFactory.getLogger(classOf[NativeResourceManager])

  private val gcFrequencyInSecProp = "mxnet.gcFrequencyInSeconds"
  private val gcAfterOffHeapBytesProp = "mxnet.gcAfterOffHeapBytes"
  private val maxPhysicalBytesProp = "mxnet.maxPhysicalBytes"

  // ask Jonathan about Singletons
  private var _scheduledExecutor: ScheduledExecutorService = null

  // set this to None at the end, so we don't run GC periodically by default
  private val defaultGCFrequency = 5

  private val periodicGCFrequency = Try(System.getProperty(
    gcFrequencyInSecProp).toInt).getOrElse(defaultGCFrequency)

  def createPeriodicGCExecutor(): Unit = {
    if (periodicGCFrequency != null && _scheduledExecutor == null) {
      val scheduledExecutor: ScheduledExecutorService =
        Executors.newSingleThreadScheduledExecutor(new ThreadFactory {
          override def newThread(r: Runnable): Thread = new Thread(r) {
            setName(classOf[NativeResourceManager].getCanonicalName)
            setDaemon(true)
          }
        })
      scheduledExecutor.scheduleAtFixedRate(new Runnable {
        override def run(): Unit = {
          logger.info("Calling System.gc")
          System.gc()
          logger.info("Done Calling System.gc")
          NativeResourcePhantomRef.cleanUp
          logger.info("Done Cleaning up Native Resources")
        }
      },
        periodicGCFrequency,
        periodicGCFrequency,
        TimeUnit.SECONDS
      )
      _scheduledExecutor = scheduledExecutor
    }
  }
}

private[mxnet] trait NativeResource extends AutoCloseable {

  def nativeAddress: CPtrAddress

  def nativeDeAllocAddress: (CPtrAddress => Int)

  val phantomRef: NativeResourcePhantomRef

  def bytesAllocated: Long

  var isDisposed: Boolean = false

  def register(referent: NativeResource): NativeResourcePhantomRef = {
    NativeResourcePhantomRef.register(this, nativeDeAllocAddress)
 }

  def deRegister(phantomRef: NativeResourcePhantomRef): Unit = {
    NativeResourcePhantomRef.deRegister(phantomRef)
  }

  /* call {@link deAllocFn} if !{@link isDispose} */
  def dispose(): Unit = {
    if (!isDisposed) {
      checkCall(nativeDeAllocAddress(this.nativeAddress))
      deRegister(phantomRef)
      isDisposed = true
    }
  }

  override def close(): Unit = {
    dispose()
  }
}

private[mxnet] class NativeResourcePhantomRef(h: NativeResource, val deAllocFn: CPtrAddress => Int)
        extends PhantomReference[NativeResource](h, NativeResourcePhantomRef.phantomRefQ) {
}

private[mxnet] object NativeResourcePhantomRef {
  private val phantomRefQ: ReferenceQueue[NativeResource] = new ReferenceQueue[NativeResource]

  private val phantomRefMap = new ConcurrentHashMap[NativeResourcePhantomRef, CPtrAddress]()

  def register(referent: NativeResource, deAllocNativeAddr: CPtrAddress => Int):
  NativeResourcePhantomRef = {
    val ref = new NativeResourcePhantomRef(referent, deAllocNativeAddr)
    phantomRefMap.put(ref, referent.nativeAddress)
    ref
  }

  def deRegister(phantomRef: NativeResourcePhantomRef): Unit = {
    val r = phantomRefMap.get(phantomRef)
    if (r != null) {
      phantomRefMap.remove(phantomRef)
    }
  }

  def cleanUp(): Unit = {
    var ref: NativeResourcePhantomRef = phantomRefQ.poll().asInstanceOf[NativeResourcePhantomRef]

    while (ref != null) {
      val hdl = phantomRefMap.get(ref)
      // may be dispose or close was called on this
      if (hdl != null) {
        ref.deAllocFn(hdl)
        phantomRefMap.remove(ref)
      }
      ref = phantomRefQ.poll().asInstanceOf[NativeResourcePhantomRef]
    }
  }
}