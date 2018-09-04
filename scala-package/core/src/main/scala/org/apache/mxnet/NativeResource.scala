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
import java.lang.ref.{WeakReference, PhantomReference, ReferenceQueue}
import java.util.concurrent._

import org.apache.mxnet.Base.checkCall
import java.lang.{AutoCloseable, ThreadLocal}

import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.{ArrayBuffer, ArrayStack}
import scala.util.Try

private[mxnet] class PeriodicGCDeAllocator {

}

private[mxnet] object PeriodicGCDeAllocator {

  private val logger = LoggerFactory.getLogger(classOf[PeriodicGCDeAllocator])

  private val gcFrequencyInSecProp = "mxnet.gcFrequencyInSeconds"
  private val gcAfterOffHeapBytesProp = "mxnet.gcAfterOffHeapBytes"
  private val maxPhysicalBytesProp = "mxnet.maxPhysicalBytes"
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
            setName(classOf[ResourceScope].getCanonicalName)
            setDaemon(true)
          }
        })
      scheduledExecutor.scheduleAtFixedRate(new Runnable {
        override def run(): Unit = {
          logger.info("Calling System.gc")
          System.gc()
          logger.info("Done Calling System.gc")
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

class ResourceScope extends AutoCloseable {
  import ResourceScope.{logger, resourceScope}

  private val resourceQ = new  ArrayBuffer[NativeResource]()
  resourceScope.get().+=(this)

  override def close(): Unit = {
    resourceQ.foreach(resource => if (resource != null) {
      logger.info("releasing resource:%x\n".format(resource.nativeAddress))
      resource.dispose()
      resource.deRegister(false)
    } else {logger.info("found resource which is null")}
    )
    ResourceScope.resourceScope.get().-=(this)
  }

   private[mxnet] def register(resource: NativeResource): Unit = {
    logger.info("ResourceScope: Registering Resource %x".format(resource.nativeAddress))
    resourceQ.+=(resource)
  }

  // TODO(@nswamy): this is linear in time, find better data structure
  private[mxnet] def deRegister(resource: NativeResource): Unit = {
    logger.info("ResourceScope: DeRegistering Resource %x".format(resource.nativeAddress))
    resourceQ.-=(resource)
  }
}

 object ResourceScope {

   private val logger = LoggerFactory.getLogger(classOf[ResourceScope])

   // inspired from slide 21 of
  def using[T](resource: ResourceScope)(block: => T): T = {
    require(resource != null)
    try {
      val ret = block
      ret match {
        case nRes: NativeResource =>
          resource.deRegister(nRes.asInstanceOf[NativeResource])
        case _ => // do nothing
      }
      ret
    } finally {
      // TODO(nswamy@): handle exceptions
      resource.close
    }
  }

   private[mxnet] val resourceScope = new ThreadLocal[ArrayBuffer[ResourceScope]] {
    override def initialValue(): ArrayBuffer[ResourceScope] =
      new ArrayBuffer[ResourceScope]()
  }

   private[mxnet] def getScope(): ResourceScope = {
     try {
       resourceScope.get().last
     } catch {
       case _: ArrayIndexOutOfBoundsException => null
       case _: NoSuchElementException => null
       case e: Exception => throw e
     }
  }
}

private[mxnet] trait NativeResource extends AutoCloseable {

  def nativeAddress: CPtrAddress

  def nativeDeAllocAddress: (CPtrAddress => Int)

  /** Call {@link NativeResource.register} to get NativeResourcePhantomRef
    *
     */
  val phantomRef: NativeResourceRef

  def bytesAllocated: Long

  var isDisposed: Boolean = false

  private var scope: ResourceScope = null

  def register(referent: NativeResource): NativeResourceRef = {
    scope = ResourceScope.getScope()
    if (scope != null)   {
      scope.register(this)
    }
    // register with PhantomRef tracking to release incase the objects go
    // out of reference within scope but are held for long time
    NativeResourceRef.register(this, nativeDeAllocAddress)
 }

  /**
    * remove from PhantomRef tracking and
    * ResourceScope tracking
    */
  def deRegister(removeFromScope: Boolean = true): Unit = {
    NativeResourceRef.deRegister(phantomRef)
    if (scope != null && removeFromScope) scope.deRegister(this)
  }

  override def close(): Unit = {
    dispose()
    deRegister(true)
  }

  /* call {@link deAllocFn} if !{@link isDispose} */
  final def dispose(): Unit = {
    if (!isDisposed) {
      print("NativeResource: Disposing NativeResource:%x\n".format(nativeAddress))
      checkCall(nativeDeAllocAddress(this.nativeAddress))
      isDisposed = true
    }
  }
}

// do not make nativeRes a member, this will hold reference and GC will not clear the object.
private[mxnet] class NativeResourceRef(resource: NativeResource,
                                       val resDeAllocAddr: CPtrAddress => Int)
        extends PhantomReference[NativeResource](resource, NativeResourceRef.referenceQueue) {
}

private[mxnet] object NativeResourceRef {

  private val referenceQueue: ReferenceQueue[NativeResource] = new ReferenceQueue[NativeResource]

  private val phantomRefMap = new ConcurrentHashMap[NativeResourceRef, CPtrAddress]()

  private val cleanupThread = new ResourceCleanupThread()

  cleanupThread.start()

  def register(resource: NativeResource, resDeAllocAddr: CPtrAddress => Int):
  NativeResourceRef = {
    val resourceRef = new NativeResourceRef(resource, resDeAllocAddr)
    phantomRefMap.put(resourceRef, resource.nativeAddress)
    resourceRef
  }

  def deRegister(resourceRef: NativeResourceRef): Unit = {
    val resDeAllocAddr = phantomRefMap.get(resourceRef)
    if (resDeAllocAddr != null) {
      phantomRefMap.remove(resourceRef)
    }
  }

  def cleanup(): Unit = {
    print("NativeResourceRef: cleanup\n")
    // remove is a blocking call
    val ref: NativeResourceRef = referenceQueue.remove().asInstanceOf[NativeResourceRef]
    print("NativeResourceRef: got a reference with deAlloc\n")
    // phantomRef will be removed from the map when NativeResource.close is called.
    val resource = phantomRefMap.get(ref)

    if (resource != null)  {
      print("NativeResourceRef: got a reference for resource\n")
      ref.resDeAllocAddr(resource)
      phantomRefMap.remove(ref)
    }
  }

  private class ResourceCleanupThread extends Thread {
    setPriority(Thread.MAX_PRIORITY)
    setName("NativeResourceDeAllocatorThread")
    setDaemon(true)

    override def run(): Unit = {
      while (true) {
        try {
          cleanup()
        }
        catch {
          case _: InterruptedException => Thread.currentThread().interrupt()
        }
      }
    }
  }

}