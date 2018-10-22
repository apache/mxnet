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

import java.util.HashSet

import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Try
import scala.util.control.{ControlThrowable, NonFatal}

/**
  * This class manages automatically releasing of [[NativeResource]]s
  */
class ResourceScope extends AutoCloseable {

  // HashSet does not take a custom comparator
  private[mxnet] val resourceQ = new mutable.TreeSet[NativeResource]()(nativeAddressOrdering)

  private object nativeAddressOrdering extends Ordering[NativeResource] {
    def compare(a: NativeResource, b: NativeResource): Int = {
      a.nativeAddress compare  b.nativeAddress
    }
  }

  ResourceScope.addToThreadLocal(this)

  /**
    * Releases all the [[NativeResource]] by calling
    * the associated [[NativeResource.close()]] method
    */
  override def close(): Unit = {
    ResourceScope.removeFromThreadLocal(this)
    resourceQ.foreach(resource => if (resource != null) resource.dispose(false) )
    resourceQ.clear()
  }

  /**
    * Add a NativeResource to the scope
    * @param resource
    */
  def add(resource: NativeResource): Unit = {
    resourceQ.+=(resource)
  }

  /**
    * Remove NativeResource from the Scope, this uses
    * object equality to find the resource in the stack.
    * @param resource
    */
  def remove(resource: NativeResource): Unit = {
    resourceQ.-=(resource)
  }
}

object ResourceScope {

  private val logger = LoggerFactory.getLogger(classOf[ResourceScope])

  /**
    * Captures all Native Resources created using the ResourceScope and
    * at the end of the body, de allocates all the Native resources by calling close on them.
    * This method will not deAllocate NativeResources returned from the block.
    * @param scope (Optional). Scope in which to capture the native resources
    * @param body  block of code to execute in this scope
    * @tparam A return type
    * @return result of the operation, if the result is of type NativeResource, it is not
    *         de allocated so the user can use it and then de allocate manually by calling
    *         close or enclose in another resourceScope.
    */
  // inspired from slide 21 of https://www.slideshare.net/Odersky/fosdem-2009-1013261
  // and https://github.com/scala/scala/blob/2.13.x/src/library/scala/util/Using.scala
  // TODO: we should move to the Scala util's Using method when we move to Scala 2.13
  def using[A](scope: ResourceScope = null)(body: => A): A = {

    val curScope = if (scope != null) scope else new ResourceScope()

    val prevScope: Option[ResourceScope] = ResourceScope.getPrevScope()

    @inline def resourceInGeneric(g: scala.collection.Iterable[_]) = {
      g.foreach( n =>
        n match {
          case nRes: NativeResource => {
            removeAndAddToPrevScope(nRes)
          }
          case kv: scala.Tuple2[_, _] => {
            if (kv._1.isInstanceOf[NativeResource]) removeAndAddToPrevScope(
              kv._1.asInstanceOf[NativeResource])
            if (kv._2.isInstanceOf[NativeResource]) removeAndAddToPrevScope(
              kv._2.asInstanceOf[NativeResource])
          }
        }
      )
    }

    @inline def removeAndAddToPrevScope(r: NativeResource) = {
      curScope.remove(r)
      if (prevScope.isDefined)  {
        prevScope.get.add(r)
        r.scope = prevScope
      }
    }

    @inline def safeAddSuppressed(t: Throwable, suppressed: Throwable): Unit = {
      if (!t.isInstanceOf[ControlThrowable]) t.addSuppressed(suppressed)
    }

    var retThrowable: Throwable = null

    try {
      val ret = body
       ret match {
          // don't de-allocate if returning any collection that contains NativeResource.
        case resInGeneric: scala.collection.Iterable[_] => resourceInGeneric(resInGeneric)
        case nRes: NativeResource => removeAndAddToPrevScope(nRes)
        case ndRet: NDArrayFuncReturn => ndRet.arr.foreach( nd => removeAndAddToPrevScope(nd) )
        case _ => // do nothing
      }
      ret
    } catch {
      case t: Throwable =>
        retThrowable = t
        null.asInstanceOf[A] // we'll throw in finally
    } finally {
      var toThrow: Throwable = retThrowable
      if (retThrowable eq null) curScope.close()
      else {
        try {
          curScope.close
        } catch {
          case closeThrowable: Throwable =>
            if (NonFatal(retThrowable) && !NonFatal(closeThrowable)) toThrow = closeThrowable
            else safeAddSuppressed(retThrowable, closeThrowable)
        } finally {
          throw toThrow
        }
      }
    }
  }

  // thread local Scopes
  private[mxnet] val threadLocalScopes = new ThreadLocal[ArrayBuffer[ResourceScope]] {
    override def initialValue(): ArrayBuffer[ResourceScope] =
      new ArrayBuffer[ResourceScope]()
  }

  /**
    * Add resource to current ThreadLocal DataStructure
    * @param r ResourceScope to add.
    */
  private[mxnet] def addToThreadLocal(r: ResourceScope): Unit = {
    threadLocalScopes.get() += r
  }

  /**
    * Remove resource from current ThreadLocal DataStructure
    * @param r ResourceScope to remove
    */
  private[mxnet] def removeFromThreadLocal(r: ResourceScope): Unit = {
    threadLocalScopes.get() -= r
  }

  /**
    * Get the latest Scope in the stack
    * @return
    */
  private[mxnet] def getCurrentScope(): Option[ResourceScope] = {
    Try(Some(threadLocalScopes.get().last)).getOrElse(None)
  }

  /**
    * Get the Last but one Scope from threadLocal Scopes.
    * @return n-1th scope or None when not found
    */
  private[mxnet] def getPrevScope(): Option[ResourceScope] = {
    val scopes = threadLocalScopes.get()
    Try(Some(scopes(scopes.size - 2))).getOrElse(None)
  }
}
