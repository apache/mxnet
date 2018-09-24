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

import org.slf4j.LoggerFactory
import scala.collection.mutable.ArrayBuffer

class ResourceScope extends AutoCloseable {
  import ResourceScope.{logger, resourceScope}

  private val resourceQ = new  ArrayBuffer[NativeResource]()
  resourceScope.get().+=(this)

  override def close(): Unit = {
    resourceQ.foreach(resource => if (resource != null) {
      logger.info("releasing resource:%x\n".format(resource.nativeAddress))
      resource.dispose(false)
    } else {logger.info("found resource which is null")}
    )
    ResourceScope.resourceScope.get().-=(this)
  }

  private[mxnet] def register(resource: NativeResource): Unit = {
    logger.info("ResourceScope: Registering Resource %x".format(resource.nativeAddress))
    resourceQ.+=(resource)
  }

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