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
package org.apache.mxnet.javaapi

import collection.JavaConverters._
import scala.language.implicitConversions

/**
  * Constructing a context which is used to specify the device and device type that will
  * be utilized by the engine.
  *
  * @param deviceTypeName {'cpu', 'gpu'} String representing the device type
  * @param deviceId The device id of the device, needed for GPU
  */
class Context private[mxnet] (val context: org.apache.mxnet.Context) {

  val deviceTypeid: Int = context.deviceTypeid

  def this(deviceTypeName: String, deviceId: Int = 0)
  = this(new org.apache.mxnet.Context(deviceTypeName, deviceId))

  def withScope[T](body: => T): T = context.withScope(body)

  /**
    * Return device type of current context.
    * @return device_type
    */
  def deviceType: String = context.deviceType

  override def toString: String = context.toString
  override def equals(other: Any): Boolean = context.equals(other)
  override def hashCode: Int = context.hashCode
}


object Context {
  implicit def fromContext(context: org.apache.mxnet.Context): Context = new Context(context)
  implicit def toContext(jContext: Context): org.apache.mxnet.Context = jContext.context

  val cpu: Context = org.apache.mxnet.Context.cpu()
  val gpu: Context = org.apache.mxnet.Context.gpu()
  val devtype2str = org.apache.mxnet.Context.devstr2type.asJava
  val devstr2type = org.apache.mxnet.Context.devstr2type.asJava
  def defaultCtx: Context = org.apache.mxnet.Context.defaultCtx
}
