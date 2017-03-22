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

package ml.dmlc.mxnet

object Context {
  val devtype2str = Map(1 -> "cpu", 2 -> "gpu", 3 -> "cpu_pinned")
  val devstr2type = Map("cpu" -> 1, "gpu" -> 2, "cpu_pinned" -> 3)
  private var _defaultCtx = new Context("cpu", 0)

  def defaultCtx: Context = _defaultCtx

  def cpu(deviceId: Int = 0): Context = {
    new Context("cpu", deviceId)
  }

  def gpu(deviceId: Int = 0): Context = {
    new Context("gpu", deviceId)
  }

  implicit def ctx2Array(ctx: Context): Array[Context] = Array(ctx)
}

/**
 * Constructing a context.

 * @param deviceTypeName {'cpu', 'gpu'} String representing the device type
 * @param deviceId (default=0) The device id of the device, needed for GPU
 */
class Context(deviceTypeName: String, val deviceId: Int = 0) extends Serializable {
  val deviceTypeid: Int = Context.devstr2type(deviceTypeName)

  def this(context: Context) = {
    this(context.deviceType, context.deviceId)
  }

  def withScope[T](body: => T): T = {
    val oldDefaultCtx = Context.defaultCtx
    Context._defaultCtx = this
    try {
      body
    } finally {
      Context._defaultCtx = oldDefaultCtx
    }
  }

  /**
   * Return device type of current context.
   * @return device_type
   */
  def deviceType: String = Context.devtype2str(deviceTypeid)

  override def toString: String = {
    s"$deviceType($deviceId)"
  }

  override def equals(other: Any): Boolean = {
    if (other != null && other.isInstanceOf[Context]) {
      val otherInst = other.asInstanceOf[Context]
      otherInst.deviceId == deviceId && otherInst.deviceTypeid == deviceTypeid
    } else {
      false
    }
  }

  override def hashCode: Int = {
    toString.hashCode
  }
}
