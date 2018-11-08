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

object DType extends Enumeration {
  type DType = Value
  val Float32 = Value(0, "float32")
  val Float64 = Value(1, "float64")
  val Float16 = Value(2, "float16")
  val UInt8 = Value(3, "uint8")
  val Int32 = Value(4, "int32")
  val Unknown = Value(-1, "unknown")
  private[mxnet] def numOfBytes(dtype: DType): Int = {
    dtype match {
      case DType.UInt8 => 1
      case DType.Int32 => 4
      case DType.Float16 => 2
      case DType.Float32 => 4
      case DType.Float64 => 8
      case DType.Unknown => 0
    }
  }
  private[mxnet] def getType(dtypeStr: String): DType = {
    dtypeStr match {
      case "UInt8" => DType.UInt8
      case "Int32" => DType.Int32
      case "Float16" => DType.Float16
      case "Float32" => DType.Float32
      case "Float64" => DType.Float64
      case _ => throw new IllegalArgumentException(
        s"DType: $dtypeStr not found! please set it in DType.scala")
    }
  }
}
