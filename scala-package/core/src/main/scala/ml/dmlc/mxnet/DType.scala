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

object DType extends Enumeration {
  type DType = Value
  val Float32 = Value(0, "float32")
  val Float64 = Value(1, "float64")
  val Float16 = Value(2, "float16")
  val UInt8 = Value(3, "uint8")
  val Int32 = Value(4, "int32")
  private[mxnet] def numOfBytes(dtype: DType): Int = {
    dtype match {
      case DType.UInt8 => 1
      case DType.Int32 => 4
      case DType.Float16 => 2
      case DType.Float32 => 4
      case DType.Float64 => 8
    }
  }
}
