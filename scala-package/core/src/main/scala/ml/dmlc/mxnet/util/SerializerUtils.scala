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

package ml.dmlc.mxnet.util

import java.io.{ObjectInputStream, ByteArrayInputStream, ObjectOutputStream, ByteArrayOutputStream}

import ml.dmlc.mxnet.NDArray

private[mxnet] object SerializerUtils {
  /**
   * Serialize NDArrays to bytes
   * @param arrays NDArrays to be serialized
   * @return serialized bytes
   */
  def serializeNDArrays(arrays: NDArray*): Array[Byte] = {
    val bos = new ByteArrayOutputStream()
    try {
      val out = new ObjectOutputStream(bos)
      out.writeInt(arrays.length)
      arrays.foreach(array => {
        val sArray = array.serialize()
        out.writeInt(sArray.length)
        out.write(sArray)
      })
      out.flush()
      bos.toByteArray
    } finally {
      try {
        bos.close()
      } catch {
        case _: Throwable =>
      }
    }
  }

  /**
   * Deserialize bytes to a list of NDArrays.
   * This should be used with SerializerUtils.serializeNDArrays
   * @param bytes serialized NDArray bytes
   * @return deserialized NDArrays
   */
  def deserializeNDArrays(bytes: Array[Byte]): IndexedSeq[NDArray] = {
    if (bytes != null) {
      val bis = new ByteArrayInputStream(bytes)
      var in: ObjectInputStream = null
      try {
        in = new ObjectInputStream(bis)
        val numArrays = in.readInt()
        (0 until numArrays).map(_ => {
          val len = in.readInt()
          val bytes = Array.fill[Byte](len)(0)
          in.readFully(bytes)
          NDArray.deserialize(bytes)
        })
      } finally {
        try {
          if (in != null) {
            in.close()
          }
        } catch {
          case _: Throwable =>
        }
      }
    } else {
      null
    }
  }
}
