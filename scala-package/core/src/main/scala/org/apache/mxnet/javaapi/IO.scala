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
import scala.language.implicitConversions

class DataDesc private[mxnet] (val dataDesc: org.apache.mxnet.DataDesc) {

  def this(name: String, shape: Shape, dType: DType.DType, layout: String) =
    this(new org.apache.mxnet.DataDesc(name, shape, dType, layout))

  override def toString(): String = dataDesc.toString()
}

object DataDesc{
  implicit def fromDataDesc(dDesc: org.apache.mxnet.DataDesc): DataDesc = new DataDesc(dDesc)

  implicit def toDataDesc(dataDesc: DataDesc): org.apache.mxnet.DataDesc = dataDesc.dataDesc

  /**
    * Get the dimension that corresponds to the batch size.
    * @param layout layout string. For example, "NCHW".
    * @return An axis indicating the batch_size dimension. When data-parallelism is used,
    *         the data will be automatically split and concatenate along the batch_size dimension.
    *         Axis can be -1, which means the whole array will be copied
    *         for each data-parallelism device.
    */
  def getBatchAxis(layout: String): Int = org.apache.mxnet.DataDesc.getBatchAxis(Some(layout))
}
