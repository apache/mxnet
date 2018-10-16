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
import scala.collection.mutable

object Model {
  def saveCheckpoint(prefix : String, epoch : Int, symbol : Symbol,
                     argParams : java.util.Map[String, NDArray], auxParams : java.util.Map[String, NDArray]) : Unit = {
    org.apache.mxnet.Model.saveCheckpoint(prefix, epoch, symbol,
      argParams.asScala.map(ele => (ele._1, NDArray.toNDArray(ele._2))).toMap,
      auxParams.asScala.map(ele => (ele._1, NDArray.toNDArray(ele._2))).toMap)
  }

  def loadCheckpoint(prefix : String, epoch : Int) : ModelContent = {
    val result = org.apache.mxnet.Model.loadCheckpoint(prefix, epoch)
    val JavaArgParams = mutable.Map[String, NDArray]()
    val JavaAuxParams = mutable.Map[String, NDArray]()
    result._2.foreach(ele => JavaArgParams(ele._1) = ele._2)
    result._3.foreach(ele => JavaAuxParams(ele._1) = ele._2)
    ModelContent(result._1, JavaArgParams.asJava, JavaAuxParams.asJava)
  }
  case class ModelContent(symbol : Symbol,
                          argParams : java.util.Map[String, NDArray],
                          auxParams : java.util.Map[String, NDArray])
}
