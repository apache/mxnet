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

package org.apache.mxnet.api.java

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import collection.JavaConverters._

/**
  * This arg Builder is intent to solve Java to Scala conversion
  * to take the input such as (arg: Any*)
  */
class ArgBuilder {
  private var data = ListBuffer[Any]()
  private var map = mutable.Map[String, Any]()

  def addArg(anyRef: AnyRef): ArgBuilder = {
    require(map.isEmpty,
      "Map is not empty, please do either key-value or positional-arg but not both")
    this.data += anyRef.asInstanceOf[Any]
    this
  }

  def addArg(key : String, value : AnyRef) : ArgBuilder = {
    require(data.isEmpty,
      "Data is not empty, please do either key-value or positional-arg but not both")
    this.map(key) = value.asInstanceOf[Any]
    this
  }

  def addBatchArgs(list : java.util.List[AnyRef]) : ArgBuilder = {
    require(map.isEmpty,
      "Map is not empty, please do either key-value or positional-arg but not both")
    for (i <- 0 to list.size()) {
      this.data += list.get(i)
    }
    this
  }

  def addBatchArgs(arr : Array[AnyRef]) : ArgBuilder = {
    require(map.isEmpty,
      "Map is not empty, please do either key-value or positional-arg but not both")
    arr.foreach(ele => this.data += ele)
    this
  }

  def buildMap() : Map[String, Any] = {
    this.map.toMap
  }

  def buildSeq() : Seq[Any] = {
    this.data
  }
}
