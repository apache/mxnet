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

import scala.collection.mutable


/**
  * typesafe Symbol API: Symbol.api._
  * Main code will be generated during compile time through Macros
  */
@AddSymbolAPIs(false)
object SymbolAPI extends SymbolAPIBase {
  def Custom (op_type : String, kwargs : mutable.Map[String, Any],
             name : String = null, attr : Map[String, String] = null) : Symbol = {
    val map = kwargs
    map.put("op_type", op_type)
    Symbol.createSymbolGeneral("Custom", name, attr, Seq(), map.toMap)
  }
}

/**
  * typesafe Symbol random module: Symbol.random._
  * Main code will be generated during compile time through Macros
  */
@AddSymbolRandomAPIs(false)
object SymbolRandomAPI extends SymbolRandomAPIBase {

}

