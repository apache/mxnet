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

package org.apache.mxnet.init

import org.apache.mxnet.init.Base._

import scala.collection.mutable.ListBuffer

class LibInfo {
  /**
    * Get the list of the symbol ids
    * @param symbolList Pass in an empty ListBuffer and obtain a list of operator IDs
    * @return Callback result
    */
  @native def mxSymbolListAtomicSymbolCreators(symbolList: ListBuffer[SymbolHandle]): Int

  /**
    * Get the detailed information of an operator
    * @param handle The ID of the operator
    * @param name Name of the operator
    * @param desc Description of the operator
    * @param numArgs Number of arguments
    * @param argNames Argument names
    * @param argTypes Argument types
    * @param argDescs Argument descriptions
    * @param keyVarNumArgs Kwargs number
    * @return Callback result
    */
  @native def mxSymbolGetAtomicSymbolInfo(handle: SymbolHandle,
                                          name: RefString,
                                          desc: RefString,
                                          numArgs: RefInt,
                                          argNames: ListBuffer[String],
                                          argTypes: ListBuffer[String],
                                          argDescs: ListBuffer[String],
                                          keyVarNumArgs: RefString): Int
  /**
    * Get the name list of all operators
    * @param names Names of all operator
    * @return Callback result
    */
  @native def mxListAllOpNames(names: ListBuffer[String]): Int

  /**
    * Get operator ID from its name
    * @param opName Operator name
    * @param opHandle Operator ID
    * @return Callback result
    */
  @native def nnGetOpHandle(opName: String, opHandle: RefLong): Int
}
