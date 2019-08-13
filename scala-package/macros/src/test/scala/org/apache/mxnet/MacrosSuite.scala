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

import org.apache.mxnet.utils.CToScalaUtils
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

class MacrosSuite extends FunSuite with BeforeAndAfterAll {

  private val logger = LoggerFactory.getLogger(classOf[MacrosSuite])


  test("MacrosSuite-testArgumentCleaner") {
    val input = List(
      "Symbol, optional, default = Null",
      "int, required",
      "Shape(tuple), optional, default = []",
      "{'csr', 'default', 'row_sparse'}, optional, default = 'csr'",
      ", required"
    )
    val output = List(
      ("org.apache.mxnet.Symbol", true),
      ("Int", false),
      ("org.apache.mxnet.Shape", true),
      ("String", true),
      ("Any", false)
    )

    for (idx <- input.indices) {
      val result = CToScalaUtils.argumentCleaner("Sample", input(idx),
        "org.apache.mxnet.Symbol", false)
      assert(result._1 === output(idx)._1 && result._2 === output(idx)._2)
    }
  }

}
