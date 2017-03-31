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

import ml.dmlc.mxnet.NDArray
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SerializerUtilsSuite extends FunSuite with BeforeAndAfterAll {
  test("serialize & deserialize NDArrays") {
    val a = NDArray.zeros(2, 3)
    val b = NDArray.ones(3, 1)
    val bytes = SerializerUtils.serializeNDArrays(a, b)
    val ndArrays = SerializerUtils.deserializeNDArrays(bytes)
    assert(ndArrays.size === 2)
    assert(ndArrays(0) === a)
    assert(ndArrays(1) === b)
  }
}
