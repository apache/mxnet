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

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SymbolSuite extends FunSuite with BeforeAndAfterAll {
  test("symbol compose") {
    val data = Symbol.Variable("data")

    var net1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> 10))
    net1 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> net1, "num_hidden" -> 100))
    assert(net1.listArguments().toArray ===
      Array("data", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"))

    var net2 = Symbol.FullyConnected(name = "fc3")()(Map("num_hidden" -> 10))
    net2 = Symbol.Activation()()(Map("data" -> net2, "act_type" -> "relu"))
    net2 = Symbol.FullyConnected(name = "fc4")()(Map("data" -> net2, "num_hidden" -> 20))
    // scalastyle:off println
    println(s"net2 debug info:\n${net2.debugStr}")
    // scalastyle:on println

    val composed = net2(name = "composed", Map("fc3_data" -> net1))
    // scalastyle:off println
    println(s"composed debug info:\n${composed.debugStr}")
    // scalastyle:on println
    val multiOut = Symbol.Group(composed, net1)
    assert(multiOut.listOutputs().length === 2)
  }

  test("symbol internal") {
    val data = Symbol.Variable("data")
    val oldfc = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> 10))
    val net1 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> oldfc, "num_hidden" -> 100))
    assert(net1.listArguments().toArray
      === Array("data", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"))
    val internal = net1.getInternals()
    val fc1 = internal.get("fc1_output")
    assert(fc1.listArguments() === oldfc.listArguments())
  }

  test("symbol infer type") {
    val data = Symbol.Variable("data")
    val f32data = Symbol.Cast()()(Map("data" -> data, "dtype" -> "float32"))
    val fc1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> f32data, "num_hidden" -> 128))
    val mlp = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc1))

    val (arg, out, aux) = mlp.inferType(Map("data" -> DType.Float64))
    assert(arg.toArray === Array(DType.Float64, DType.Float32, DType.Float32, DType.Float32))
    assert(out.toArray === Array(DType.Float32))
    assert(aux.isEmpty)
  }

  test("symbol copy") {
    val data = Symbol.Variable("data")
    val data2 = data.clone()
    assert(data.toJson === data2.toJson)
  }
}
