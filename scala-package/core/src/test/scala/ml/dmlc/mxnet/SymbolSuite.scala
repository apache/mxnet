package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SymbolSuite extends FunSuite with BeforeAndAfterAll {
  test("symbol compose") {
    val data = Symbol.Variable("data")

    var net1 = Symbol.FullyConnected(name = "fc1")(Map("data" -> data, "num_hidden" -> 10))
    net1 = Symbol.FullyConnected(name = "fc2")(Map("data" -> net1, "num_hidden" -> 100))
    assert(net1.listArguments().toArray ===
      Array("data", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"))

    var net2 = Symbol.FullyConnected(name = "fc3")(Map("num_hidden" -> 10))
    net2 = Symbol.Activation()(Map("data" -> net2, "act_type" -> "relu"))
    net2 = Symbol.FullyConnected(name = "fc4")(Map("data" -> net2, "num_hidden" -> 20))
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
    val oldfc = Symbol.FullyConnected(name = "fc1")(Map("data" -> data, "num_hidden" -> 10))
    val net1 = Symbol.FullyConnected(name = "fc2")(Map("data" -> oldfc, "num_hidden" -> 100))
    assert(net1.listArguments().toArray
      === Array("data", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"))
    val internal = net1.getInternals()
    val fc1 = internal.get("fc1_output")
    assert(fc1.listArguments() === oldfc.listArguments())
  }

  test("symbol infer type") {
    val data = Symbol.Variable("data")
    val f32data = Symbol.Cast()(Map("data" -> data, "dtype" -> "float32"))
    val fc1 = Symbol.FullyConnected(name = "fc1")(Map("data" -> f32data, "num_hidden" -> 128))
    val mlp = Symbol.SoftmaxOutput(name = "softmax")(Map("data" -> fc1))

    val (arg, out, aux) = mlp.inferType(Map("data" -> classOf[Double]))
    assert(arg.toArray === Array(classOf[Double], classOf[Float], classOf[Float], classOf[Float]))
    assert(out.toArray === Array(classOf[Float]))
    assert(aux.isEmpty)
  }

  test("symbol copy") {
    val data = Symbol.Variable("data")
    val data2 = data.clone()
    assert(data.toJson === data2.toJson)
  }
}
