package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SymbolSuite extends FunSuite with BeforeAndAfterAll {
  test("symbol compose") {
    val data = Symbol.Variable("data")

    var net1 = Symbol.FullyConnected(Map("data" -> data, "name" -> "fc1", "num_hidden" -> 10))
    net1 = Symbol.FullyConnected(Map("data" -> net1, "name" -> "fc2", "num_hidden" -> 100))
    assert(net1.listArguments() ===
      Array("data", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"))

    var net2 = Symbol.FullyConnected(Map("name" -> "fc3", "num_hidden" -> 10))
    net2 = Symbol.Activation(Map("data" -> net2, "act_type" -> "relu"))
    net2 = Symbol.FullyConnected(Map("data" -> net2, "name" -> "fc4", "num_hidden" -> 20))
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
}
