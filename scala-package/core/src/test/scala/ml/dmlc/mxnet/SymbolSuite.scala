package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class SymbolSuite extends FunSuite with BeforeAndAfterAll {
  test("plus") {
    val sym1 = Symbol.Variable("data1")
    val sym2 = Symbol.Variable("data2")
    val symPlus = sym1 + sym2
    // TODO: check result
  }
}
