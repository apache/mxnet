package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}


class IOSuite extends FunSuite with BeforeAndAfterAll {
  test("create iter funcs") {
    val iterCreateFuncs: Map[String, IO.IterCreateFunc] = IO._initIOModule()
    println(iterCreateFuncs.keys.toList)
  }
}
