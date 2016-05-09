package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class ShapeSuite extends FunSuite with BeforeAndAfterAll {
  test("to string") {
    val s = Shape(1, 2, 3)
    assert(s.toString === "(1,2,3)")
  }

  test("equals") {
    assert(Shape(1, 2, 3) === Shape(1, 2, 3))
    assert(Shape(1, 2) != Shape(1, 2, 3))
  }
}
