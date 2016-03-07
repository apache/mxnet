package ml.dmlc.mxnet

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class AttrScopeSuite extends FunSuite with BeforeAndAfterAll {
  test("attr basic") {
    val (data, gdata) =
    AttrScope(Map("group" -> "4", "data" -> "great")).withScope {
      val data = Symbol.Variable("data", attr = Map("dtype" -> "data", "group" -> "1"))
      val gdata = Symbol.Variable("data2")
      (data, gdata)
    }
    assert(gdata.attr("group").get === "4")
    assert(data.attr("group").get === "1")

    val exceedScopeData = Symbol.Variable("data3")
    assert(exceedScopeData.attr("group") === None, "No group attr in global attr scope")
  }
}
