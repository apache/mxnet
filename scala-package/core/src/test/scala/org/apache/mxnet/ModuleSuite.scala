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

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.apache.mxnet.module._
import org.apache.mxnet.optimizer._
import org.apache.mxnet.io._

class ModuleSuite extends FunSuite with BeforeAndAfterAll {

  class myModule(symbol : Symbol) extends Module (symbol) {
    override def predictEveryBatch(evalData: DataIter,
                                   numBatch: Int = 1, reset: Boolean = true):
    IndexedSeq[IndexedSeq[NDArray]] = {
      val data = IndexedSeq(
        NDArray.ones(Shape(1, 10, 1)),
        NDArray.ones(Shape(1, 10, 1)),
        NDArray.ones(Shape(1, 10, 4))
      )
      List.fill(numBatch)(data).toIndexedSeq
    }
  }

  test("predict") {
    val sym = Symbol.Variable("data")
    val mod = new myModule(sym)
    val dummyIter = new NDArrayIter(IndexedSeq(NDArray.ones(1)))
    var output = mod.predict(dummyIter, 1)
    require(output(0).shape == Shape(1, 10, 1))
    require(output(1).shape == Shape(1, 10, 1))
    require(output(2).shape == Shape(1, 10, 4))
    output = mod.predict(dummyIter, 2)
    require(output(0).shape == Shape(2, 10, 1))
    require(output(1).shape == Shape(2, 10, 1))
    require(output(2).shape == Shape(2, 10, 4))
  }

  test ("model dtype") {
    val dType = DType.Float32
    val dShape = Shape(3, 8, 7)

    var sym = Symbol.Variable("data")
    sym = Symbol.Activation(attr = Map("__layout__" -> "TNC"))()(
      Map("data" -> sym, "act_type" -> "relu"))

    val mod = new Module(sym, IndexedSeq("data"), null,
      contexts = Array(Context.cpu(0), Context.cpu(1)))
    mod.bind(dataShapes = IndexedSeq(DataDesc("data", dShape, dType, "TNC")))
    mod.initParams()
    mod.forward(new DataBatch(
      data = IndexedSeq(NDArray.ones(dShape, dtype = dType)),
      label = null, index = null, pad = 0))
    mod.backward(Array(NDArray.ones(dShape, dtype = dType)))

    assert(mod.getOutputs.flatten.forall(_.dtype == dType))
  }

  test ("module input_grads") {
    val a = Symbol.Variable("a", kwargs = Map("__layout__" -> "NC"))
    val b = Symbol.Variable("b", kwargs = Map("__layout__" -> "NC"))
    var c = Symbol.Variable("c", kwargs = Map("__layout__" -> "NC"))

    import SymbolConversions._
    c = a + 2 * b + 3 * c

    val mod = new Module.Builder(c)
      .setDataNames("b", "c", "a")
      .setLabelNames(null)
      .setContext(Context.cpu(0), Context.cpu(1))
      .build()
    mod.bind(dataShapes = IndexedSeq(
      DataDesc("b", Shape(5, 5), layout = "NT"),
      DataDesc("c", Shape(5, 5), layout = "NT"),
      DataDesc("a", Shape(5, 5), layout = "NT")),
      inputsNeedGrad = true
    )
    mod.initParams()
    mod.forward(new DataBatch(
      data = IndexedSeq(
        NDArray.ones(5, 5), NDArray.ones(5, 5), NDArray.ones(5, 5)),
      label = null, index = null, pad = 0))
    mod.backward(Array(NDArray.ones(5, 5)))

    val inputGrads = mod.getInputGradsMerged()
    val aGrad = inputGrads(0).toArray
    val bGrad = inputGrads(1).toArray
    val cGrad = inputGrads(2).toArray

    assert(aGrad.forall(_ == 1f))
    assert(bGrad.forall(_ == 2f))
    assert(cGrad.forall(_ == 3f))
  }

  test ("module layout") {
    var sym = Symbol.Variable("data")
    sym = Symbol.Activation(attr = Map("__layout__" -> "TNC"))()(
      Map("data" -> sym, "act_type" -> "relu"))

    val dShape = Shape(3, 8, 7)
    val mod = new Module(sym, IndexedSeq("data"), null,
      contexts = Array(Context.cpu(0), Context.cpu(1)))
    mod.bind(dataShapes = IndexedSeq(DataDesc("data", dShape, layout = "TNC")))
    mod.initParams()
    mod.forward(new DataBatch(
      data = IndexedSeq(NDArray.ones(dShape)),
      label = null, index = null, pad = 0))
    mod.backward(Array(NDArray.ones(dShape)))
    assert(mod.getOutputsMerged()(0).shape == dShape)

    val hdShape = Shape(3, 4, 7)
    for (x <- mod.getOutputs) assert(x(0).shape == hdShape)
  }

  test ("save load") {
    def mapEqu(a: Map[String, NDArray], b: Map[String, NDArray]): Unit = {
      assert(a.toSet == b.toSet)
      for (k <- a.keys) assert(a(k) == b(k))
    }

    var sym = Symbol.Variable("data")
    sym = Symbol.FullyConnected()()(Map("data" -> sym, "num_hidden" -> 100))

    // single device
    var mod = new Module(sym, IndexedSeq("data"), null)
    mod.bind(dataShapes = IndexedSeq(DataDesc("data", Shape(10, 10), layout = "NT")))
    mod.initParams()
    mod.initOptimizer(optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f))
    mod.update()
    mod.saveCheckpoint("test", 0, saveOptStates = true)

    var mod2 = Module.loadCheckpoint("test", 0, loadOptimizerStates = true)
    mod2.bind(dataShapes = IndexedSeq(DataDesc("data", Shape(10, 10), layout = "NT")))
    mod2.initOptimizer(optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f))
    assert(mod.getSymbol.toJson == mod2.getSymbol.toJson)
    mapEqu(mod.getParams._1, mod2.getParams._1)

    // multi device
    mod = new Module(sym, IndexedSeq("data"), null,
      contexts = Array(Context.cpu(0), Context.cpu(1)))
    mod.bind(dataShapes = IndexedSeq(DataDesc("data", Shape(10, 10), layout = "NT" )))
    mod.initParams()
    mod.initOptimizer(optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f))
    mod.update()
    mod.saveCheckpoint("test", 0, saveOptStates = true)

    mod2 = Module.loadCheckpoint("test", 0, loadOptimizerStates = true)
    mod2.bind(dataShapes = IndexedSeq(DataDesc("data", Shape(10, 10), layout = "NT")))
    mod2.initOptimizer(optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f))
    assert(mod.getSymbol.toJson == mod2.getSymbol.toJson)
    mapEqu(mod.getParams._1, mod2.getParams._1)
  }

  test ("module reshape") {
    CancelTestUtil.assumeStandardDecimalSeparator()

    var sym = Symbol.Variable("data")
    sym = Symbol.FullyConnected("fc")()(Map("data" -> sym, "num_hidden" -> 20))

    var dShape = Shape(7, 20)
    val mod = new Module(sym, IndexedSeq("data"), null,
      contexts = Array(Context.cpu(0), Context.cpu(1)))
    mod.bind(dataShapes = IndexedSeq(DataDesc("data", dShape, layout = "NT")))
    mod.initParams()
    mod.initOptimizer(optimizer = new SGD(learningRate = 1f))

    mod.forward(new DataBatch(
      data = IndexedSeq(NDArray.ones(dShape)),
      label = null, index = null, pad = 0))
    mod.backward(Array(NDArray.ones(dShape)))
    mod.update()
    assert(mod.getOutputsMerged()(0).shape == dShape)
    assert(mod.getParams._1("fc_bias").toArray.forall(_ == -1f))

    // reshape module
    dShape = Shape(14, 20)
    mod.reshape(IndexedSeq(DataDesc("data", dShape, layout = "NT")))
    mod.forward(new DataBatch(
      data = IndexedSeq(NDArray.ones(dShape)),
      label = null, index = null, pad = 0))
    mod.backward(Array(NDArray.ones(dShape)))
    mod.update()
    assert(mod.getOutputsMerged()(0).shape == dShape)
    assert(mod.getParams._1("fc_bias").toArray.forall(x => (x - -3f) < 1e-3))

    // return to original binded shape
    dShape = Shape(7, 20)
    mod.reshape(IndexedSeq(DataDesc("data", dShape, layout = "NT")))
    mod.forward(new DataBatch(
      data = IndexedSeq(NDArray.ones(dShape)),
      label = null, index = null, pad = 0))
    mod.backward(Array(NDArray.ones(dShape)))
    mod.update()
    assert(mod.getOutputsMerged()(0).shape == dShape)
    assert(mod.getParams._1("fc_bias").toArray.forall(x => (x - -3f) < 1e-3))
  }
}
