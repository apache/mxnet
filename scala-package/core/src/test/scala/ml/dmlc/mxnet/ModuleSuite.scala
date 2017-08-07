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
import ml.dmlc.mxnet.CheckUtils._
import ml.dmlc.mxnet.module._
import ml.dmlc.mxnet.optimizer._
import ml.dmlc.mxnet.io._

class ModuleSuite extends FunSuite with BeforeAndAfterAll {
  test ("model dtype") {
    val dType = DType.Float16
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

    val mod = new Module(c, IndexedSeq("b", "c", "a"), null,
      contexts = Array(Context.cpu(0), Context.cpu(1)))
    mod.bind(dataShapes = IndexedSeq(
      DataDesc("b", Shape(5, 5)),
      DataDesc("c", Shape(5, 5)),
      DataDesc("a", Shape(5, 5))),
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
    mod.bind(dataShapes = IndexedSeq(DataDesc("data", Shape(10, 10))))
    mod.initParams()
    mod.initOptimizer(optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f))
    mod.update()
    mod.saveCheckpoint("test", 0, saveOptStates = true)

    var mod2 = Module.loadCheckpoint("test", 0, loadOptimizerStates = true)
    mod2.bind(dataShapes = IndexedSeq(DataDesc("data", Shape(10, 10))))
    mod2.initOptimizer(optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f))
    assert(mod.getSymbol.toJson == mod2.getSymbol.toJson)
    mapEqu(mod.getParams._1, mod2.getParams._1)

    // multi device
    mod = new Module(sym, IndexedSeq("data"), null,
      contexts = Array(Context.cpu(0), Context.cpu(1)))
    mod.bind(dataShapes = IndexedSeq(DataDesc("data", Shape(10, 10))))
    mod.initParams()
    mod.initOptimizer(optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f))
    mod.update()
    mod.saveCheckpoint("test", 0, saveOptStates = true)

    mod2 = Module.loadCheckpoint("test", 0, loadOptimizerStates = true)
    mod2.bind(dataShapes = IndexedSeq(DataDesc("data", Shape(10, 10))))
    mod2.initOptimizer(optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f))
    assert(mod.getSymbol.toJson == mod2.getSymbol.toJson)
    mapEqu(mod.getParams._1, mod2.getParams._1)
  }

  test ("module reshape") {
    var sym = Symbol.Variable("data")
    sym = Symbol.FullyConnected("fc")()(Map("data" -> sym, "num_hidden" -> 20))

    var dShape = Shape(7, 20)
    val mod = new Module(sym, IndexedSeq("data"), null,
      contexts = Array(Context.cpu(0), Context.cpu(1)))
    mod.bind(dataShapes = IndexedSeq(DataDesc("data", dShape)))
    mod.initParams()
    mod.initOptimizer(optimizer = new SGD(learningRate = 1f))

    mod.forward(new DataBatch(
      data = IndexedSeq(NDArray.ones(dShape)),
      label = null, index = null, pad = 0))
    mod.backward(Array(NDArray.ones(dShape)))
    mod.update()
    assert(mod.getOutputsMerged()(0).shape == dShape)
    assert(mod.getParams._1("fc_bias").toArray.forall(_ == -1f))

    dShape = Shape(14, 20)
    mod.reshape(IndexedSeq(DataDesc("data", dShape)))
    mod.forward(new DataBatch(
      data = IndexedSeq(NDArray.ones(dShape)),
      label = null, index = null, pad = 0))
    mod.backward(Array(NDArray.ones(dShape)))
    mod.update()
    assert(mod.getOutputsMerged()(0).shape == dShape)
    assert(mod.getParams._1("fc_bias").toArray.forall(x => (x - -3f) < 1e-3))
  }

  test ("module setParams") {
    val data = NDArray.array(Array(0.05f, 0.1f), Shape(1, 2))
    val label = NDArray.array(Array(0.01f, 0.99f), Shape(1, 2))
    val trainData = new NDArrayIter(
      IndexedSeq(data), IndexedSeq(label), labelName = "softmax_label")

    // symbols
    var x = Symbol.Variable("data")
    x = Symbol.FullyConnected(name = "fc_0")()(Map("data" -> x, "num_hidden" -> 2))
    x = Symbol.Activation(name = "act_0")()(Map("data" -> x, "act_type" -> "sigmoid"))
    x = Symbol.FullyConnected(name = "fc_1")()(Map("data" -> x, "num_hidden" -> 2))
    x = Symbol.Activation(name = "act_1")()(Map("data" -> x, "act_type" -> "sigmoid"))
    x = Symbol.LinearRegressionOutput(name = "softmax")()(Map("data" -> x, "grad_scale" -> 2))

    // create module
    val mod = new Module(x, contexts = Array(Context.cpu()))
    mod.bind(dataShapes = trainData.provideData,
      Option(trainData.provideLabel))
    val argParamsCorrect = Map(
      "fc_0_weight" -> NDArray.array(Array(0.15f, 0.2f, 0.25f, 0.3f), Shape(2, 2)),
      "fc_0_bias" -> NDArray.array(Array(0.35f, 0.35f), Shape(2)),
      "fc_1_weight" -> NDArray.array(Array(0.4f, 0.45f, 0.5f, 0.55f), Shape(2, 2)),
      "fc_1_bias" -> NDArray.array(Array(0.6f, 0.6f), Shape(2))
    )
    val argParamsMissing = Map(
      "fc_0_weight" -> NDArray.array(Array(0.15f, 0.2f, 0.25f, 0.3f), Shape(2, 2)),
      "fc_0_bias" -> NDArray.array(Array(0.35f, 0.35f), Shape(2)),
      "fc_1_weight" -> NDArray.array(Array(0.4f, 0.45f, 0.5f, 0.55f), Shape(2, 2))
    )
    val argParamsExtra = Map(
      "fc_0_weight" -> NDArray.array(Array(0.15f, 0.2f, 0.25f, 0.3f), Shape(2, 2)),
      "fc_0_bias" -> NDArray.array(Array(0.35f, 0.35f), Shape(2)),
      "fc_1_weight" -> NDArray.array(Array(0.4f, 0.45f, 0.5f, 0.55f), Shape(2, 2)),
      "fc_1_bias" -> NDArray.array(Array(0.6f, 0.6f), Shape(2)),
      "fc_2_weight" -> NDArray.array(Array(0.6f, 0.6f), Shape(2))
    )

    mod.setParams(forceInit = true, argParams = argParamsCorrect,
      auxParams = null)

    // test allow missing
    mod.setParams(forceInit = true, argParams = argParamsMissing,
      auxParams = null, allowMissing = true)

    // test allow extra
    mod.setParams(forceInit = true, argParams = argParamsExtra, auxParams = null,
      allowMissing = true, allowExtra = true)
  }

  test ("monitor") {
    // data iter
    val data = NDArray.array(Array(0.05f, 0.1f), Shape(1, 2))
    val label = NDArray.array(Array(0.01f, 0.99f), Shape(1, 2))
    val trainData = new NDArrayIter(
      IndexedSeq(data), IndexedSeq(label), labelName = "softmax_label")

    // symbols
    var x = Symbol.Variable("data")
    x = Symbol.FullyConnected(name = "fc_0")()(Map("data" -> x, "num_hidden" -> 2))
    x = Symbol.Activation(name = "act_0")()(Map("data" -> x, "act_type" -> "sigmoid"))
    x = Symbol.FullyConnected(name = "fc_1")()(Map("data" -> x, "num_hidden" -> 2))
    x = Symbol.Activation(name = "act_1")()(Map("data" -> x, "act_type" -> "sigmoid"))
    x = Symbol.LinearRegressionOutput(name = "softmax")()(Map("data" -> x, "grad_scale" -> 2))

    // create monitor
    def meanAbs(x: NDArray): NDArray = {
      val sumAbs = NDArray.sum(NDArray.abs(x))
      sumAbs / x.shape.product
    }
    val mon = new Monitor(1, statFunc = meanAbs)

    // create module
    val mod = new Module(x, contexts = Array(Context.cpu()))
    mod.bind(dataShapes = trainData.provideData,
      Option(trainData.provideLabel))
    mod.installMonitor(mon)
    val argParams = Map(
      "fc_0_weight" -> NDArray.array(Array(0.15f, 0.2f, 0.25f, 0.3f), Shape(2, 2)),
      "fc_0_bias" -> NDArray.array(Array(0.35f, 0.35f), Shape(2)),
      "fc_1_weight" -> NDArray.array(Array(0.4f, 0.45f, 0.5f, 0.55f), Shape(2, 2)),
      "fc_1_bias" -> NDArray.array(Array(0.6f, 0.6f), Shape(2))
    )
    mod.initParams(argParams = argParams)

    val dataBatch = trainData.next()
    mon.tic()
    mod.forwardBackward(dataBatch)
    val res = mon.toc()
    val keys = Array("act_0", "act_1", "data", "fc_0", "fc_1", "softmax")
    val monResultCounts = Array(0, 0, 0, 0, 0, 0)
    assert(res.length == 21)
    for ((n, k, v) <- res) {
      var break = false
      for ((key, idx) <- keys.zipWithIndex) {
        if (!break && k.startsWith(key)) {
          monResultCounts(idx) += 1
          break = true
        }
      }
    }
    assert(monResultCounts.zip(Array(2, 2, 1, 6, 6, 4)).forall(x => x._1 == x._2))
  }

  test ("forward reshape") {
    val numClass = 10
    val data1 = Symbol.Variable("data1")
    val data2 = Symbol.Variable("data2")
    val conv1 = Symbol.Convolution()()(Map("data" -> data1,
        "kernel" -> "(2, 2)", "num_filter" -> 2, "stride" -> "(2, 2)"))
    val conv2 = Symbol.Convolution()()(Map("data" -> data2,
        "kernel" -> "(3, 3)", "num_filter" -> 3, "stride" -> "(1, 1)"))
    val pooling1 = Symbol.Pooling()()(Map("data" -> conv1,
        "kernel" -> "(2, 2)", "pool_type" -> "avg", "stride" -> "(1, 1)"))
    val pooling2 = Symbol.Pooling()()(Map("data" -> conv2,
        "kernel" -> "(2, 2)", "pool_type" -> "max", "stride" -> "(1, 1)"))
    val flatten1 = Symbol.flatten()()(Map("data" -> pooling1))
    val flatten2 = Symbol.flatten()()(Map("data" -> pooling2))
    val sum = Symbol.sum()()(Map("data" -> flatten1, "axis" -> 1)) +
      Symbol.sum()()(Map("data" -> flatten2, "axis" -> 1))
    val fc = Symbol.FullyConnected()()(
      Map("data" -> sum, "num_hidden" -> numClass))
    val sym = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc))

    var dShape1 = Shape(10, 3, 64, 64)
    var dShape2 = Shape(10, 3, 32, 32)
    var lShape = Shape(10)

    val mod = new Module(sym, IndexedSeq("data1", "data2"))
    mod.bind(dataShapes = IndexedSeq(
      DataDesc("data1", dShape1), DataDesc("data2", dShape2)),
      labelShapes = Option(IndexedSeq(DataDesc("softmax_label", lShape)))
    )
    mod.initParams()
    mod.initOptimizer(optimizer = new SGD(learningRate = 0.01f))

    // Train with original data shapes
    var dataBatch = new DataBatch(
      data = IndexedSeq(
        NDArray.random_uniform(Map("low" -> 0, "high" -> 9, "shape" -> dShape1.toString()))(),
        NDArray.random_uniform(Map("low" -> 5, "high" -> 15, "shape" -> dShape2.toString()))()),
      label = IndexedSeq(NDArray.ones(lShape)), index = null, pad = 0)
    mod.forward(dataBatch)
    assert(mod.getOutputsMerged()(0).shape == Shape(lShape(0), numClass))
    mod.backward()
    mod.update()

    dShape1 = Shape(3, 3, 64, 64)
    dShape2 = Shape(3, 3, 32, 32)
    lShape = Shape(3)
    dataBatch = new DataBatch(
      data = IndexedSeq(
        NDArray.random_uniform(Map("low" -> 0, "high" -> 9, "shape" -> dShape1.toString()))(),
        NDArray.random_uniform(Map("low" -> 5, "high" -> 15, "shape" -> dShape2.toString()))()),
      label = IndexedSeq(NDArray.ones(lShape)), index = null, pad = 0)
    mod.forward(dataBatch)
    assert(mod.getOutputsMerged()(0).shape == Shape(lShape(0), numClass))
    mod.backward()
    mod.update()

    dShape1 = Shape(20, 3, 64, 64)
    dShape2 = Shape(20, 3, 32, 32)
    lShape = Shape(20)
    dataBatch = new DataBatch(
      data = IndexedSeq(
        NDArray.random_uniform(Map("low" -> 3, "high" -> 5, "shape" -> dShape1.toString()))(),
        NDArray.random_uniform(Map("low" -> 10, "high" -> 25, "shape" -> dShape2.toString()))()),
      label = IndexedSeq(NDArray.ones(lShape)), index = null, pad = 0)
    mod.forward(dataBatch)
    assert(mod.getOutputsMerged()(0).shape == Shape(lShape(0), numClass))
    mod.backward()
    mod.update()

    // Train with both different batch size and data shapes
    dShape1 = Shape(20, 3, 120, 120)
    dShape2 = Shape(20, 3, 32, 64)
    lShape = Shape(20)
    dataBatch = new DataBatch(
      data = IndexedSeq(
        NDArray.random_uniform(Map("low" -> 0, "high" -> 9, "shape" -> dShape1.toString()))(),
        NDArray.random_uniform(Map("low" -> 5, "high" -> 15, "shape" -> dShape2.toString()))()),
      label = IndexedSeq(NDArray.ones(lShape)), index = null, pad = 0)
    mod.forward(dataBatch)
    assert(mod.getOutputsMerged()(0).shape == Shape(lShape(0), numClass))
    mod.backward()
    mod.update()

    dShape1 = Shape(5, 3, 28, 40)
    dShape2 = Shape(5, 3, 24, 16)
    lShape = Shape(5)
    dataBatch = new DataBatch(
      data = IndexedSeq(
        NDArray.random_uniform(Map("low" -> 0, "high" -> 9, "shape" -> dShape1.toString()))(),
        NDArray.random_uniform(Map("low" -> 15, "high" -> 25, "shape" -> dShape2.toString()))()),
      label = IndexedSeq(NDArray.ones(lShape)), index = null, pad = 0)
    mod.forward(dataBatch)
    assert(mod.getOutputsMerged()(0).shape == Shape(lShape(0), numClass))
    mod.backward()
    mod.update()
  }
}
