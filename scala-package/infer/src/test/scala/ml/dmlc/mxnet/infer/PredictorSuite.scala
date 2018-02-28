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

package ml.dmlc.mxnet.infer


import ml.dmlc.mxnet.{DataDesc, Shape}
import ml.dmlc.mxnet.module.{BaseModule, Module}
import org.scalatest.{BeforeAndAfterAll, FunSuite, Ignore}
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.io.NDArrayIter
import org.mockito.Matchers._
import org.mockito.Mockito

class PredictorSuite extends FunSuite with BeforeAndAfterAll {

  class MyPredictor(val modelPathPrefix: String,
                    override val inputDescriptors: IndexedSeq[DataDesc],
                    override val outputDescriptors: Option[IndexedSeq[DataDesc]] = None)
    extends Predictor(modelPathPrefix, inputDescriptors, outputDescriptors) {

    override def loadModule(): Module = MyPredictor.mockModule

    val getIDescriptor: IndexedSeq[DataDesc] = iDescriptors
    val getBatchSize: Int = batchSize
    val getBatchIndex: Int = batchIndex
  }

  object MyPredictor {
    val mockModule: Module = Mockito.mock(classOf[Module])
  }

  test("testPredictorConstruction") {
    // BatchIndex missing
    // BatchIndex is different for different inputs
    // OutputDescriptor is bound to the Module
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(1, 3, 2, 2)))

    val mockPredictor = new MyPredictor("xyz", inputDescriptor)

    assert(mockPredictor.getBatchSize == 1)
    assert(mockPredictor.getBatchIndex == inputDescriptor(0).layout.indexOf('N'))

    val inputDescriptor2 = IndexedSeq[DataDesc](new DataDesc("data", Shape(1, 3, 2, 2)),
      new DataDesc("data", Shape(2, 3, 2, 2)))

    assertThrows[IllegalArgumentException] {
      val mockPredictor = new MyPredictor("xyz", inputDescriptor2)
    }

    //batchsize is
    val iDesc2 = IndexedSeq[DataDesc](new DataDesc("data", Shape(3, 2, 2), layout = "CHW"))
    val p2 = new MyPredictor("xyz", inputDescriptor)

  }

  test("testWithFlatArrays") {

    val mockSymbol = Mockito.mock(classOf[Symbol])
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(1, 3, 2, 2)))

    val inputData = Array.fill[Float](12)(1)
    val inputNDIter = new NDArrayIter(IndexedSeq(NDArray.array(inputData,
      inputDescriptor(0).shape)))

    val predictResult = IndexedSeq(NDArray.ones(Shape(1, 3, 2, 2)))

    val testPredictor = new MyPredictor("xyz", inputDescriptor)

    Mockito.doReturn(predictResult).when(MyPredictor.mockModule).predict(any(classOf[NDArrayIter]),
      any[Int], any[Boolean])

    //    Mockito.doReturn(Unit).when(MyPredictor.mockModule).bind(any(classOf[IndexedSeq[DataDesc]]),
    //      any(classOf[Option[IndexedSeq[DataDesc]]]), any(classOf[Boolean]),
    //      any(classOf[Boolean]), any(classOf[Boolean]), any(classOf[Option[BaseModule]]),
    //      any(classOf[String]))

    val testFun = testPredictor.predict(IndexedSeq(inputData))

  }

  test("testWithNDArray") {

  }
}