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

package org.apache.mxnet.infer

import org.apache.mxnet.io.NDArrayIter
import org.apache.mxnet.module.{BaseModule, Module}
import org.apache.mxnet._
import org.mockito.Matchers._
import org.mockito.Mockito
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class PredictorSuite extends FunSuite with BeforeAndAfterAll {

  class MyPredictor(val modelPathPrefix: String,
                    override val inputDescriptors: IndexedSeq[DataDesc])
    extends Predictor(modelPathPrefix, inputDescriptors, epoch = Some(0)) {

    override def loadModule(): Module = mockModule

    val getIDescriptor: IndexedSeq[DataDesc] = iDescriptors
    val getBatchSize: Int = batchSize
    val getBatchIndex: Int = batchIndex

    lazy val mockModule: Module = Mockito.mock(classOf[Module])
  }

  test("PredictorSuite-testPredictorConstruction") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(1, 3, 2, 2),
      layout = Layout.NCHW))

    val mockPredictor = new MyPredictor("xyz", inputDescriptor)

    assert(mockPredictor.getBatchSize == 1)
    assert(mockPredictor.getBatchIndex == inputDescriptor(0).layout.indexOf('N'))

    val inputDescriptor2 = IndexedSeq[DataDesc](new DataDesc("data", Shape(1, 3, 2, 2),
      layout = Layout.NCHW),
      new DataDesc("data", Shape(2, 3, 2, 2), layout = Layout.NCHW))

    assertThrows[IllegalArgumentException] {
      val mockPredictor = new MyPredictor("xyz", inputDescriptor2)
    }

    // batchsize is defaulted to 1
    val iDesc2 = IndexedSeq[DataDesc](new DataDesc("data", Shape(3, 2, 2), layout = "CHW"))
    val p2 = new MyPredictor("xyz", inputDescriptor)
    assert(p2.getBatchSize == 1, "should use a default batch size of 1")

  }

  test("PredictorSuite-testWithFlatArrays") {

    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2),
      layout = Layout.NCHW))
    val inputData = Array.fill[Float](12)(1)

    // this will disposed at the end of the predict call on Predictor.
    val predictResult = IndexedSeq(NDArray.ones(Shape(1, 3, 2, 2)))

    val testPredictor = new MyPredictor("xyz", inputDescriptor)

    Mockito.doReturn(predictResult).when(testPredictor.mockModule)
      .predict(any(classOf[NDArrayIter]), any[Int], any[Boolean])

    val testFun = testPredictor.predict(IndexedSeq(inputData))

    assert(testFun.size == 1, "output size should be 1 ")

    assert(Array.fill[Float](12)(1).mkString == testFun(0).mkString)

    // Verify that the module was bound with batch size 1 and rebound back to the original
    // input descriptor. the number of times is twice here because loadModule overrides the
    // initial bind.
    Mockito.verify(testPredictor.mockModule, Mockito.times(2)).bind(any[IndexedSeq[DataDesc]],
      any[Option[IndexedSeq[DataDesc]]], any[Boolean], any[Boolean], any[Boolean]
      , any[Option[BaseModule]], any[String])
  }

  test("PredictorSuite-testWithFlatFloat64Arrays") {

    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2),
      layout = Layout.NCHW, dtype = DType.Float64))
    val inputData = Array.fill[Double](12)(1d)

    // this will disposed at the end of the predict call on Predictor.
    val predictResult = IndexedSeq(NDArray.ones(Shape(1, 3, 2, 2), dtype = DType.Float64))

    val testPredictor = new MyPredictor("xyz", inputDescriptor)

    Mockito.doReturn(predictResult).when(testPredictor.mockModule)
      .predict(any(classOf[NDArrayIter]), any[Int], any[Boolean])

    val testFun = testPredictor.predict(IndexedSeq(inputData))

    assert(testFun.size == 1, "output size should be 1 ")

    assert(testFun(0)(0).getClass == 1d.getClass)

    assert(Array.fill[Double](12)(1d).mkString == testFun(0).mkString)

    // Verify that the module was bound with batch size 1 and rebound back to the original
    // input descriptor. the number of times is twice here because loadModule overrides the
    // initial bind.
    Mockito.verify(testPredictor.mockModule, Mockito.times(2)).bind(any[IndexedSeq[DataDesc]],
      any[Option[IndexedSeq[DataDesc]]], any[Boolean], any[Boolean], any[Boolean]
      , any[Option[BaseModule]], any[String])
  }

  test("PredictorSuite-testWithNDArray") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2),
      layout = Layout.NCHW))
    val inputData = NDArray.ones(Shape(1, 3, 2, 2))

    // this will disposed at the end of the predict call on Predictor.
    val predictResult = IndexedSeq(NDArray.ones(Shape(1, 3, 2, 2)))

    val testPredictor = new MyPredictor("xyz", inputDescriptor)

    Mockito.doReturn(predictResult).when(testPredictor.mockModule)
      .predict(any(classOf[NDArrayIter]), any[Int], any[Boolean])

    val testFun = testPredictor.predictWithNDArray(IndexedSeq(inputData))

    assert(testFun.size == 1, "output size should be 1")

    assert(Array.fill[Float](12)(1).mkString == testFun(0).toArray.mkString)

    Mockito.verify(testPredictor.mockModule, Mockito.times(2)).bind(any[IndexedSeq[DataDesc]],
      any[Option[IndexedSeq[DataDesc]]], any[Boolean], any[Boolean], any[Boolean]
      , any[Option[BaseModule]], any[String])
  }
}
