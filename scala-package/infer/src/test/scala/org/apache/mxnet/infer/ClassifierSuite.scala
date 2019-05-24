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

import java.io.File
import java.nio.file.{Files, Paths}
import java.util

import org.apache.mxnet.module.Module
import org.apache.mxnet.{Context, DType, DataDesc, NDArray, Shape}
import org.mockito.Matchers._
import org.mockito.Mockito
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.io

class ClassifierSuite extends FunSuite with BeforeAndAfterAll {

  private val logger = LoggerFactory.getLogger(classOf[Predictor])

  var modelPath = ""

  var synFilePath = ""

  def createTempModelFiles(): Unit = {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    logger.info("tempDirPath: %s".format(tempDirPath))

    val modelDirPath = tempDirPath + File.separator + "model"
    val synPath = tempDirPath + File.separator + "synset.txt"
    val synsetFile = new File(synPath)
    synsetFile.createNewFile()
    val lines: util.List[String] = util.Arrays.
      asList("class1 label1", "class2 label2", "class3 label3", "class4 label4")
    val path = Paths.get(synPath)
    Files.write(path, lines)

    this.modelPath = modelDirPath
    this.synFilePath = synsetFile.getCanonicalPath
    logger.info("modelPath: %s".format(this.modelPath))
    logger.info("synFilePath: %s".format(this.synFilePath))
  }

  override def beforeAll() {
    createTempModelFiles
  }

  override def afterAll() {
    new File(synFilePath).delete()
  }

  class MyClassyPredictor(val modelPathPrefix: String,
                          override val inputDescriptors: IndexedSeq[DataDesc])
    extends Predictor(modelPathPrefix, inputDescriptors, epoch = Some(0)) {

    override def loadModule(): Module = mockModule

    val getIDescriptor: IndexedSeq[DataDesc] = iDescriptors
    val getBatchSize: Int = batchSize
    val getBatchIndex: Int = batchIndex

    lazy val mockModule: Module = Mockito.mock(classOf[Module])
  }

  class MyClassifier(modelPathPrefix: String,
                     protected override val inputDescriptors: IndexedSeq[DataDesc])
    extends Classifier(modelPathPrefix, inputDescriptors, Context.cpu(), Some(0)) {

    override def getPredictor(): MyClassyPredictor = {
      Mockito.mock(classOf[MyClassyPredictor])
    }
    def getSynset(): IndexedSeq[String] = synset
  }

  test("ClassifierSuite-getSynsetFilePath") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2)))
    val testClassifer = new MyClassifier(modelPath, inputDescriptor)

    assertResult(this.synFilePath) {
      testClassifer.synsetFilePath
    }
  }

  test("ClassifierSuite-readSynsetFile") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2)))
    val testClassifer = new MyClassifier(modelPath, inputDescriptor)

    assertResult(io.Source.fromFile(this.synFilePath).getLines().toList) {
      testClassifer.getSynset()
    }
  }

  test("ClassifierSuite-flatArray-topK") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2)))
    val inputData = Array.fill[Float](12)(1)

    val predictResult : IndexedSeq[Array[Float]] =
      IndexedSeq[Array[Float]](Array(.98f, 0.97f, 0.96f, 0.99f))

    val testClassifier = new MyClassifier(modelPath, inputDescriptor)

    Mockito.doReturn(predictResult).when(testClassifier.predictor)
      .predict(any(classOf[IndexedSeq[Array[Float]]]))

    val result: IndexedSeq[(String, Float)] = testClassifier.
          classify(IndexedSeq(inputData), topK = Some(10))

    assertResult(predictResult(0).sortBy(-_)) {
      result.map(_._2).toArray
    }

  }

  test("ClassifierSuite-flatFloat64Array-topK") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2)))
    val inputData = Array.fill[Double](12)(1d)

    val predictResult : IndexedSeq[Array[Double]] =
      IndexedSeq[Array[Double]](Array(.98d, 0.97d, 0.96d, 0.99d))

    val testClassifier = new MyClassifier(modelPath, inputDescriptor)

    Mockito.doReturn(predictResult).when(testClassifier.predictor)
      .predict(any(classOf[IndexedSeq[Array[Double]]]))

    val result: IndexedSeq[(String, Double)] = testClassifier.
      classify(IndexedSeq(inputData), topK = Some(10))

    assert((result(0)._2).getClass == 1d.getClass)

    assertResult(predictResult(0).sortBy(-_)) {
      result.map(_._2).toArray
    }

  }

  test("ClassifierSuite-flatArrayInput") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2)))
    val inputData = Array.fill[Float](12)(1)

    val predictResult : IndexedSeq[Array[Float]] =
      IndexedSeq[Array[Float]](Array(.98f, 0.97f, 0.96f, 0.99f))

    val testClassifier = new MyClassifier(modelPath, inputDescriptor)

    Mockito.doReturn(predictResult).when(testClassifier.predictor)
      .predict(any(classOf[IndexedSeq[Array[Float]]]))

    val result: IndexedSeq[(String, Float)] = testClassifier.
          classify(IndexedSeq(inputData))

    assertResult(predictResult(0)) {
      result.map(_._2).toArray
    }
  }

  test("ClassifierSuite-flatArrayFloat64Input") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2)))
    val inputData = Array.fill[Double](12)(1d)

    val predictResult : IndexedSeq[Array[Double]] =
      IndexedSeq[Array[Double]](Array(.98d, 0.97d, 0.96d, 0.99d))

    val testClassifier = new MyClassifier(modelPath, inputDescriptor)

    Mockito.doReturn(predictResult).when(testClassifier.predictor)
      .predict(any(classOf[IndexedSeq[Array[Double]]]))

    val result: IndexedSeq[(String, Double)] = testClassifier.
      classify(IndexedSeq(inputData))

    assert((result(0)._2).getClass == 1d.getClass)

    assertResult(predictResult(0)) {
      result.map(_._2).toArray
    }
  }

  test("ClassifierSuite-NDArray1InputWithoutTopK") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2)))
    val inputDataShape = Shape(1, 3, 2, 2)
    val inputData = NDArray.ones(inputDataShape)
    val predictResult: IndexedSeq[Array[Float]] =
      IndexedSeq[Array[Float]](Array(.98f, 0.97f, 0.96f, 0.99f))

    val predictResultND: NDArray = NDArray.array(predictResult.flatten.toArray, Shape(1, 4))

    val testClassifier = new MyClassifier(modelPath, inputDescriptor)

    Mockito.doReturn(IndexedSeq(predictResultND)).when(testClassifier.predictor)
      .predictWithNDArray(any(classOf[IndexedSeq[NDArray]]))

    val result: IndexedSeq[IndexedSeq[(String, Float)]] = testClassifier.
          classifyWithNDArray(IndexedSeq(inputData))

    assert(predictResult.size == result.size)

    for(i <- predictResult.indices) {
      assertResult(predictResult(i)) {
        result(i).map(_._2).toArray
      }
    }
  }

  test("ClassifierSuite-NDArray3InputWithTopK") {

    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc("data", Shape(2, 3, 2, 2)))
    val inputDataShape = Shape(3, 3, 2, 2)
    val inputData = NDArray.ones(inputDataShape)

    val predictResult: IndexedSeq[Array[Float]] =
      IndexedSeq[Array[Float]](Array(.98f, 0.97f, 0.96f, 0.99f),
        Array(.98f, 0.97f, 0.96f, 0.99f), Array(.98f, 0.97f, 0.96f, 0.99f))

    val predictResultND: NDArray = NDArray.array(predictResult.flatten.toArray, Shape(3, 4))

    val testClassifier = new MyClassifier(modelPath, inputDescriptor)

    Mockito.doReturn(IndexedSeq(predictResultND)).when(testClassifier.predictor)
      .predictWithNDArray(any(classOf[IndexedSeq[NDArray]]))

    val result: IndexedSeq[IndexedSeq[(String, Float)]] = testClassifier.
          classifyWithNDArray(IndexedSeq(inputData), topK = Some(10))

    assert(predictResult.size == result.size)

    for(i <- predictResult.indices) {
      assertResult(predictResult(i).sortBy(-_)) {
        result(i).map(_._2).toArray
      }
    }
  }

}
