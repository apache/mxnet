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

import ml.dmlc.mxnet.io.NDArrayIter
import ml.dmlc.mxnet.module.{BaseModule, Module}
import ml.dmlc.mxnet.{DataDesc, NDArray, Shape}
import org.mockito.Matchers._
import org.mockito.Mockito
import java.nio.file.{Files, Paths}
import java.util
import java.io.File
import scala.io
import scala.collection.mutable.ListBuffer
import scala.io

import org.scalatest.{BeforeAndAfterAll, FunSuite}

class ClassifierSuite extends FunSuite with BeforeAndAfterAll {

  var modelPath = ""

  var synFilePath = ""

  def createTempModelFiles(): Unit = {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    val modelDirPath = tempDirPath + "model"
    val synPath = tempDirPath + File.separator + "synset.txt"
    val synsetFile = new File(synPath)
    synsetFile.createNewFile()
    val lines: util.List[String] = util.Arrays.
      asList("class1 label1", "class2 label2", "class3 label3", "class4 label4")
    val path = Paths.get(synPath)
    Files.write(path, lines)

    this.modelPath = modelDirPath
    this.synFilePath = synsetFile.getCanonicalPath
  }

  override def beforeAll() {
    createTempModelFiles
  }

  override def afterAll() {
    new File(synFilePath).delete()
  }

  class MyClassyPredictor(val modelPathPrefix: String,
                          override val inputDescriptors: IndexedSeq[DataDesc])
    extends Predictor(modelPathPrefix, inputDescriptors) {

    override def loadModule(): Module = mockModule

    val getIDescriptor: IndexedSeq[DataDesc] = iDescriptors
    val getBatchSize: Int = batchSize
    val getBatchIndex: Int = batchIndex

    lazy val mockModule: Module = Mockito.mock(classOf[Module])
  }

  class MyClassifier(modelPathPrefix: String,
                     protected override val inputDescriptors: IndexedSeq[DataDesc])
    extends Classifier(modelPathPrefix, inputDescriptors) {

    override def getPredictor(modelPathPrefix: String,
                              inputDescriptors: IndexedSeq[DataDesc]): MyClassyPredictor = {
      Mockito.mock(classOf[MyClassyPredictor])
    }
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
      testClassifer.synset
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

    val result: List[(String, Float)] = testClassifier.
      classify(IndexedSeq(inputData), topK = Some(10))

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

    val result: List[(String, Float)] = testClassifier.
      classify(IndexedSeq(inputData))

    assertResult(predictResult(0)) {
      result.map(_._2).toArray
    }
  }

}
