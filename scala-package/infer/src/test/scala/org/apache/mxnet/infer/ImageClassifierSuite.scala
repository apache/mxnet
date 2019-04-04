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

import org.apache.mxnet.{DType, DataDesc, Shape, NDArray, Context}

import org.mockito.Matchers._
import org.mockito.Mockito
import org.scalatest.BeforeAndAfterAll

// scalastyle:off
import java.awt.image.BufferedImage
// scalastyle:on

/**
  * Unit tests for ImageClassifier
  */
class ImageClassifierSuite extends ClassifierSuite with BeforeAndAfterAll {

  class MyImageClassifier(modelPathPrefix: String,
                          inputDescriptors: IndexedSeq[DataDesc])
    extends ImageClassifier(modelPathPrefix, inputDescriptors) {

    override def getPredictor(): MyClassyPredictor = {
      Mockito.mock(classOf[MyClassyPredictor])
    }

    override def getClassifier(modelPathPrefix: String, inputDescriptors:
    IndexedSeq[DataDesc], contexts: Array[Context] = Context.cpu(),
                               epoch: Option[Int] = Some(0)): Classifier = {
      Mockito.mock(classOf[Classifier])
    }

    def getSynset(): IndexedSeq[String] = synset
  }

  test("ImageClassifierSuite-testRescaleImage") {
    val image1 = new BufferedImage(100, 200, BufferedImage.TYPE_BYTE_GRAY)
    val image2 = ImageClassifier.reshapeImage(image1, 1000, 2000)

    assert(image2.getWidth === 1000)
    assert(image2.getHeight === 2000)
  }

  test("ImageClassifierSuite-testConvertBufferedImageToNDArray") {
    val dType = DType.Float32
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(1, 3, 2, 2),
      dType, "NCHW"))

    val image1 = new BufferedImage(100, 200, BufferedImage.TYPE_BYTE_GRAY)
    val image2 = ImageClassifier.reshapeImage(image1, 2, 2)

    val result = ImageClassifier.bufferedImageToPixels(image2, Shape(3, 2, 2))

    assert(result.shape == inputDescriptor(0).shape.drop(1))
  }

  test("ImageClassifierSuite-testWithInputImage") {
    val dType = DType.Float32
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(1, 3, 512, 512),
      dType, "NCHW"))

    val inputImage = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB)

    val testImageClassifier: ImageClassifier =
      new MyImageClassifier(modelPath, inputDescriptor)

    val predictExpected: IndexedSeq[Array[Float]] =
      IndexedSeq[Array[Float]](Array(.98f, 0.97f, 0.96f, 0.99f))

    val synset = testImageClassifier.synset

    val predictExpectedOp: List[(String, Float)] =
      List[(String, Float)]((synset(1), .98f), (synset(2), .97f),
        (synset(3), .96f), (synset(0), .99f))

    val predictExpectedND: NDArray = NDArray.array(predictExpected.flatten.toArray, Shape(1, 4))

    Mockito.doReturn(IndexedSeq(predictExpectedND)).when(testImageClassifier.predictor)
      .predictWithNDArray(any(classOf[IndexedSeq[NDArray]]))

    Mockito.doReturn(IndexedSeq(predictExpectedOp))
      .when(testImageClassifier.getClassifier(modelPath, inputDescriptor))
      .classifyWithNDArray(any(classOf[IndexedSeq[NDArray]]), Some(anyInt()))

    val predictResult: IndexedSeq[IndexedSeq[(String, Float)]] =
      testImageClassifier.classifyImage(inputImage, Some(4))

    for (i <- predictExpected.indices) {
      assertResult(predictExpected(i).sortBy(-_)) {
        predictResult(i).map(_._2).toArray
      }
    }
  }

  test("ImageClassifierSuite-testWithInputBatchImage") {
    val dType = DType.Float32
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(1, 3, 512, 512),
      dType, "NCHW"))

    val inputImage = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB)
    val imageBatch = IndexedSeq[BufferedImage](inputImage, inputImage)

    val testImageClassifier: ImageClassifier =
      new MyImageClassifier(modelPath, inputDescriptor)

    val predictExpected: IndexedSeq[Array[Array[Float]]] =
      IndexedSeq[Array[Array[Float]]](Array(Array(.98f, 0.97f, 0.96f, 0.99f),
        Array(.98f, 0.97f, 0.96f, 0.99f)))

    val synset = testImageClassifier.synset

    val predictExpectedOp: List[List[(String, Float)]] =
      List[List[(String, Float)]](List((synset(1), .98f), (synset(2), .97f),
        (synset(3), .96f), (synset(0), .99f)),
        List((synset(1), .98f), (synset(2), .97f),
          (synset(3), .96f), (synset(0), .99f)))

    val predictExpectedND: NDArray = NDArray.array(predictExpected.flatten.flatten.toArray,
      Shape(2, 4))

    Mockito.doReturn(IndexedSeq(predictExpectedND)).when(testImageClassifier.predictor)
      .predictWithNDArray(any(classOf[IndexedSeq[NDArray]]))

    Mockito.doReturn(IndexedSeq(predictExpectedOp))
      .when(testImageClassifier.getClassifier(modelPath, inputDescriptor))
      .classifyWithNDArray(any(classOf[IndexedSeq[NDArray]]), Some(anyInt()))

    val result: IndexedSeq[IndexedSeq[(String, Float)]] =
      testImageClassifier.classifyImageBatch(imageBatch, Some(4))

    for (i <- predictExpected.indices) {
      for (idx <- predictExpected(i).indices) {
        assertResult(predictExpected(i)(idx).sortBy(-_)) {
          result(i).map(_._2).toArray
        }
      }
    }
  }
}
