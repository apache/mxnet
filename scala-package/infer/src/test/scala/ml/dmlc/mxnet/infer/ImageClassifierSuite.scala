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

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{DataDesc, Shape}

import org.mockito.Matchers._
import org.mockito.Mockito
import org.scalatest.{BeforeAndAfterAll}

// scalastyle:off
import java.awt.image.BufferedImage
// scalastyle:on

class ImageClassifierSuite extends ClassifierSuite with BeforeAndAfterAll {

  class MyImageClassifier(modelPathPrefix: String,
                           inputDescriptors: IndexedSeq[DataDesc])
    extends ImageClassifier(modelPathPrefix, inputDescriptors) {

    override def getPredictor(): MyClassyPredictor = {
      Mockito.mock(classOf[MyClassyPredictor])
    }

    override def getClassifier(modelPathPrefix: String, inputDescriptors: IndexedSeq[DataDesc]):
      Classifier = {
      Mockito.mock(classOf[Classifier])
    }
  }

  test("Rescale an image") {
    val image1 = new BufferedImage(100, 200, BufferedImage.TYPE_BYTE_GRAY)
    val image2 = ImageClassifier.reshapeImage(image1, 1000, 2000)

    assert(image2.getWidth === 1000)
    assert(image2.getHeight === 2000)
  }

  test("Convert BufferedImage to NDArray and rescale it") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(1, 3, 2, 2)))
    val testImageClassifier =
      new MyImageClassifier(modelPath, inputDescriptor)

    val image1 = new BufferedImage(100, 200, BufferedImage.TYPE_BYTE_GRAY)
    val image2 = ImageClassifier.reshapeImage(image1, 2, 2)

    val result = testImageClassifier.bufferedImageToPixels(image2)

    assert(result.shape == inputDescriptor(0).shape)
  }

  test("testWithInputImage") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(1, 3, 512, 512)))

    val inputImage = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB)

    val testImageClassifier: ImageClassifier =
      new MyImageClassifier(modelPath, inputDescriptor)

    val predictResult: IndexedSeq[Array[Float]] =
      IndexedSeq[Array[Float]](Array(.98f, 0.97f, 0.96f, 0.99f))

    val predictResultND: NDArray = NDArray.array(predictResult.flatten.toArray, Shape(1, 4))

    val predictResultOp : List[(String, Float)] =
            List[(String, Float)](("class1 label1", .98f), ("class2 label2", .97f),
              ("class3 label3", .96f), ("class4 label4", .99f))

    Mockito.doReturn(IndexedSeq(predictResultND)).when(testImageClassifier.predictor)
      .predictWithNDArray(any(classOf[IndexedSeq[NDArray]]))

    Mockito.doReturn(IndexedSeq(predictResultOp)).when(testImageClassifier.classifier)
      .classifyWithNDArray(any(classOf[IndexedSeq[NDArray]]), Some(anyInt()))

    val result: IndexedSeq[IndexedSeq[(String, Float)]] =
      testImageClassifier.classifyImage(inputImage, Some(4))

    for(i <- predictResult.indices) {
      assertResult(predictResult(i).sortBy(-_)) {
        result(i).map(_._2).toArray
      }
    }

  }

  test("testWithInputBatchImage") {
  }
}
