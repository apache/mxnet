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

// scalastyle:off
import java.awt.image.BufferedImage
// scalastyle:on
import org.apache.mxnet.Context
import org.apache.mxnet.DataDesc
import org.apache.mxnet.{NDArray, Shape}
import org.mockito.Matchers.any
import org.mockito.Mockito
import org.scalatest.BeforeAndAfterAll


class ObjectDetectorSuite extends ClassifierSuite with BeforeAndAfterAll {

  class MyObjectDetector(modelPathPrefix: String,
                         inputDescriptors: IndexedSeq[DataDesc])
    extends ObjectDetector(modelPathPrefix, inputDescriptors) {

    override def getImageClassifier(modelPathPrefix: String, inputDescriptors:
        IndexedSeq[DataDesc], contexts: Array[Context] = Context.cpu(),
        epoch: Option[Int] = Some(0)): ImageClassifier = {
      new MyImageClassifier(modelPathPrefix, inputDescriptors)
    }

  }

  class MyImageClassifier(modelPathPrefix: String,
                     protected override val inputDescriptors: IndexedSeq[DataDesc])
    extends ImageClassifier(modelPathPrefix, inputDescriptors, Context.cpu(), Some(0)) {

    override def getPredictor(): MyClassyPredictor = {
      Mockito.mock(classOf[MyClassyPredictor])
    }

    override def getClassifier(modelPathPrefix: String, inputDescriptors: IndexedSeq[DataDesc],
                               contexts: Array[Context] = Context.cpu(),
                               epoch: Option[Int] = Some(0)):
    Classifier = {
      new MyClassifier(modelPathPrefix, inputDescriptors)
    }
  }

  class MyClassifier(modelPathPrefix: String,
                     protected override val inputDescriptors: IndexedSeq[DataDesc])
    extends Classifier(modelPathPrefix, inputDescriptors) {

    override def getPredictor(): MyClassyPredictor = {
      Mockito.mock(classOf[MyClassyPredictor])
    }
    def getSynset(): IndexedSeq[String] = synset
  }

  test("objectDetectWithInputImage") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(1, 3, 512, 512)))
    val inputImage = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB)
    val testObjectDetector: ObjectDetector =
      new MyObjectDetector(modelPath, inputDescriptor)

    val predictRaw: IndexedSeq[Array[Array[Float]]] =
      IndexedSeq[Array[Array[Float]]](Array(
        Array(1.0f, 0.42f, 0.45f, 0.66f, 0.72f, 0.88f),
        Array(2.0f, 0.88f, 0.21f, 0.33f, 0.45f, 0.66f),
        Array(3.0f, 0.62f, 0.50f, 0.42f, 0.68f, 0.99f)
      ))
    val predictResultND: NDArray =
      NDArray.array(predictRaw.flatten.flatten.toArray, Shape(1, 3, 6))

    val synset = testObjectDetector.synset

    val predictResult: IndexedSeq[IndexedSeq[(String, Array[Float])]] =
      IndexedSeq[IndexedSeq[(String, Array[Float])]](
        IndexedSeq[(String, Array[Float])](
          (synset(2), Array(0.88f, 0.21f, 0.33f, 0.45f, 0.66f)),
          (synset(3), Array(0.62f, 0.50f, 0.42f, 0.68f, 0.99f)),
          (synset(1), Array(0.42f, 0.45f, 0.66f, 0.72f, 0.88f))
        )
      )

    Mockito.doReturn(IndexedSeq(predictResultND)).when(testObjectDetector.predictor)
      .predictWithNDArray(any(classOf[IndexedSeq[NDArray]]))

    val result: IndexedSeq[IndexedSeq[(String, Array[Float])]] =
      testObjectDetector.imageObjectDetect(inputImage, Some(3))

    for (idx <- predictResult(0).indices) {
      assert(predictResult(0)(idx)._1 == result(0)(idx)._1)
      for (arridx <- predictResult(0)(idx)._2.indices) {
        assert(predictResult(0)(idx)._2(arridx) == result(0)(idx)._2(arridx))
      }
    }
  }

  test("objectDetectWithBatchImages") {
    val inputDescriptor = IndexedSeq[DataDesc](new DataDesc(modelPath, Shape(1, 3, 512, 512)))
    val inputImage = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB)
    val imageBatch = IndexedSeq[BufferedImage](inputImage, inputImage)

    val testObjectDetector: ObjectDetector =
      new MyObjectDetector(modelPath, inputDescriptor)

    val predictRaw: IndexedSeq[Array[Array[Float]]] =
      IndexedSeq[Array[Array[Float]]](
        Array(
          Array(1.0f, 0.42f, 0.45f, 0.66f, 0.72f, 0.88f),
          Array(2.0f, 0.88f, 0.21f, 0.33f, 0.45f, 0.66f),
          Array(3.0f, 0.62f, 0.50f, 0.42f, 0.68f, 0.99f)
        ),
        Array(
          Array(0.0f, 0.42f, 0.45f, 0.66f, 0.72f, 0.88f),
          Array(2.0f, 0.23f, 0.21f, 0.33f, 0.45f, 0.66f),
          Array(2.0f, 0.94f, 0.50f, 0.42f, 0.68f, 0.99f)
        )
      )
    val predictResultND: NDArray =
      NDArray.array(predictRaw.flatten.flatten.toArray, Shape(2, 3, 6))

    val synset = testObjectDetector.synset

    val predictResult: IndexedSeq[IndexedSeq[(String, Array[Float])]] =
      IndexedSeq[IndexedSeq[(String, Array[Float])]](
        IndexedSeq[(String, Array[Float])](
          (synset(2), Array(0.88f, 0.21f, 0.33f, 0.45f, 0.66f)),
          (synset(3), Array(0.62f, 0.50f, 0.42f, 0.68f, 0.99f)),
          (synset(1), Array(0.42f, 0.45f, 0.66f, 0.72f, 0.88f))
        ),
        IndexedSeq[(String, Array[Float])](
          (synset(2), Array(0.94f, 0.50f, 0.42f, 0.68f, 0.99f)),
          (synset(0), Array(0.42f, 0.45f, 0.66f, 0.72f, 0.88f)),
          (synset(2), Array(0.23f, 0.21f, 0.33f, 0.45f, 0.66f))
        )
      )

    Mockito.doReturn(IndexedSeq(predictResultND)).when(testObjectDetector.predictor)
      .predictWithNDArray(any(classOf[IndexedSeq[NDArray]]))

    val result: IndexedSeq[IndexedSeq[(String, Array[Float])]] =
      testObjectDetector.imageBatchObjectDetect(imageBatch, Some(3))
    for (preidx <- predictResult.indices) {
      for (idx <- predictResult(preidx).indices) {
        assert(predictResult(preidx)(idx)._1 == result(preidx)(idx)._1)
        for (arridx <- predictResult(preidx)(idx)._2.indices) {
          assert(predictResult(preidx)(idx)._2(arridx) == result(preidx)(idx)._2(arridx))
        }
      }
    }

  }

}
