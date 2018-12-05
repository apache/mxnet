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

import scala.collection.parallel.mutable.ParArray
// scalastyle:on
import org.apache.mxnet.NDArray
import org.apache.mxnet.DataDesc
import org.apache.mxnet.Context
import scala.collection.mutable.ListBuffer

/**
  * A class for object detection tasks
  *
  * @param modelPathPrefix    Path prefix from where to load the model artifacts.
  *                           These include the symbol, parameters, and synset.txt.
  *                           Example: file://model-dir/ssd_resnet50_512 (containing
  *                           ssd_resnet50_512-symbol.json, ssd_resnet50_512-0000.params,
  *                           and synset.txt)
  * @param inputDescriptors   Descriptors defining the input node names, shape,
  *                           layout and type parameters
  * @param contexts           Device contexts on which you want to run inference.
  *                           Defaults to CPU.
  * @param epoch              Model epoch to load; defaults to 0
  */
class ObjectDetector(modelPathPrefix: String,
                     inputDescriptors: IndexedSeq[DataDesc],
                     contexts: Array[Context] = Context.cpu(),
                     epoch: Option[Int] = Some(0)) {

  protected[infer]  val imgClassifier: ImageClassifier =
    getImageClassifier(modelPathPrefix, inputDescriptors, contexts, epoch)

  protected[infer] val inputShape = imgClassifier.inputShape

  protected[infer] val handler = imgClassifier.handler

  protected[infer] val predictor = imgClassifier.predictor

  protected[infer] val synset = imgClassifier.synset

  protected[infer] val height = imgClassifier.height

  protected[infer] val width = imgClassifier.width

  /**
    * Detects objects and returns bounding boxes with corresponding class/label
    *
    * @param inputImage       Path prefix of the input image
    * @param topK             Number of result elements to return, sorted by probability
    * @return                 List of list of tuples of
    *                         (class, [probability, xmin, ymin, xmax, ymax])
    */
  def imageObjectDetect(inputImage: BufferedImage,
                        topK: Option[Int] = None)
  : IndexedSeq[IndexedSeq[(String, Array[Float])]] = {

    val scaledImage = ImageClassifier.reshapeImage(inputImage, width, height)
    val imageShape = inputShape.drop(1)
    val pixelsNDArray = ImageClassifier.bufferedImageToPixels(scaledImage, imageShape)
    val pixelsNDWithBatch = NDArray.api.expand_dims(pixelsNDArray, 0)
    handler.execute(pixelsNDArray.dispose())
    val output = objectDetectWithNDArray(IndexedSeq(pixelsNDWithBatch), topK)
    handler.execute(pixelsNDWithBatch.dispose())
    output
  }

  /**
    * Takes input images as NDArrays. Useful when you want to perform multiple operations on
    * the input array, or when you want to pass a batch of input images.
    *
    * @param input            Indexed Sequence of NDArrays
    * @param topK             (Optional) How many top_k (sorting will be based on the last axis)
    *                         elements to return. If not passed, returns all unsorted output.
    * @return                 List of list of tuples of
    *                         (class, [probability, xmin, ymin, xmax, ymax])
    */
  def objectDetectWithNDArray(input: IndexedSeq[NDArray], topK: Option[Int])
  : IndexedSeq[IndexedSeq[(String, Array[Float])]] = {

    // Copy NDArray to CPU to avoid frequent GPU to CPU copying
    val predictResult = predictor.predictWithNDArray(input)(0).asInContext(Context.cpu())
    // Parallel Execution with ParArray for better performance
    var batchResult = new ParArray[IndexedSeq[(String, Array[Float])]](predictResult.shape(0))
    (0 until predictResult.shape(0)).toArray.par.foreach( i => {
      val r = predictResult.at(i)
      batchResult(i) = sortAndReformat(r, topK)
      handler.execute(r.dispose())
    })
    handler.execute(predictResult.dispose())
    batchResult.toIndexedSeq
  }

  private[infer] def sortAndReformat(predictResultND: NDArray, topK: Option[Int])
  : IndexedSeq[(String, Array[Float])] = {
    // iterating over the all the predictions
    val length = predictResultND.shape(0)

    val predictResult = (0 until length).toArray.par.flatMap( i => {
      val r = predictResultND.at(i)
      val tempArr = r.toArray
      val res = if (tempArr(0) != -1.0) {
        Array[Array[Float]](tempArr)
      } else {
        // Ignore the minus 1 part
        Array[Array[Float]]()
      }
      handler.execute(r.dispose())
      res
    }).toArray
    var result = IndexedSeq[(String, Array[Float])]()
    if (topK.isDefined) {
      var sortedIndices = predictResult.zipWithIndex.sortBy(-_._1(1)).map(_._2)
      sortedIndices = sortedIndices.take(topK.get)
      // takeRight(5) would provide the output as Array[Accuracy, Xmin, Ymin, Xmax, Ymax
      result = sortedIndices.map(idx
      => (synset(predictResult(idx)(0).toInt),
          predictResult(idx).takeRight(5))).toIndexedSeq
    } else {
      result = predictResult.map(ele
      => (synset(ele(0).toInt), ele.takeRight(5))).toIndexedSeq
    }
    result
  }

  /**
    * To classify batch of input images according to the provided model
    *
    * @param inputBatch       Input array of buffered images
    * @param topK             Number of result elements to return, sorted by probability
    * @return                 List of list of tuples of (class, probability)
    */
  def imageBatchObjectDetect(inputBatch: Traversable[BufferedImage], topK: Option[Int] = None):
  IndexedSeq[IndexedSeq[(String, Array[Float])]] = {

    val inputBatchSeq = inputBatch.toIndexedSeq
    val imageBatch = inputBatchSeq.indices.par.map(idx => {
      val scaledImage = ImageClassifier.reshapeImage(inputBatchSeq(idx), width, height)
      val imageShape = inputShape.drop(1)
      val pixelsND = ImageClassifier.bufferedImageToPixels(scaledImage, imageShape)
      val pixelsNDWithBatch = NDArray.api.expand_dims(pixelsND, 0).get
      handler.execute(pixelsND.dispose())
      pixelsNDWithBatch
    })
    val op = NDArray.concatenate(imageBatch.toList)

    val result = objectDetectWithNDArray(IndexedSeq(op), topK)
    handler.execute(op.dispose())
    handler.execute(imageBatch.foreach(_.dispose()))
    result
  }

  private[infer] def getImageClassifier(modelPathPrefix: String,
                                        inputDescriptors: IndexedSeq[DataDesc],
                         contexts: Array[Context] = Context.cpu(),
                         epoch: Option[Int] = Some(0)):
  ImageClassifier = {
    new ImageClassifier(modelPathPrefix, inputDescriptors, contexts, epoch)
  }

}
