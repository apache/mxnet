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

package org.apache.mxnet.infer.javaapi

// scalastyle:off
import java.awt.image.BufferedImage
// scalastyle:on

import org.apache.mxnet.javaapi.{Context, DataDesc, NDArray, Shape}

import scala.collection.JavaConverters
import scala.collection.JavaConverters._
import scala.language.implicitConversions

/**
  * The ObjectDetector class helps to run ObjectDetection tasks where the goal
  * is to find bounding boxes and corresponding labels for objects in a image.
  *
  * @param objDetector A source Scala Object detector
  */
class ObjectDetector private[mxnet] (val objDetector: org.apache.mxnet.infer.ObjectDetector){

  /**
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
  def this(modelPathPrefix: String, inputDescriptors: java.lang.Iterable[DataDesc], contexts:
  java.lang.Iterable[Context], epoch: Int)
  = this {
    val informationDesc = JavaConverters.asScalaIteratorConverter(inputDescriptors.iterator)
      .asScala.toIndexedSeq map {a => a: org.apache.mxnet.DataDesc}
    val inContexts = (contexts.asScala.toList map {a => a: org.apache.mxnet.Context}).toArray
    // scalastyle:off
    new org.apache.mxnet.infer.ObjectDetector(modelPathPrefix, informationDesc, inContexts, Some(epoch))
    // scalastyle:on
  }

  /**
    * Detects objects and returns bounding boxes with corresponding class/label
    *
    * @param inputImage       Path prefix of the input image
    * @param topK             Number of result elements to return, sorted by probability
    * @return                 List of list of tuples of
    *                         (class, [probability, xmin, ymin, xmax, ymax])
    */
  def imageObjectDetect(inputImage: BufferedImage, topK: Int):
  java.util.List[java.util.List[ObjectDetectorOutput]] = {
    val ret = objDetector.imageObjectDetect(inputImage, Some(topK))
    (ret map {a => (a map {e => new ObjectDetectorOutput(e._1, e._2)}).asJava}).asJava
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
  def objectDetectWithNDArray(input: java.lang.Iterable[NDArray], topK: Int):
  java.util.List[java.util.List[ObjectDetectorOutput]] = {
    val ret = objDetector.objectDetectWithNDArray(convert(input.asScala.toIndexedSeq), Some(topK))
    (ret map {a => (a map {e => new ObjectDetectorOutput(e._1, e._2)}).asJava}).asJava
  }

  /**
    * To classify batch of input images according to the provided model
    *
    * @param inputBatch       Input array of buffered images
    * @param topK             Number of result elements to return, sorted by probability
    * @return                 List of list of tuples of (class, probability)
    */
  def imageBatchObjectDetect(inputBatch: java.lang.Iterable[BufferedImage], topK: Int):
      java.util.List[java.util.List[ObjectDetectorOutput]] = {
    val ret = objDetector.imageBatchObjectDetect(inputBatch.asScala, Some(topK))
    (ret map {a => (a map {e => new ObjectDetectorOutput(e._1, e._2)}).asJava}).asJava
  }

  /**
    * Helper to map an implicit conversion
    * @param l The value to convert
    * @tparam B The desired type
    * @tparam A The input type
    * @return The converted result
    */
  def convert[B, A <% B](l: IndexedSeq[A]): IndexedSeq[B] = l map { a => a: B }

}


object ObjectDetector {

  /**
    * Loads an input images from file
    * @param inputImagePath   Path of single input image
    * @return                 BufferedImage Buffered image
    */
  def loadImageFromFile(inputImagePath: String): BufferedImage = {
    org.apache.mxnet.infer.ImageClassifier.loadImageFromFile(inputImagePath)
  }

  /**
    * Reshape the input image to a new shape
    *
    * @param img              Input image
    * @param newWidth         New width for rescaling
    * @param newHeight        New height for rescaling
    * @return                 Rescaled BufferedImage
    */
  def reshapeImage(img : BufferedImage, newWidth: Int, newHeight: Int): BufferedImage = {
    org.apache.mxnet.infer.ImageClassifier.reshapeImage(img, newWidth, newHeight)
  }

  /**
    * Convert input BufferedImage to NDArray of input shape
    * Note: Caller is responsible to dispose the NDArray
    * returned by this method after the use.
    *
    * @param resizedImage BufferedImage to get pixels from
    * @param inputImageShape Input shape; for example for resnet it is (3,224,224).
    *                        Should be same as inputDescriptor shape.
    * @return NDArray pixels array with shape (3, 224, 224) in CHW format
    */
  def bufferedImageToPixels(resizedImage: BufferedImage, inputImageShape: Shape): NDArray = {
    org.apache.mxnet.infer.ImageClassifier.bufferedImageToPixels(resizedImage, inputImageShape)
  }

  /**
    * Loads a batch of images from a folder
    * @param inputImagePaths  Path to a folder of images
    * @return                   List of buffered images
    */
  def loadInputBatch(inputImagePaths: java.lang.Iterable[String]): java.util.List[BufferedImage] = {
    org.apache.mxnet.infer.ImageClassifier
      .loadInputBatch(inputImagePaths.asScala.toList).toList.asJava
  }

  /**
    * Implicitly convert a Scala ObjectDetector to a Java ObjectDetector
    * @param OD The Scala ObjectDetector
    * @return The Java ObjectDetector
    */
  implicit def fromObjectDetector(OD: org.apache.mxnet.infer.ObjectDetector):
  ObjectDetector = new ObjectDetector(OD)

  /**
    * Implicitly converts a Java ObjectDetector to a Scala ObjectDetector
    * @param jOD The Java ObjectDetector
    * @return The Scala ObjectDetector
    */
  implicit def toObjectDetector(jOD: ObjectDetector):
  org.apache.mxnet.infer.ObjectDetector = jOD.objDetector
}
