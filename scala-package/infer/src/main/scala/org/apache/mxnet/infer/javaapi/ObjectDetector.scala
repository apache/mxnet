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

import java.awt.image.BufferedImage
import javafx.util.Pair

import org.apache.mxnet.javaapi.{Context, DataDesc, NDArray}

import scala.collection.JavaConverters
import scala.collection.JavaConverters._


class ObjectDetector(val objectDetector: org.apache.mxnet.infer.ObjectDetector){

  def this(modelPathPrefix: String, inputDescriptors: java.util.List[DataDesc], contexts: java.util.List[Context], epoch: Int)
  = this {
    val informationDesc = JavaConverters.asScalaIteratorConverter(inputDescriptors.iterator).asScala.toIndexedSeq map {a=>a: org.apache.mxnet.DataDesc}
    val inContexts = (contexts.asScala.toList map {a => a: org.apache.mxnet.Context}).toArray
    new org.apache.mxnet.infer.ObjectDetector(modelPathPrefix, informationDesc, inContexts, Some(epoch))
  }

  def imageObjectDetect(inputImage: BufferedImage, topK: Int): java.util.List[java.util.List[Pair[String, java.util.List[java.lang.Float]]]] = {
    val ret = objectDetector.imageObjectDetect(inputImage, Some(topK))
    (ret map {a=> (a map {entry => new Pair[String, java.util.List[java.lang.Float]](entry._1, entry._2.map(f => Float.box(f)).toList.asJava)}).asJava }).asJava
  }

  def objectDetectWithNDArray(input: java.util.List[NDArray], topK: Int): java.util.List[java.util.List[(String, java.util.List[java.lang.Float])]] = {
    val ret = objectDetector.objectDetectWithNDArray(convert(input.asScala.toIndexedSeq), Some(topK))
    (ret map {a=> (a map {entry => (entry._1, entry._2.map(f => Float.box(f)).toList.asJava)}).asJava }).asJava
  }

  def imageBatchObjectDetect(inputBatch: java.util.List[BufferedImage], topK: Int):
      java.util.List[java.util.List[Pair[String, java.util.List[java.lang.Float]]]] = {
    val ret = objectDetector.imageBatchObjectDetect(inputBatch.asScala, Some(topK))
    (ret map {a=> (a map {entry => new Pair[String, java.util.List[java.lang.Float]](entry._1, entry._2.map(f => Float.box(f)).toList.asJava)}).asJava }).asJava
  }

  def convert[B, A <% B](l: IndexedSeq[A]): IndexedSeq[B] = l map { a => a: B }

}


object ObjectDetector {
  implicit def fromObjectDetector(OD: org.apache.mxnet.infer.ObjectDetector): ObjectDetector = new ObjectDetector(OD)

  implicit def toObjectDetector(jOD: ObjectDetector): org.apache.mxnet.infer.ObjectDetector = jOD.objectDetector

  def loadImageFromFile(inputImagePath: String): BufferedImage = {
    org.apache.mxnet.infer.ImageClassifier.loadImageFromFile(inputImagePath)
  }

  def loadInputBatch(inputImagePaths: java.util.List[String]): java.util.List[BufferedImage] = {
    org.apache.mxnet.infer.ImageClassifier.loadInputBatch(inputImagePaths.asScala.toList).toList.asJava
  }
}
