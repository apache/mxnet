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
