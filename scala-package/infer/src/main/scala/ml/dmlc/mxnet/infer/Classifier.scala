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

import ml.dmlc.mxnet.{DataDesc, NDArray}
import java.io.File

import scala.io
import scala.collection.mutable.ListBuffer

trait ClassifierBase {

  /**
    * Takes an Array of Floats and returns corresponding labels, score tuples.
    * @param input: IndexedSequence one-dimensional array of Floats.
    * @param topK: (Optional) How many top_k(sorting will be based on the last axis)
    *             elements to return, if not passed returns unsorted output.
    * @return IndexedSequence of (Label, Score) tuples.
    */
  def classify(input: IndexedSeq[Array[Float]],
               topK: Option[Int] = None): List[(String, Float)]

  /**
    * Takes a Sequence of NDArrays and returns Label, Score tuples.
    * @param input: Indexed Sequence of NDArrays
    * @param topK: (Optional) How many top_k(sorting will be based on the last axis)
    *             elements to return, if not passed returns unsorted output.
    * @return Traversable Sequence of (Label, Score) tuple, Score will be in the form of NDArray
    */
  def classifyWithNDArray(input: IndexedSeq[NDArray],
                          topK: Option[Int] = None): IndexedSeq[List[(String, Float)]]
}

/**
  * A class for classifier tasks
  * @param modelPathPrefix PathPrefix from where to load the symbol, parameters and synset.txt
  *                        Example: file://model-dir/resnet-152(containing resnet-152-symbol.json
  *                        file://model-dir/synset.txt
  * @param inputDescriptors Descriptors defining the input node names, shape,
  *                         layout and Type parameters
  * @param outputDescriptor Output Descriptor defining the output node name, shape,
  *                          layout and Type parameter
  */
class Classifier(modelPathPrefix: String,
                 protected val inputDescriptors: IndexedSeq[DataDesc],
                 protected var outputDescriptor:
                 Option[DataDesc] = None) extends ClassifierBase {

  val synsetFilePath = getSynsetFilePath

  if (outputDescriptor.isDefined) {
    require(outputDescriptor.size == 1, "expected single output")
  }

  val outDescriptor : Option[IndexedSeq[DataDesc]] = if (!outputDescriptor.isDefined) None
    else Some(IndexedSeq(outputDescriptor.get))

  val predictor: PredictBase = new Predictor(modelPathPrefix, inputDescriptors, outDescriptor)

  val synset = readSynsetFile(synsetFilePath)

  val handler = MXNetHandler()

  /**
    * Takes a flat arrays as input and returns a List of (Label, tuple)
    * @param input: IndexedSequence one-dimensional array of Floats.
    * @param topK: (Optional) How many top_k(sorting will be based on the last axis)
    *             elements to return, if not passed returns unsorted output.
    * @return IndexedSequence of (Label, Score) tuples.
    */
  override def classify(input: IndexedSeq[Array[Float]],
                        topK: Option[Int] = None): List[(String, Float)] = {

    // considering only the first output
    val predictResult = predictor.predict(input)(0)
    var result: List[(String, Float)] = List.empty

    if (topK.isDefined) {
      val sortedIndex = predictResult.zipWithIndex.sortBy(_._1).map(_._2).take(topK.get)
      result = sortedIndex.map(i => (synset(i), predictResult(i))).toList
    } else {
      result = synset.zip(predictResult).toList
    }
    result
  }

  /**
    * Takes input as NDArrays, useful when
    * @param input: Indexed Sequence of NDArrays
    * @param topK: (Optional) How many top_k(sorting will be based on the last axis)
    *             elements to return, if not passed returns unsorted output.
    * @return Traversable Sequence of (Label, Score) tuple, Score will be in the form of NDArray
    */
  override def classifyWithNDArray(input: IndexedSeq[NDArray], topK: Option[Int] = None)
  : IndexedSeq[List[(String, Float)]] = {

    val predictResultND = predictor.predictWithNDArray(input)
    val predictResult = predictResultND.map(_.toArray)


    var result: ListBuffer[List[(String, Float)]] = ListBuffer.empty[List[(String, Float)]]

    if (topK.isDefined) {
      val sortedIndices = predictResult.map(r =>
        r.zipWithIndex.sortBy(_._1).map(_._2).take(topK.get)
      )
      for (i <- sortedIndices.indices) {
        result += sortedIndices(i).map(sIndx => (synset(sIndx), predictResult(i)(sIndx))).toList
      }
    } else {
      for (i <- predictResult.indices) {
        result += synset.zip(predictResult(i))
      }
    }

    handler.execute(predictResultND.foreach(_.dispose()))

    result.toIndexedSeq
  }

  def getSynsetFilePath: String = {
    val dirPath = modelPathPrefix.substring(0, 1 + modelPathPrefix.lastIndexOf(File.pathSeparator))
    val d = new File(dirPath)
    require(d.exists && d.isDirectory, "directory: %s not found".format(dirPath))

    val s = new File(dirPath + File.pathSeparator + "synset.txt")
    require(s.exists() && s.isFile, "File synset.txt should exist inside modelPath: %s".format
    (dirPath + File.pathSeparator + "synset.txt"))

    s.getCanonicalPath
  }

  protected def readSynsetFile(synsetFilePath: String): List[String] = {
    val f = io.Source.fromFile(synsetFilePath)
    val lines = for ( line <- f.getLines()) yield line
    f.close
    lines.toList
  }

}
