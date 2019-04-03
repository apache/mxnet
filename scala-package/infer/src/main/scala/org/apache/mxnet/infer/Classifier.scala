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

import org.apache.mxnet.{Context, DataDesc, NDArray}
import java.io.File

import org.slf4j.LoggerFactory

import scala.io
import scala.collection.mutable.ListBuffer

trait ClassifierBase {

  /**
    * Takes an array of floats and returns corresponding (Label, Score) tuples
    * @param input            Indexed sequence one-dimensional array of floats
    * @param topK             (Optional) How many result (sorting based on the last axis)
    *                         elements to return. Default returns unsorted output.
    * @return                 Indexed sequence of (Label, Score) tuples
    */
  def classify(input: IndexedSeq[Array[Float]],
               topK: Option[Int] = None): IndexedSeq[(String, Float)]

  /**
    * Takes a sequence of NDArrays and returns (Label, Score) tuples
    * @param input            Indexed sequence of NDArrays
    * @param topK             (Optional) How many result (sorting based on the last axis)
    *                         elements to return. Default returns unsorted output.
    * @return                 Traversable sequence of (Label, Score) tuple
    */
  def classifyWithNDArray(input: IndexedSeq[NDArray],
              topK: Option[Int] = None): IndexedSeq[IndexedSeq[(String, Float)]]
}

/**
  * A class for classifier tasks
  * @param modelPathPrefix    Path prefix from where to load the model artifacts
  *                           These include the symbol, parameters, and synset.txt
  *                           Example: file://model-dir/resnet-152 (containing
  *                           resnet-152-symbol.json, resnet-152-0000.params, and synset.txt)
  * @param inputDescriptors   Descriptors defining the input node names, shape,
  *                           layout and type parameters
  * @param contexts           Device contexts on which you want to run inference; defaults to CPU
  * @param epoch              Model epoch to load; defaults to 0
  */
class Classifier(modelPathPrefix: String,
                 protected val inputDescriptors: IndexedSeq[DataDesc],
                 protected val contexts: Array[Context] = Context.cpu(),
                 protected val epoch: Option[Int] = Some(0))
  extends ClassifierBase {

  private val logger = LoggerFactory.getLogger(classOf[Classifier])

  protected[infer] val predictor: PredictBase = getPredictor()

  protected[infer] val synsetFilePath = getSynsetFilePath(modelPathPrefix)

  protected[infer] val synset = readSynsetFile(synsetFilePath)

  protected[infer] val handler = MXNetHandler()

  /**
    * Takes flat arrays as input and returns (Label, Score) tuples.
    * @param input            Indexed sequence one-dimensional array of floats
    * @param topK             (Optional) How many result (sorting based on the last axis)
    *                         elements to return. Default returns unsorted output.
    * @return                 Indexed sequence of (Label, Score) tuples
    */
  override def classify(input: IndexedSeq[Array[Float]],
                        topK: Option[Int] = None): IndexedSeq[(String, Float)] = {

    // considering only the first output
    val predictResult = predictor.predict(input)(0)
    var result: IndexedSeq[(String, Float)] = IndexedSeq.empty

    if (topK.isDefined) {
      val sortedIndex = predictResult.zipWithIndex.sortBy(-_._1).map(_._2).take(topK.get)
      result = sortedIndex.map(i => (synset(i), predictResult(i))).toIndexedSeq
    } else {
      result = synset.zip(predictResult).toIndexedSeq
    }
    result
  }

  /**
    * Perform multiple classification operations on NDArrays.
    * Also works with batched input.
    * @param input            Indexed sequence of NDArrays
    * @param topK             (Optional) How many result (sorting based on the last axis)
    *                         elements to return. Default returns unsorted output.
    * @return                 Traversable sequence of (Label, Score) tuples
    */
  override def classifyWithNDArray(input: IndexedSeq[NDArray], topK: Option[Int] = None)
  : IndexedSeq[IndexedSeq[(String, Float)]] = {

    // considering only the first output
    val predictResultND: NDArray = predictor.predictWithNDArray(input)(0)

    val predictResult: ListBuffer[Array[Float]] = ListBuffer[Array[Float]]()

    // iterating over the individual items(batch size is in axis 0)
    for (i <- 0 until predictResultND.shape(0)) {
      val r = predictResultND.at(i)
      predictResult += r.toArray
      r.dispose()
    }

    var result: ListBuffer[IndexedSeq[(String, Float)]] =
      ListBuffer.empty[IndexedSeq[(String, Float)]]

    if (topK.isDefined) {
      val sortedIndices = predictResult.map(r =>
        r.zipWithIndex.sortBy(-_._1).map(_._2).take(topK.get)
      )
      for (i <- sortedIndices.indices) {
        result += sortedIndices(i).map(sIndx =>
          (synset(sIndx), predictResult(i)(sIndx))).toIndexedSeq
      }
    } else {
      for (i <- predictResult.indices) {
        result += synset.zip(predictResult(i)).toIndexedSeq
      }
    }

    handler.execute(predictResultND.dispose())

    result.toIndexedSeq
  }

  private[infer] def getSynsetFilePath(modelPathPrefix: String): String = {
    val dirPath = modelPathPrefix.substring(0, 1 + modelPathPrefix.lastIndexOf(File.separator))
    val d = new File(dirPath)
    require(d.exists && d.isDirectory, s"directory: $dirPath not found")

    val s = new File(dirPath + "synset.txt")
    require(s.exists() && s.isFile,
      s"File synset.txt should exist inside modelPath: ${dirPath + "synset.txt"}")

    s.getCanonicalPath
  }

  private[infer]  def readSynsetFile(synsetFilePath: String): IndexedSeq[String] = {
    val f = io.Source.fromFile(synsetFilePath)
    try {
      f.getLines().toIndexedSeq
    } finally {
      f.close
    }
  }

  private[infer] def getPredictor(): PredictBase = {
      new Predictor(modelPathPrefix, inputDescriptors, contexts, epoch)
  }

}
