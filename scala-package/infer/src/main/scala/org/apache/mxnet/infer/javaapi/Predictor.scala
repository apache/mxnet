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

import org.apache.mxnet.javaapi.{Context, DataDesc, NDArray}

import scala.collection.JavaConverters
import scala.collection.JavaConverters._

/**
  * Implementation of prediction routines.
  *
  * @param modelPathPrefix     Path prefix from where to load the model artifacts.
  *                            These include the symbol, parameters, and synset.txt
  *                            Example: file://model-dir/resnet-152 (containing
  *                            resnet-152-symbol.json, resnet-152-0000.params, and synset.txt).
  * @param inputDescriptors    Descriptors defining the input node names, shape,
  *                            layout and type parameters
  *                            <p>Note: If the input Descriptors is missing batchSize
  *                            ('N' in layout), a batchSize of 1 is assumed for the model.
  * @param contexts            Device contexts on which you want to run inference; defaults to CPU
  * @param epoch               Model epoch to load; defaults to 0

  */

// JavaDoc description of class to be updated in https://issues.apache.org/jira/browse/MXNET-1178
class Predictor private[mxnet] (val predictor: org.apache.mxnet.infer.Predictor){
  def this(modelPathPrefix: String, inputDescriptors: java.util.List[DataDesc],
           contexts: java.util.List[Context], epoch: Int)
  = this {
    val informationDesc = JavaConverters.asScalaIteratorConverter(inputDescriptors.iterator)
      .asScala.toIndexedSeq map {a => a: org.apache.mxnet.DataDesc}
    val inContexts = (contexts.asScala.toList map {a => a: org.apache.mxnet.Context}).toArray
    new org.apache.mxnet.infer.Predictor(modelPathPrefix, informationDesc, inContexts, Some(epoch))
  }

  /**
    * Takes input as Array of one dimensional arrays and creates the NDArray needed for inference
    * The array will be reshaped based on the input descriptors. Example of calling in Java:
    *
    * <pre>
    * {@code
    * float tmp[][] = new float[1][224];
    * for (int x = 0; x < 1; x++)
    *   for (int y = 0; y < 224; y++)
    *     tmp[x][y] = (int)(Math.random()*10);
    * predictor.predict(tmp);
    * }
    * </pre>
    *
    * @param input:            An Array of a one-dimensional array.
                              An extra Array is needed for when the model has more than one input.
    * @return                  Indexed sequence array of outputs
    */
  def predict(input: Array[Array[Float]]):
  Array[Array[Float]] = {
    predictor.predict(input).toArray
  }

  /**
    * Takes input as Array of one dimensional arrays and creates the NDArray needed for inference
    * The array will be reshaped based on the input descriptors. Example of calling in Java:
    *
    * <pre>
    * {@code
    * double tmp[][] = new double[1][224];
    * for (int x = 0; x < 1; x++)
    *   for (int y = 0; y < 224; y++)
    *     tmp[x][y] = (int)(Math.random()*10);
    * predictor.predict(tmp);
    * }
    * </pre>
    *
    * @param input:            An Array of a one-dimensional array.
                              An extra Array is needed for when the model has more than one input.
    * @return                  Indexed sequence array of outputs
    */

  def predict(input: Array[Array[Double]]):
  Array[Array[Double]] = {
    predictor.predict(input).toArray
  }

  /**
    * Takes input as List of one dimensional arrays and creates the NDArray needed for inference
    * The array will be reshaped based on the input descriptors.
    *
    * @param input:            A List of a one-dimensional array.
                              An extra List is needed for when the model has more than one input.
    * @return                  Indexed sequence array of outputs
    */
  def predict(input: java.util.List[java.util.List[java.lang.Float]]):
  java.util.List[java.util.List[java.lang.Float]] = {
    val in = JavaConverters.asScalaIteratorConverter(input.iterator).asScala.toIndexedSeq
    (predictor.predict(in map {a => a.asScala.map(Float2float).toArray})
      map {b => b.map(float2Float).toList.asJava}).asJava
  }



  /**
    * Predict using NDArray as input
    * This method is useful when the input is a batch of data
    * Note: User is responsible for managing allocation/deallocation of input/output NDArrays.
    *
    * @param input             List of NDArrays
    * @return                  Output of predictions as NDArrays
    */
  def predictWithNDArray(input: java.util.List[NDArray]):
  java.util.List[NDArray] = {
    val ret = predictor.predictWithNDArray(convert(JavaConverters
      .asScalaIteratorConverter(input.iterator).asScala.toIndexedSeq))
    // TODO: For some reason the implicit wasn't working here when trying to use convert.
    // So did it this way. Needs to be figured out
    (ret map {a => new NDArray(a)}).asJava
  }

  private def convert[B, A <% B](l: IndexedSeq[A]): IndexedSeq[B] = l map { a => a: B }
}
