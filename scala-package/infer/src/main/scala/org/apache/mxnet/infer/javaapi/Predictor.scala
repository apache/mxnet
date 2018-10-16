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

class Predictor(val predictor: org.apache.mxnet.infer.Predictor){
  def this(modelPathPrefix: String, inputDescriptors: java.util.List[DataDesc],
           contexts: java.util.List[Context], epoch: Int)
  = this {
    val informationDesc = JavaConverters.asScalaIteratorConverter(inputDescriptors.iterator)
      .asScala.toIndexedSeq map {a => a: org.apache.mxnet.DataDesc}
    val inContexts = (contexts.asScala.toList map {a => a: org.apache.mxnet.Context}).toArray
    new org.apache.mxnet.infer.Predictor(modelPathPrefix, informationDesc, inContexts, Some(epoch))
  }


  /**
    * Takes input as List of one dimensional arrays and creates the NDArray needed for inference
    * The array will be reshaped based on the input descriptors.
    *
    * @param input:            A List of a one-dimensional array.
                              An extra List is needed for when the model has more than one input.
    * @return                  Indexed sequence array of outputs
    */
  def predict(input: java.util.List[java.util.List[Float]]):
  java.util.List[java.util.List[Float]] = {
    val in = JavaConverters.asScalaIteratorConverter(input.iterator).asScala.toIndexedSeq
    (predictor.predict(in map {a => a.asScala.toArray}) map {b => b.toList.asJava}).asJava
  }


  /**
    * Predict using NDArray as input
    * This method is useful when the input is a batch of data
    * Note: User is responsible for managing allocation/deallocation of input/output NDArrays.
    *
    * @param input       List of NDArrays
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
