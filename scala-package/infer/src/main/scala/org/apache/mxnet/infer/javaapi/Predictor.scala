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

import org.apache.mxnet.infer.Predictor
import org.apache.mxnet.javaapi.{Context, DataDesc, NDArray}

import scala.collection.JavaConverters
import scala.collection.JavaConverters._

class Predictor(val predictor: org.apache.mxnet.infer.Predictor){
  def this(modelPathPrevious:String, inputDescriptors: java.util.List[DataDesc], contexts: java.util.List[Context], epoch: Int)
  = this {
    val informationDesc = JavaConverters.asScalaIteratorConverter(inputDescriptors.iterator).asScala.toIndexedSeq map {a=>a: org.apache.mxnet.DataDesc}
    val inContexts = (contexts.asScala.toList map {a => a: org.apache.mxnet.Context}).toArray
    new org.apache.mxnet.infer.Predictor(modelPathPrevious, informationDesc, inContexts, Some(epoch))
  }

  def predict(input: java.util.List[java.util.List[Float]]): java.util.List[java.util.List[Float]] = {
    val in = JavaConverters.asScalaIteratorConverter(input.iterator).asScala.toIndexedSeq
    (predictor.predict(in map {a => a.asScala.toArray}) map {b => b.toList.asJava}).asJava
  }

  def predictWithNDArray(input: java.util.List[NDArray]): java.util.List[NDArray] = {
    val ret = predictor.predictWithNDArray(convert(JavaConverters.asScalaIteratorConverter(input.iterator).asScala.toIndexedSeq))
    // For some reason the implicit wasn't working here when trying to use convert. So did it this way. Needs to be figured out
    (ret map {a => new NDArray(a)}).asJava
  }

  def convert[B, A <% B](l: IndexedSeq[A]): IndexedSeq[B] = l map { a => a: B }
}
