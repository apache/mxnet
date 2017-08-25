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

package ml.dmlc.mxnetexamples.customop

import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.IO
import ml.dmlc.mxnet.DataIter

/**
 * @author Depeng Liang
 */
object Data {
  // return train and val iterators for mnist
  def mnistIterator(dataPath: String, batchSize: Int, inputShape: Shape): (DataIter, DataIter) = {
    val flat = if (inputShape.length == 3) "False" else "True"
    val trainParams = Map(
      "image" -> s"$dataPath/train-images-idx3-ubyte",
      "label" -> s"$dataPath/train-labels-idx1-ubyte",
      "input_shape" -> inputShape.toString(),
      "batch_size" -> s"$batchSize",
      "shuffle" -> "True",
      "flat" -> flat
    )
    val trainDataIter = IO.MNISTIter(trainParams)
    val testParams = Map(
      "image" -> s"$dataPath/t10k-images-idx3-ubyte",
      "label" -> s"$dataPath/t10k-labels-idx1-ubyte",
      "input_shape" -> inputShape.toString(),
      "batch_size" -> s"$batchSize",
      "flat" -> flat
    )
    val testDataIter = IO.MNISTIter(testParams)
    (trainDataIter, testDataIter)
  }
}
