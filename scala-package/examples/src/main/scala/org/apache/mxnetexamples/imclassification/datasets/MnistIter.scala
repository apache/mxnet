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

package org.apache.mxnetexamples.imclassification.datasets

import org.apache.mxnet._

object MnistIter {
  /**
    * Returns an iterator over the MNIST dataset
    * @param dataShape Image size (channels, height, width)
    * @param dataDir The path to the image data
    * @param batchSize Number of images per batch
    * @param kv KVStore to use
    * @return
    */
  def getIterator(dataShape: Shape, dataDir: String)
                 (batchSize: Int, kv: KVStore): (DataIter, DataIter) = {
    val flat = if (dataShape.size == 3) "False" else "True"

    val train = IO.MNISTIter(Map(
      "image" -> (dataDir + "train-images-idx3-ubyte"),
      "label" -> (dataDir + "train-labels-idx1-ubyte"),
      "label_name" -> "softmax_label",
      "input_shape" -> dataShape.toString,
      "batch_size" -> batchSize.toString,
      "shuffle" -> "True",
      "flat" -> flat,
      "num_parts" -> kv.numWorkers.toString,
      "part_index" -> kv.`rank`.toString))

    val eval = IO.MNISTIter(Map(
      "image" -> (dataDir + "t10k-images-idx3-ubyte"),
      "label" -> (dataDir + "t10k-labels-idx1-ubyte"),
      "label_name" -> "softmax_label",
      "input_shape" -> dataShape.toString,
      "batch_size" -> batchSize.toString,
      "flat" -> flat,
      "num_parts" -> kv.numWorkers.toString,
      "part_index" -> kv.`rank`.toString))

    (train, eval)
  }

}
