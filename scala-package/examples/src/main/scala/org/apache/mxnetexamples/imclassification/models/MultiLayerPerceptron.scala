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

package org.apache.mxnetexamples.imclassification.models

import org.apache.mxnet.DType.DType
import org.apache.mxnet._

object MultiLayerPerceptron {

  /**
    * Gets MultiLayer Perceptron Model Symbol
    * @param numClasses Number of classes to classify into
    * @return model symbol
    */
  def getSymbol(numClasses: Int, dtype: DType = DType.Float32): Symbol = {
    val data = Symbol.Variable("data", dType = dtype)

    val fc1 = Symbol.api.FullyConnected(data = Some(data), num_hidden = 128, name = "fc1")
    val act1 = Symbol.api.Activation(data = Some(fc1), "relu", name = "relu")
    val fc2 = Symbol.api.FullyConnected(Some(act1), None, None, 64, name = "fc2")
    val act2 = Symbol.api.Activation(data = Some(fc2), "relu", name = "relu2")
    val fc3 = Symbol.api.FullyConnected(Some(act2), None, None, numClasses, name = "fc3")
    val mlp = Symbol.api.SoftmaxOutput(name = "softmax", data = Some(fc3))
    mlp
  }

}
