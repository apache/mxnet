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

import org.apache.mxnet._

object Lenet {

  /**
    * Gets Lenet Model Symbol
    * @param numClasses Number of classes to classify into
    * @return model symbol
    */
  def getSymbol(numClasses: Int): Symbol = {
    val data = Symbol.Variable("data")
    // first conv
    val conv1 = Symbol.api.Convolution(data = Some(data), kernel = Shape(5, 5), num_filter = 20)
    val tanh1 = Symbol.api.tanh(data = Some(conv1))
    val pool1 = Symbol.api.Pooling(data = Some(tanh1), pool_type = Some("max"),
      kernel = Some(Shape(2, 2)), stride = Some(Shape(2, 2)))
    // second conv
    val conv2 = Symbol.api.Convolution(data = Some(pool1), kernel = Shape(5, 5), num_filter = 50)
    val tanh2 = Symbol.api.tanh(data = Some(conv2))
    val pool2 = Symbol.api.Pooling(data = Some(tanh2), pool_type = Some("max"),
      kernel = Some(Shape(2, 2)), stride = Some(Shape(2, 2)))
    // first fullc
    val flatten = Symbol.api.Flatten(data = Some(pool2))
    val fc1 = Symbol.api.FullyConnected(data = Some(flatten), num_hidden = 500)
    val tanh3 = Symbol.api.tanh(data = Some(fc1))
    // second fullc
    val fc2 = Symbol.api.FullyConnected(data = Some(tanh3), num_hidden = numClasses)
    // loss
    val lenet = Symbol.api.SoftmaxOutput(name = "softmax", data = Some(fc2))
    lenet
  }

}
