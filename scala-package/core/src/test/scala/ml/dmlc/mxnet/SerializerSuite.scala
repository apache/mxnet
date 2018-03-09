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

package ml.dmlc.mxnet

import ml.dmlc.mxnet.optimizer.SGD
import org.scalatest.{Matchers, BeforeAndAfterAll, FunSuite}

class SerializerSuite extends FunSuite with BeforeAndAfterAll with Matchers {
  test("serialize and deserialize optimizer") {
    val optimizer: Optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0005f)
    val optSerialized: String = Serializer.encodeBase64String(
      Serializer.getSerializer.serialize(optimizer))
    assert(optSerialized.length > 0)

    val bytes = Serializer.decodeBase64String(optSerialized)
    val optDeserialized = Serializer.getSerializer.deserialize[Optimizer](bytes)

    assert(optDeserialized.isInstanceOf[SGD])
    val sgd = optDeserialized.asInstanceOf[SGD]

    val learningRate = classOf[SGD].getDeclaredField("learningRate")
    learningRate.setAccessible(true)
    assert(learningRate.get(sgd).asInstanceOf[Float] === 0.1f +- 1e-6f)

    val momentum = classOf[SGD].getDeclaredField("momentum")
    momentum.setAccessible(true)
    assert(momentum.get(sgd).asInstanceOf[Float] === 0.9f +- 1e-6f)

    val wd = classOf[SGD].getDeclaredField("wd")
    wd.setAccessible(true)
    assert(wd.get(sgd).asInstanceOf[Float] === 0.0005f +- 1e-6f)
  }
}
