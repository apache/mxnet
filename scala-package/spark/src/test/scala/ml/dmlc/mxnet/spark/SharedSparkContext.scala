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

package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.{Context, Shape, Symbol}
import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfterAll, FunSuite}

trait SharedSparkContext extends FunSuite with BeforeAndAfterAll {

  protected var sc: SparkContext = _

  override def beforeAll(): Unit = {
    sc = new SparkContext()
  }

  override def afterAll(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  private def getMlp: Symbol = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> 128))
    val act1 = Symbol.Activation(name = "relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> act1, "num_hidden" -> 64))
    val act2 = Symbol.Activation(name = "relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected(name = "fc3")()(Map("data" -> act2, "num_hidden" -> 10))
    val mlp = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc3))
    mlp
  }

  protected def buildMxNet(): Unit = {
    val mxnet = new MXNet()
      .setBatchSize(128)
      .setLabelName("softmax_label")
      .setContext(Array(Context.cpu(0), Context.cpu(1)))
      .setDimension(Shape(784))
      .setNetwork(getMlp)
      .setNumEpoch(10)
      .setNumServer(1)
      .setNumWorker(Runtime.getRuntime.availableProcessors() - 1)
      .setExecutorJars(
        "/Users/nanzhu/code/mxnet/scala-package/spark/target/mxnet-spark_2.11-1.1.0-SNAPSHOT.jar")
      .setJava("java")
  }
}
