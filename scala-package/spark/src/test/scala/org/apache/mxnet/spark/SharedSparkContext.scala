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

package org.apache.mxnet.spark

import java.io.{File, FileFilter}

import org.apache.mxnet.{Context, Shape, Symbol}

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfterEach, FunSuite}

trait SharedSparkContext extends FunSuite with BeforeAndAfterEach with BeforeAndAfterAll {

  protected var sc: SparkContext = _

  protected val numWorkers: Int = math.min(Runtime.getRuntime.availableProcessors(), 2)

  override def beforeEach() {
    sc = new SparkContext(new SparkConf().setMaster("local[*]").setAppName("mxnet-spark-test"))
  }

  override def afterEach(): Unit = {
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

  def getLenet: Symbol = {
    val data = Symbol.Variable("data")
    // first conv
    val conv1 = Symbol.Convolution()()(
      Map("data" -> data, "kernel" -> "(5, 5)", "num_filter" -> 20))
    val tanh1 = Symbol.Activation()()(Map("data" -> conv1, "act_type" -> "tanh"))
    val pool1 = Symbol.Pooling()()(Map("data" -> tanh1, "pool_type" -> "max",
      "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    // second conv
    val conv2 = Symbol.Convolution()()(
      Map("data" -> pool1, "kernel" -> "(5, 5)", "num_filter" -> 50))
    val tanh2 = Symbol.Activation()()(Map("data" -> conv2, "act_type" -> "tanh"))
    val pool2 = Symbol.Pooling()()(Map("data" -> tanh2, "pool_type" -> "max",
      "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    // first fullc
    val flatten = Symbol.Flatten()()(Map("data" -> pool2))
    val fc1 = Symbol.FullyConnected()()(Map("data" -> flatten, "num_hidden" -> 500))
    val tanh3 = Symbol.Activation()()(Map("data" -> fc1, "act_type" -> "tanh"))
    // second fullc
    val fc2 = Symbol.FullyConnected()()(Map("data" -> tanh3, "num_hidden" -> 10))
    // loss
    val lenet = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc2))
    lenet
  }

  private def composeWorkingDirPath: String = {
    System.getProperty("user.dir")
  }

  private def getJarFilePath(root: String): String = {
    for (platform <- List("linux-x86_64-cpu", "linux-x86_64-gpu", "osx-x86_64-cpu")) {
      val jarFiles = new File(s"$root/$platform/target/").listFiles(new FileFilter {
        override def accept(pathname: File) = {
          pathname.getAbsolutePath.endsWith(".jar") &&
            !pathname.getAbsolutePath.contains("javadoc") &&
            !pathname.getAbsolutePath.contains("sources")
        }
      })
      if (jarFiles != null && jarFiles.nonEmpty) {
        return jarFiles.head.getAbsolutePath
      }
    }
    null
  }

  private def getSparkJar: String = {
    val jarFiles = new File(s"$composeWorkingDirPath/target/").listFiles(new FileFilter {
      override def accept(pathname: File) = {
        pathname.getAbsolutePath.endsWith(".jar") &&
          !pathname.getAbsolutePath.contains("javadoc") &&
          !pathname.getAbsolutePath.contains("sources")
      }
    })
    if (jarFiles != null && jarFiles.nonEmpty) {
      jarFiles.head.getAbsolutePath
    } else {
      null
    }
  }

  protected def buildLeNet(): MXNet = {
    val workingDir = composeWorkingDirPath
    val assemblyRoot = s"$workingDir/../assembly"
    new MXNet()
      .setBatchSize(128)
      .setLabelName("softmax_label")
      .setContext(Array(Context.cpu(0), Context.cpu(1)))
      .setDimension(Shape(1, 28, 28))
      .setNetwork(getLenet)
      .setNumEpoch(10)
      .setNumServer(1)
      .setNumWorker(numWorkers)
      .setExecutorJars(s"${getJarFilePath(assemblyRoot)},$getSparkJar")
      .setJava("java")
  }

  protected def buildMlp(): MXNet = {
    val workingDir = composeWorkingDirPath
    val assemblyRoot = s"$workingDir/../assembly"
    new MXNet()
      .setBatchSize(128)
      .setLabelName("softmax_label")
      .setContext(Array(Context.cpu(0), Context.cpu(1)))
      .setDimension(Shape(784))
      .setNetwork(getMlp)
      .setNumEpoch(10)
      .setNumServer(1)
      .setNumWorker(numWorkers)
      .setExecutorJars(s"${getJarFilePath(assemblyRoot)},$getSparkJar")
      .setJava("java")
      .setTimeout(0)
  }
}
