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

import java.io.File

import ml.dmlc.mxnet.{Context, Shape, Symbol}
import org.apache.spark.SparkFiles

/**
 * MXNet on Spark training arguments
 * @author Yizhi Liu
 */
private[mxnet] class MXNetParams extends Serializable {
  // training batch size
  var batchSize: Int = 128
  // dimension of input data
  var dimension: Shape = null
  // number of training epochs
  var numEpoch: Int = 10

  // network architecture
  private var network: String = null
  def setNetwork(net: Symbol): Unit = {
    network = net.toJson
  }
  def getNetwork: Symbol = {
    if (network == null) {
      null
    } else {
      Symbol.loadJson(network)
    }
  }

  // executor running context
  var context: Array[Context] = Context.cpu()

  var numWorker: Int = 1
  var numServer: Int = 1

  var dataName: String = "data"
  var labelName: String = "label"

  var timeout: Int = 300

  // jars on executors for running mxnet application
  var jars: Array[String] = null
  def runtimeClasspath: String = {
    jars.map(jar => SparkFiles.get(new File(jar).getName)).mkString(":")
  }

  // java binary
  var javabin: String = "java"
}
