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

import java.io.{BufferedReader, File, InputStreamReader}
import java.nio.file.Files

import scala.sys.process.Process

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class MXNetGeneralSuite extends SharedSparkContext {

  private var testDataDir: String = _

  private def parseRawData(sc: SparkContext, path: String): RDD[LabeledPoint] = {
    val raw = sc.textFile(path)
    raw.map { s =>
      val parts = s.split(' ')
      val label = java.lang.Double.parseDouble(parts(0))
      val features = Vectors.dense(parts(1).trim().split(',').map(java.lang.Double.parseDouble))
      LabeledPoint(label, features)
    }
  }

  private def downloadTestData(): Unit = {
    Process("wget https://s3.us-east-2.amazonaws.com/mxnet-scala" +
      "/scala-example-ci/Spark/train_full.txt" + " -P " + testDataDir + " -q") !
  }

//  override def beforeAll(): Unit = {
//  val tempDirFile = Files.createTempDirectory(s"mxnet-spark-test-${System.currentTimeMillis()}").
//      toFile
//    testDataDir = tempDirFile.getPath
//    tempDirFile.deleteOnExit()
//    downloadTestData()
//  }

  test("Dummy test on Spark") {

  }
//  test("run spark with MLP") {
//    val trainData = parseRawData(sc, s"$testDataDir/train_full.txt.txt")
//    val model = buildMlp().fit(trainData)
//    assert(model != null)
//  }
//
//  test("run spark with LeNet") {
//    val trainData = parseRawData(sc, s"$testDataDir/train_full.txt.txt")
//    val model = buildLeNet().fit(trainData)
//    assert(model != null)
//  }
}
