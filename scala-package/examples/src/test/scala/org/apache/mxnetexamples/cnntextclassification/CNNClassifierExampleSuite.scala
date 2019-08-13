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

package org.apache.mxnetexamples.cnntextclassification

import java.io.File
import java.net.URL

import org.apache.commons.io.FileUtils
import org.apache.mxnet.Context
import org.apache.mxnetexamples.Util
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.language.postfixOps
import scala.sys.process.Process

/**
  * Integration test for CNN example.
  */
class CNNClassifierExampleSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[CNNClassifierExampleSuite])

  test("Example CI - CNN Example") {

    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      val context = Context.gpu()
      val tempDirPath = System.getProperty("java.io.tmpdir")
      val w2vModelName = "GoogleNews-vectors-negative300-SLIM.bin"

      logger.info("tempDirPath: %s".format(tempDirPath))

      logger.info("Downloading CNN text...")
      val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala"
      Util.downloadUrl(baseUrl + "/scala-example-ci/CNN/rt-polarity.pos",
        tempDirPath + "/CNN/rt-polarity.pos")
      Util.downloadUrl(baseUrl + "/scala-example-ci/CNN/rt-polarity.neg",
        tempDirPath + "/CNN/rt-polarity.neg")
      logger.info("Downloading pretrianed Word2Vec Model, may take a while")
      Util.downloadUrl(baseUrl + "/scala-example-ci/CNN/" + w2vModelName,
        tempDirPath + "/CNN/" + w2vModelName)

      val modelDirPath = tempDirPath + File.separator + "CNN"

      val output = CNNTextClassification.test(modelDirPath + File.separator + w2vModelName,
        modelDirPath, context, modelDirPath)

      Process("rm -rf " + modelDirPath) !

      assert(output >= 0.4f)
    } else {
      logger.info("Skip this test as it intended for GPU only")
    }
  }
}
