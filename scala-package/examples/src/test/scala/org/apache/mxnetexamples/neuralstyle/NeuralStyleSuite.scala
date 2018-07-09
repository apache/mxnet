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

package org.apache.mxnetexamples.neuralstyle

import java.io.File
import java.net.URL

import org.apache.commons.io.FileUtils
import org.apache.mxnet.Context
import org.apache.mxnetexamples.neuralstyle.end2end.BoostInference
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.sys.process.Process

/**
  * Neural Suite Test package
  * Currently there is no plan to run to test accuracy
  * This test is just to verify the model is runnable
  */
class NeuralStyleSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[NeuralStyleSuite])

  def downloadUrl(url: String, filePath: String) : Unit = {
    val tmpFile = new File(filePath)
    if (!tmpFile.exists()) {
      FileUtils.copyURLToFile(new URL(url), tmpFile)
    }
  }

  test("Example CI: Test Boost Inference") {
    logger.info("Downloading vgg model")
    val tempDirPath = System.getProperty("java.io.tmpdir")
    logger.info("tempDirPath: %s".format(tempDirPath))
    val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/"
    downloadUrl(baseUrl + "IMG_4343.jpg", tempDirPath + "/NS/IMG_4343.jpg")
    downloadUrl(baseUrl + "model.zip", tempDirPath + "/NS/model.zip")
    var ctx = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      ctx = Context.gpu()
    }

    // TODO: Need to confirm with Windows
    Process("unzip " + tempDirPath + "/NS/model.zip -d "
      + tempDirPath + "/NS/") !

    BoostInference.runInference(tempDirPath + "/NS/model", tempDirPath + "/NS", 2,
      tempDirPath + "/NS/IMG_4343.jpg", ctx)
  }
}
