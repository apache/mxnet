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

package org.apache.mxnet.examples.imclassification

import java.io.File

import org.apache.mxnet.Context
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.sys.process.Process

/**
  * Integration test for imageClassifier example.
  * This will run as a part of "make scalatest"
  */
class MNISTExampleSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[MNISTExampleSuite])

  test("Example CI: Test MNIST Training") {
    // This test is CPU only
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      logger.info("CPU test only, skipped...")
    } else {
      logger.info("Downloading mnist model")
      val tempDirPath = System.getProperty("java.io.tmpdir")
      val modelDirPath = tempDirPath + File.separator + "mnist/"
      logger.info("tempDirPath: %s".format(tempDirPath))
      Process("wget https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci" +
        "/mnist/mnist.zip " + "-P " + tempDirPath + "/mnist/ -q") !

      Process("unzip " + tempDirPath + "/mnist/mnist.zip -d "
        + tempDirPath + "/mnist/") !

      var context = Context.cpu()

      val output = TrainMnist.test(modelDirPath)
      Process("rm -rf " + modelDirPath) !

      assert(output >= 0.95f)
    }

  }
}
