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
package org.apache.mxnetexamples.customop

import java.io.File
import java.net.URL

import org.apache.commons.io.FileUtils
import org.apache.mxnet.Context
import org.apache.mxnet.ResourceScope;
import org.apache.mxnetexamples.Util
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.language.postfixOps
import scala.sys.process.Process

class CustomOpExampleSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[CustomOpExampleSuite])

  test("Example CI: Test Customop MNIST") {
    // This test is CPU only
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      logger.info("CPU test only, skipped...")
    } else {
      ResourceScope.using() {
        logger.info("Downloading mnist model")
        val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci"
        val tempDirPath = System.getProperty("java.io.tmpdir")
        val modelDirPath = tempDirPath + File.separator + "mnist/"
        val tmpFile = new File(tempDirPath + "/mnist/mnist.zip")
        if (!tmpFile.exists()) {
          FileUtils.copyURLToFile(new URL(baseUrl + "/mnist/mnist.zip"),
                                  tmpFile)
        }
        // TODO: Need to confirm with Windows
        Process("unzip " + tempDirPath + "/mnist/mnist.zip -d "
                  + tempDirPath + "/mnist/") !
        val context = Context.cpu()
        val output = ExampleCustomOp.test(modelDirPath, context)
        assert(output >= 0.95f)
      }
    }
  }

  test("Example CI: Test CustomopRtc MNIST") {
    // This test is GPU only
    // TODO: RTC is depreciated, need to change to CUDA Module
    val RTC_fixed = false
    if (RTC_fixed) {
      if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
        System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
        ResourceScope.using() {
          logger.info("Downloading mnist model")
          val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci"
          val tempDirPath = System.getProperty("java.io.tmpdir")
          val modelDirPath = tempDirPath + File.separator + "mnist/"
          Util.downloadUrl(baseUrl + "/mnist/mnist.zip",
                           tempDirPath + "/mnist/mnist.zip")
          // TODO: Need to confirm with Windows
          Process("unzip " + tempDirPath + "/mnist/mnist.zip -d "
                    + tempDirPath + "/mnist/") !
          val context = Context.gpu()
          val output = ExampleCustomOpWithRtc.test(modelDirPath, context)
          assert(output >= 0.95f)
        }
      } else {
        logger.info("GPU test only, skipped...")
      }
    } else {
      logger.warn("RTC module is not up to date, please don't use this" +
      "\nCreate CudaModule for this")
    }
  }
}
