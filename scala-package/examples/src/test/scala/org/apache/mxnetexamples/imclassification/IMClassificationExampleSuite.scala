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

package org.apache.mxnetexamples.imclassification

import java.io.File

import org.apache.mxnet.{Context, DType, ResourceScope}
import org.apache.mxnetexamples.Util
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.language.postfixOps
import scala.sys.process.Process

/**
  * Integration test for MNIST example.
  */
class IMClassificationExampleSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[IMClassificationExampleSuite])

  test("Example CI: Test MNIST Training") {

    ResourceScope.using() {
      logger.info("Downloading mnist model")
      val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci"
      val tempDirPath = System.getProperty("java.io.tmpdir")
      val modelDirPath = tempDirPath + File.separator + "mnist/"
      logger.info("tempDirPath: %s".format(tempDirPath))
      Util.downloadUrl(baseUrl + "/mnist/mnist.zip",
        tempDirPath + "/mnist/mnist.zip")
      // TODO: Need to confirm with Windows
      Process("unzip " + tempDirPath + "/mnist/mnist.zip -d "
        + tempDirPath + "/mnist/") !

      var context = Context.cpu()

      val valAccuracy = TrainModel.test("mlp", modelDirPath)
      Process("rm -rf " + modelDirPath) !

      assert(valAccuracy >= 0.95f)
    }
  }

  for(model <- List("mlp", "lenet", "resnet")) {
    test(s"Example CI: Test Image Classification Model ${model}") {
      ResourceScope.using() {
        val valAccuracy = TrainModel.test(model, "", 10, 1, benchmark = true)
      }
    }
  }

  for(model <- List("mlp", "lenet", "resnet")) {
    test(s"Example CI: Test Image Classification Model ${model} with Float64 input") {
      ResourceScope.using() {
        val valAccuracy = TrainModel.test(model, "", 10, 1, benchmark = true,
          dtype = DType.Float64)
      }
    }
  }

}
