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

package org.apache.mxnetexamples.infer.imageclassifier

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory
import java.io.File
import org.apache.mxnet.Context
import org.apache.mxnet.ResourceScope
import org.apache.mxnetexamples.Util

import scala.language.postfixOps
import sys.process.Process

/**
  * Integration test for imageClassifier example.
  */
class ImageClassifierExampleSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[ImageClassifierExampleSuite])

  test("testImageClassifierExample") {
    logger.info("Downloading resnet-18 model")

    val tempDirPath = System.getProperty("java.io.tmpdir")
    logger.info("tempDirPath: %s".format(tempDirPath))

    val baseUrl = "https://s3.us-east-2.amazonaws.com/scala-infer-models"

    Util.downloadUrl(baseUrl + "/resnet-18/resnet-18-symbol.json",
      tempDirPath + "/resnet18/resnet-18-symbol.json")
    Util.downloadUrl(baseUrl + "/resnet-18/resnet-18-0000.params",
      tempDirPath + "/resnet18/resnet-18-0000.params")
    Util.downloadUrl(baseUrl + "/resnet-18/synset.txt",
      tempDirPath + "/resnet18/synset.txt")
    Util.downloadUrl("https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg",
      tempDirPath + "/inputImages/resnet18/Pug-Cookie.jpg")

    ResourceScope.using() {
      val modelDirPath = tempDirPath + File.separator + "resnet18/"
      val inputImagePath = tempDirPath + File.separator +
        "inputImages/resnet18/Pug-Cookie.jpg"
      val inputImageDir = tempDirPath + File.separator + "inputImages/resnet18/"

      var context = Context.cpu()
      if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
            System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
        context = Context.gpu()
      }

      val output = ImageClassifierExample.runInferenceOnSingleImage(modelDirPath + "resnet-18",
                                                                    inputImagePath, context)

      val outputList = ImageClassifierExample.runInferenceOnBatchOfImage(modelDirPath + "resnet-18",
                                                                         inputImageDir, context)

      Process("rm -rf " + modelDirPath + " " + inputImageDir) !

      assert(output(0).toList.head._1 === "n02110958 pug, pug-dog")
      assert(outputList(0).toList.head._1 === "n02110958 pug, pug-dog")
    }

  }
}
