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

import org.apache.mxnet.{Context, NDArrayCollector}
import org.apache.mxnetexamples.Util
import org.apache.mxnetexamples.neuralstyle.end2end.{BoostInference, BoostTrain}
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


  override def beforeAll(): Unit = {
    logger.info("Downloading vgg model")
    val tempDirPath = System.getProperty("java.io.tmpdir")
    logger.info("tempDirPath: %s".format(tempDirPath))
    val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/NeuralStyle/"
    Util.downloadUrl(baseUrl + "IMG_4343.jpg", tempDirPath + "/NS/IMG_4343.jpg")
    Util.downloadUrl(baseUrl + "starry_night.jpg", tempDirPath + "/NS/starry_night.jpg")
    Util.downloadUrl(baseUrl + "model.zip", tempDirPath + "/NS/model.zip")
    Util.downloadUrl(baseUrl + "vgg19.params", tempDirPath + "/NS/vgg19.params")
    // TODO: Need to confirm with Windows
    Process(s"unzip $tempDirPath/NS/model.zip -d $tempDirPath/NS/") !

    Process(s"mkdir $tempDirPath/NS/images") !

    for (i <- 0 until 20) {
      Process(s"cp $tempDirPath/NS/IMG_4343.jpg $tempDirPath/NS/images/img$i.jpg") !
    }
  }

  test("Example CI: Test Boost Inference") {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    var ctx = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      ctx = Context.gpu()
    }
    BoostInference.runInference(tempDirPath + "/NS/model", tempDirPath + "/NS", 2,
      tempDirPath + "/NS/IMG_4343.jpg", ctx)
  }

  test("Example CI: Test Boost Training") {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      val ctx = Context.gpu()
      BoostTrain.runTraining(tempDirPath + "/NS/images", tempDirPath + "/NS/vgg19.params", ctx,
        tempDirPath + "/NS/starry_night.jpg", tempDirPath + "/NS")
    } else {
      logger.info("GPU test only, skip CPU...")
    }
  }

  test("Example CI: Test Neural Style") {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      val ctx = Context.gpu()
      NeuralStyle.runTraining("vgg19", tempDirPath + "/NS/IMG_4343.jpg",
        tempDirPath + "/NS/starry_night.jpg",
        ctx, tempDirPath + "/NS/vgg19.params", tempDirPath + "/NS",
        1f, 20f, 0.01f, 1, 10f, 60, 600, 50, 0.0005f)
    } else {
      logger.info("GPU test only, skip CPU")
    }
  }
}
