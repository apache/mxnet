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

package org.apache.mxnetexamples.infer.objectdetector

import java.io.File

import org.apache.mxnet.Context
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.sys.process.Process

class ObjectDetectorExampleSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[ObjectDetectorExampleSuite])

  test("testObjectDetectionExample") {
    logger.info("Downloading resnetssd model")
    val tempDirPath = System.getProperty("java.io.tmpdir")

    logger.info("tempDirPath: %s".format(tempDirPath))

    val modelBase = "https://s3.amazonaws.com/model-server/models/resnet50_ssd/"
    val synsetBase = "https://s3.amazonaws.com/model-server/models/resnet50_ssd/"
    val imageBase = "https://s3.amazonaws.com/model-server/inputs/"

    Process("wget " + modelBase + "resnet50_ssd_model-symbol.json " + "-P " +
      tempDirPath + "/resnetssd/ -q") !


    Process("wget " + modelBase + "resnet50_ssd_model-0000.params " +
      "-P " + tempDirPath + "/resnetssd/ -q") !


    Process("wget  " + synsetBase + "synset.txt " + "-P" +
      tempDirPath + "/resnetssd/ -q") !

    Process("wget " +
      imageBase + "dog-ssd.jpg " +
      "-P " + tempDirPath + "/inputImages/") !


    val modelDirPath = tempDirPath + File.separator + "resnetssd/"
    val inputImagePath = tempDirPath + File.separator +
      "inputImages/dog-ssd.jpg"
    val inputImageDir = tempDirPath + File.separator + "inputImages/"

    var context = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }

    val output = SSDClassifierExample.runObjectDetectionSingle(modelDirPath + "resnet50_ssd_model",
      inputImagePath, context)

    assert(output(0)(0)._1 === "car")

    val outputList = SSDClassifierExample.runObjectDetectionBatch(
      modelDirPath + "resnet50_ssd_model",
      inputImageDir, context)

    assert(output(0)(0)._1 === "car")

    Process("rm -rf " + modelDirPath + " " + inputImageDir) !
  }
}
