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

package ml.dmlc.mxnetexamples.inferexample.imageclassifier

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import java.io.File
import sys.process._

/**
  * Integration test for imageClassifier example.
  * This will run as a part of "make scalatest"
  */
class ImageClassifierExampleSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[ImageClassifierExampleSuite])

  test("testImageClassifierExample"){
    printf("Downloading resnet-18 model")

    val tempDirPath = System.getProperty("java.io.tmpdir")
    logger.info("tempDirPath: %s".format(tempDirPath))

    "wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json " +
      "-P " + tempDirPath + "resnet18/ -q"!

    "wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params " +
      "-P " + tempDirPath + "resnet18/ -q"!

    "wget http://data.mxnet.io/models/imagenet/resnet/synset.txt -P " + tempDirPath + "resnet18/" +
      " -q"!

    "wget " +
      "http://thenotoriouspug.com/wp-content/uploads/2015/01/Pug-Cookie-1920x1080-1024x576.jpg " +
      "-P " + tempDirPath + "inputImages/"!

    val modelDirPath = tempDirPath + File.separator + "resnet18/"
    val inputImagePath = tempDirPath + File.separator +
      "inputImages/Pug-Cookie-1920x1080-1024x576.jpg"
    val inputImageDir = tempDirPath + File.separator + "inputImages/"

    val output = ImageClassifierExample.runInferenceOnSingleImage(modelDirPath + "resnet-18",
      inputImagePath)

    assert(output(0).toList.head._1 === "n02110958 pug, pug-dog")

    val outputList = ImageClassifierExample.runInferenceOnBatchOfImage(modelDirPath + "resnet-18",
      inputImageDir)

    assert(outputList(0).toList.head._1 === "n02110958 pug, pug-dog")

    "rm -rf " + modelDirPath + " " + inputImageDir!

  }
}
