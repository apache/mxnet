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
package org.apache.mxnetexamples.benchmark

import java.io.File

import org.apache.mxnetexamples.Util
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.language.postfixOps
import scala.sys.process.Process

class ScalaInferenceBenchmarkSuite  extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[ScalaInferenceBenchmarkSuite])
  override def beforeAll(): Unit = {
  }

  test("Testing Benchmark -- Image Classification") {
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
    val modelDirPath = tempDirPath + File.separator + "resnet18/"
    val inputImagePath = tempDirPath + File.separator +
      "inputImages/resnet18/Pug-Cookie.jpg"
    val inputImageDir = tempDirPath + File.separator + "inputImages/resnet18/"
    val args = Array(
      "--example", "ImageClassifierExample",
      "--count", "1",
      "--batchSize", "10",
      "--model-path-prefix", s"$modelDirPath/resnet-18",
      "--input-image", inputImagePath,
      "--input-dir", inputImageDir
    )
    ScalaInferenceBenchmark.main(args)
  }

  test("Testing Benchmark -- Object Detection") {
    logger.info("Downloading resnetssd model")
    val tempDirPath = System.getProperty("java.io.tmpdir")
    logger.info("tempDirPath: %s".format(tempDirPath))
    val modelBase = "https://s3.amazonaws.com/model-server/models/resnet50_ssd/"
    val imageBase = "https://s3.amazonaws.com/model-server/inputs/"
    Util.downloadUrl(modelBase + "resnet50_ssd_model-symbol.json",
      tempDirPath + "/resnetssd/resnet50_ssd_model-symbol.json")
    Util.downloadUrl(modelBase + "resnet50_ssd_model-0000.params",
      tempDirPath + "/resnetssd/resnet50_ssd_model-0000.params")
    Util.downloadUrl(modelBase + "synset.txt",
      tempDirPath + "/resnetssd/synset.txt")
    Util.downloadUrl(imageBase + "dog-ssd.jpg",
      tempDirPath + "/inputImages/resnetssd/dog-ssd.jpg")
    val modelDirPath = tempDirPath + File.separator + "resnetssd/"
    val inputImagePath = tempDirPath + File.separator +
      "inputImages/resnetssd/dog-ssd.jpg"
    val inputImageDir = tempDirPath + File.separator + "inputImages/resnetssd/"
    val args = Array(
      "--example", "ObjectDetectionExample",
      "--count", "1",
      "--batchSize", "10",
      "--model-path-prefix", s"$modelDirPath/resnet50_ssd_model",
      "--input-image", inputImagePath,
      "--input-dir", inputImageDir
    )
    ScalaInferenceBenchmark.main(args)
  }

  test("Testing Benchmark -- charRNN Model") {
    logger.info("Downloading LSTM model")
    val tempDirPath = System.getProperty("java.io.tmpdir")
    logger.info("tempDirPath: %s".format(tempDirPath))
    val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/RNN/"
    Util.downloadUrl(baseUrl + "obama.zip", tempDirPath + "/RNN/obama.zip")
    Util.downloadUrl(baseUrl + "sherlockholmes.train.txt",
      tempDirPath + "/RNN/sherlockholmes.train.txt")
    Util.downloadUrl(baseUrl + "sherlockholmes.valid.txt",
      tempDirPath + "/RNN/sherlockholmes.valid.txt")
    // TODO: Need to confirm with Windows
    Process(s"unzip $tempDirPath/RNN/obama.zip -d $tempDirPath/RNN/") !

    val args = Array(
      "--example", "CharRnn",
      "--count", "1",
      "--data-path", s"$tempDirPath/RNN/obama.txt",
      "--model-prefix", s"$tempDirPath/RNN/obama",
      "--starter-sentence", "The joke"
    )
    ScalaInferenceBenchmark.main(args)
  }

}
