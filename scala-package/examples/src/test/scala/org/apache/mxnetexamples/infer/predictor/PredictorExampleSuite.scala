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

package org.apache.mxnetexamples.infer.predictor

import java.io.File

import org.apache.mxnet._
import org.apache.mxnetexamples.Util
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

class PredictorExampleSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[PredictorExampleSuite])
  private var modelDirPrefix = ""
  private var inputImagePath = ""
  private var context = Context.cpu()

  override def beforeAll(): Unit = {
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

    modelDirPrefix = tempDirPath + File.separator + "resnet18/resnet-18"
    inputImagePath = tempDirPath + File.separator +
      "inputImages/resnet18/Pug-Cookie.jpg"

    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }
    val props = System.getProperties
    props.setProperty("mxnet.disableShapeCheck", "true")
  }

  override def afterAll(): Unit = {
    val props = System.getProperties
    props.setProperty("mxnet.disableShapeCheck", "false")
  }

  test("test Predictor With Fixed Shape and random shape") {
    val inputDesc = IndexedSeq(new DataDesc("data", Shape(1, 3, 224, 224),
      DType.Float32, Layout.NCHW))
    val predictor = PredictorExample.loadModel(modelDirPrefix, inputDesc, context, 0)
    // fix size
    var img = PredictorExample.preProcess(inputImagePath, 224, 224)
    var result = PredictorExample.doInference(predictor, img)(0)
    var top1 = PredictorExample.postProcess(modelDirPrefix, result.toArray)
    assert(top1 === "n02110958 pug, pug-dog")
    // random size 512
    img = PredictorExample.preProcess(inputImagePath, 512, 512)
    result = PredictorExample.doInference(predictor, img)(0)
    top1 = PredictorExample.postProcess(modelDirPrefix, result.toArray)
    assert(top1 === "n02110958 pug, pug-dog")
    // original size
    img = PredictorExample.preProcess(inputImagePath, 1024, 576)
    result = PredictorExample.doInference(predictor, img)(0)
    top1 = PredictorExample.postProcess(modelDirPrefix, result.toArray)
    assert(top1 === "n02110958 pug, pug-dog")
  }
}
