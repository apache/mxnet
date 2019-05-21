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

package sample

import org.apache.mxnet.{Context, DType, DataDesc, Shape}
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import org.apache.mxnet.infer.{ImageClassifier, _}

import scala.collection.JavaConverters._
import java.io.File
import java.net.URL
import org.apache.commons.io._

import scala.collection.mutable.ListBuffer

/**
  * Example showing usage of Infer package to do inference on resnet-18 model
  * Follow instructions in README.md to run this example.
  */
object ImageClassificationExample {

  def downloadUrl(url: String, filePath: String) : Unit = {
    var tmpFile = new File(filePath)
    if (!tmpFile.exists()) {
      FileUtils.copyURLToFile(new URL(url), tmpFile)
    }
  }

  def downloadModelImage() : (String, String) = {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    printf("tempDirPath: %s".format(tempDirPath))
    val imgPath = tempDirPath + "/inputImages/resnet18/Pug-Cookie.jpg"
    val imgURL = "https://s3.amazonaws.com/model-server/inputs/Pug-Cookie.jpg"
    downloadUrl(imgURL, imgPath)

    val baseUrl = "https://s3.us-east-2.amazonaws.com/scala-infer-models"
    var tmpPath = tempDirPath + "/resnet18/resnet-18-symbol.json"
    var tmpUrl = baseUrl + "/resnet-18/resnet-18-symbol.json"
    downloadUrl(tmpUrl, tmpPath)

    tmpPath = tempDirPath + "/resnet18/resnet-18-0000.params"
    tmpUrl = baseUrl + "/resnet-18/resnet-18-0000.params"
    downloadUrl(tmpUrl, tmpPath)

    tmpPath = tempDirPath + "/resnet18/synset.txt"
    tmpUrl = baseUrl + "/resnet-18/synset.txt"
    downloadUrl(tmpUrl, tmpPath)

    (imgPath, tempDirPath + "/resnet18/resnet-18")
  }

  def main(args: Array[String]): Unit = {

    var context = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }
    val (inputImagePath, modelPathPrefix) = downloadModelImage()

    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)
    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    // Create object of ImageClassifier class
    val imgClassifier: ImageClassifier = new
        ImageClassifier(modelPathPrefix, inputDescriptor, context)

    // Loading single image from file and getting BufferedImage
    val img = ImageClassifier.loadImageFromFile(inputImagePath)

    // Running inference on single image
    val output = imgClassifier.classifyImage(img, Some(5))

    // Printing top 5 class probabilities
    for (i <- output) {
      printf("Classes with top 5 probability = %s \n", i)
    }

  }
}
