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

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.{DType}

import org.scalatest.{BeforeAndAfterAll, FunSuite}
import sys.process._

/**
  * Integration test for imageClassifier example.
  * This will run as a part of "make scalatest"
  */
class ImageClassifierExampleSuite extends FunSuite with BeforeAndAfterAll {

  test("testImageClassifierExample"){
    printf("Downloading resnet-18 model")

    "wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json " +
      "-P /tmp/resnet18/ -q --show-progress"!

    "wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params " +
      "-P /tmp/resnet18/ -q --show-progress"!

    "wget http://data.mxnet.io/models/imagenet/resnet/synset.txt -P /tmp/resnet18/" +
      " -q --show-progress"!

    "wget " +
      "http://thenotoriouspug.com/wp-content/uploads/2015/01/Pug-Cookie-1920x1080-1024x576.jpg " +
      "-P /tmp/inputImages/"!

    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)

    val output = ImageClassifierExample.runInferenceOnSingleImage("/tmp/resnet18/resnet-18",
     "/tmp/inputImages/Pug-Cookie-1920x1080-1024x576.jpg")

    assert(output(0).toList.head._1 === "n02110958 pug, pug-dog")

    val outputList = ImageClassifierExample.runInferenceOnBatchOfImage("/tmp/resnet18/resnet-18",
    "/tmp/inputImages/")

    assert(outputList(0).toList.head._1 === "n02110958 pug, pug-dog")

    "rm -rf /tmp/resnet18/ /tmp/inputImages/"!

  }
}
