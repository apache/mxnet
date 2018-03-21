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

package ml.dmlc.mxnetexamples.inferexample

import ml.dmlc.mxnet._
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import ml.dmlc.mxnet.{DType, DataDesc}
import ml.dmlc.mxnet.infer._

import scala.collection.JavaConverters._

object ImageClassifierExample {
  private val logger = LoggerFactory.getLogger(classOf[ImageClassifierExample])

  def runInference(modelPathPrefix: String, inputImagePath: String, inputImageDir: String):
  IndexedSeq[IndexedSeq[(String, Float)]] = {
    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)

    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    val imgClassifier: ImageClassifier = new
        ImageClassifier(modelPathPrefix, inputDescriptor)

    val img = ImageClassifier.loadImageFromFile(inputImagePath)

    val output = imgClassifier.classifyImage(img, Some(5))

    for (i <- output) {
      printf("Class with probability=%s \n", i)
    }

    val imgList = ImageClassifier.loadInputBatch(inputImageDir)
    val outputList = imgClassifier.classifyImageBatch(imgList, Some(1))

    for (i <- outputList) {
      printf("Class with probability=%s \n", i)
    }
    output
  }

  def main(args: Array[String]): Unit = {
    val inst = new ImageClassifierExample
    val parser: CmdLineParser = new CmdLineParser(inst)
    try {
      parser.parseArgument(args.toList.asJava)

      val modelPathPrefix = if (inst.modelPathPrefix == null) System.getenv("MXNET_DATA_DIR")
      else inst.modelPathPrefix

      val inputImagePath = if (inst.inputImagePath == null) System.getenv("MXNET_DATA_DIR")
      else inst.inputImagePath

      val inputImageDir = if (inst.inputImageDir == null) System.getenv("MXNET_DATA_DIR")
      else inst.inputImageDir

      runInference(modelPathPrefix, inputImagePath, inputImageDir)

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class ImageClassifierExample {
  @Option(name = "--model-dir", usage = "the input model directory")
  private val modelPathPrefix: String = "/resnet/resnet-152"
  @Option(name = "--input-image", usage = "the input image")
  private val inputImagePath: String = "/images/Cat-hd-wallpapers.jpg"
  @Option(name = "--input-dir", usage = "the input batch of images directory")
  private val inputImageDir: String = "/images/"
}
