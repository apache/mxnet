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

import ml.dmlc.mxnet.Shape
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import ml.dmlc.mxnet.{DType, DataDesc}
import ml.dmlc.mxnet.infer.ImageClassifier

import scala.collection.JavaConverters._
import java.io.File

/**
  * Example showing usage of Infer package to do inference on resnet-152 model
  * Follow instructions in README.md to run this example.
  */
object ImageClassifierExample {
  private val logger = LoggerFactory.getLogger(classOf[ImageClassifierExample])

  def runInferenceOnSingleImage(modelPathPrefix: String, inputImagePath: String):
  IndexedSeq[IndexedSeq[(String, Float)]] = {
    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)

    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    // Create object of ImageClassifier class
    val imgClassifier: ImageClassifier = new
        ImageClassifier(modelPathPrefix, inputDescriptor)

    // Loading single image from file and getting BufferedImage
    val img = ImageClassifier.loadImageFromFile(inputImagePath)

    // Running inference on single image
    val output = imgClassifier.classifyImage(img, Some(5))

    output
  }

  def runInferenceOnBatchOfImage(modelPathPrefix: String, inputImageDir: String):
  IndexedSeq[IndexedSeq[(String, Float)]] = {
    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)

    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    // Create object of ImageClassifier class
    val imgClassifier: ImageClassifier = new
        ImageClassifier(modelPathPrefix, inputDescriptor)

    // Loading batch of images from the directory path
    val imgList = ImageClassifier.loadInputBatch(inputImageDir)

    // Running inference on batch of images loaded in previous step
    val outputList = imgClassifier.classifyImageBatch(imgList, Some(5))

    outputList
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

      val singleOutput = runInferenceOnSingleImage(modelPathPrefix, inputImagePath)

      // Printing top 5 class probabilities
      for (i <- singleOutput) {
        printf("Classes with top 5 probability = %s \n", i)
      }

      val batchOutput = runInferenceOnBatchOfImage(modelPathPrefix, inputImageDir)

      val d = new File(inputImageDir)
      val filenames = d.listFiles.filter(_.isFile).toList

      // Printing filename and inference class with top 5 probabilities
      for ((f, inferOp) <- (filenames zip batchOutput)) {
        printf("Input image %s ", f)
        printf("Class with probability =%s \n", inferOp)
      }
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
  @Option(name = "--model-path-prefix", usage = "the input model directory")
  private val modelPathPrefix: String = "/resnet-152/resnet-152"
  @Option(name = "--input-image", usage = "the input image")
  private val inputImagePath: String = "/images/kitten.jpg"
  @Option(name = "--input-dir", usage = "the input batch of images directory")
  private val inputImageDir: String = "/images/"
}
