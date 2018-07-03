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

import scala.collection.mutable.ListBuffer

/**
  * Example showing usage of Infer package to do inference on resnet-152 model
  * Follow instructions in README.md to run this example.
  */
object ImageClassificationExample {
  private val logger = LoggerFactory.getLogger(classOf[ImageClassificationExample])

  def runInferenceOnSingleImage(modelPathPrefix: String, inputImagePath: String,
                                context: Array[Context]):
  IndexedSeq[IndexedSeq[(String, Float)]] = {
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

    output
  }

  def runInferenceOnBatchOfImage(modelPathPrefix: String, inputImageDir: String,
                                 context: Array[Context]):
  IndexedSeq[IndexedSeq[(String, Float)]] = {
    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)

    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    // Create object of ImageClassifier class
    val imgClassifier: ImageClassifier = new
        ImageClassifier(modelPathPrefix, inputDescriptor, context)

    // Loading batch of images from the directory path
    val batchFiles = generateBatches(inputImageDir, 20)
    var outputList = IndexedSeq[IndexedSeq[(String, Float)]]()

    for (batchFile <- batchFiles) {
      val imgList = ImageClassifier.loadInputBatch(batchFile)
      // Running inference on batch of images loaded in previous step
      outputList ++= imgClassifier.classifyImageBatch(imgList, Some(5))
    }

    outputList
  }

  def generateBatches(inputImageDirPath: String, batchSize: Int = 100): List[List[String]] = {
    val dir = new File(inputImageDirPath)
    require(dir.exists && dir.isDirectory,
      "input image directory: %s not found".format(inputImageDirPath))
    val output = ListBuffer[List[String]]()
    var batch = ListBuffer[String]()
    for (imgFile: File <- dir.listFiles()){
      batch += imgFile.getPath
      if (batch.length == batchSize) {
        output += batch.toList
        batch = ListBuffer[String]()
      }
    }
    if (batch.length > 0) {
      output += batch.toList
    }
    output.toList
  }

  def main(args: Array[String]): Unit = {
    val inst = new ImageClassificationExample
    val parser: CmdLineParser = new CmdLineParser(inst)

    var context = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }

    try {
      parser.parseArgument(args.toList.asJava)


      val modelPathPrefix = if (inst.modelPathPrefix == null) System.getenv("MXNET_DATA_DIR")
      else inst.modelPathPrefix

      val inputImagePath = if (inst.inputImagePath == null) System.getenv("MXNET_DATA_DIR")
      else inst.inputImagePath

      val inputImageDir = if (inst.inputImageDir == null) System.getenv("MXNET_DATA_DIR")
      else inst.inputImageDir

      val singleOutput = runInferenceOnSingleImage(modelPathPrefix, inputImagePath, context)

      // Printing top 5 class probabilities
      for (i <- singleOutput) {
        printf("Classes with top 5 probability = %s \n", i)
      }

      val batchOutput = runInferenceOnBatchOfImage(modelPathPrefix, inputImageDir, context)

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

class ImageClassificationExample {
  @Option(name = "--model-path-prefix", usage = "the input model directory")
  private val modelPathPrefix: String = "/Users/roshanin/Downloads/resnet/resnet-152"
  @Option(name = "--input-image", usage = "the input image")
  private val inputImagePath: String = "/Users/roshanin/Downloads/kitten.jpg"
  @Option(name = "--input-dir", usage = "the input batch of images directory")
  private val inputImageDir: String = "/Users/roshanin/Downloads/images/"
  @Option(name = "--num-run", usage = "number of times to run inference")
  private val numRun: Int = 1
}