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

package org.apache.mxnetexamples.infer.imageclassifier

import org.apache.mxnet._
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import org.apache.mxnet.infer.{Classifier, ImageClassifier}

import scala.collection.JavaConverters._
import java.io.File

import org.apache.mxnetexamples.benchmark.CLIParserBase
// scalastyle:off
import java.awt.image.BufferedImage
// scalastyle:on

import org.apache.mxnetexamples.InferBase

import scala.collection.mutable.ListBuffer

// scalastyle:off
/**
  * <p>
  * Example inference showing usage of the Infer package on a resnet-152 model.
  * @see <a href="https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/imageclassifier" target="_blank">Instructions to run this example</a>
  */
// scalastyle:on
object ImageClassifierExample {

  private val logger = LoggerFactory.getLogger(classOf[ImageClassifierExample])


  def runInferenceOnSingleImage(modelPathPrefix: String, inputImagePath: String,
                                context: Array[Context]):
  IndexedSeq[IndexedSeq[(String, Float)]] = {
    NDArrayCollector.auto().withScope {
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
  }

  def runInferenceOnBatchOfImage(modelPathPrefix: String, inputImageDir: String,
                                 context: Array[Context]):
  IndexedSeq[IndexedSeq[(String, Float)]] = {
    NDArrayCollector.auto().withScope {
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
    val inst = new CLIParser
    val parser: CmdLineParser = new CmdLineParser(inst)

    var context = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }

    try {
      parser.parseArgument(args.toList.asJava)


      val modelPathPrefix = if (inst.modelPathPrefix == null) System.getenv("MXNET_HOME")
      else inst.modelPathPrefix

      val inputImagePath = if (inst.inputImagePath == null) System.getenv("MXNET_HOME")
      else inst.inputImagePath

      val inputImageDir = if (inst.inputImageDir == null) System.getenv("MXNET_HOME")
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

class CLIParser extends CLIParserBase{
  @Option(name = "--model-path-prefix", usage = "the input model directory")
  val modelPathPrefix: String = "/resnet-152/resnet-152"
  @Option(name = "--input-image", usage = "the input image")
  val inputImagePath: String = "/images/kitten.jpg"
  @Option(name = "--input-dir", usage = "the input batch of images directory")
  val inputImageDir: String = "/images/"
}

class ImageClassifierExample(CLIParser: CLIParser) extends InferBase{

  override def loadModel(context: Array[Context],
                         batchInference : Boolean = false): Classifier = {
    val dType = DType.Float32
    val batchSize = if (batchInference) CLIParser.batchSize else 1
    val inputShape = Shape(batchSize, 3, 224, 224)

    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    // Create object of ImageClassifier class
    val imgClassifier: ImageClassifier = new ImageClassifier(CLIParser.modelPathPrefix,
      inputDescriptor, context)
    imgClassifier
  }

  override def loadSingleData(): Any = {
    val img = ImageClassifier.loadImageFromFile(CLIParser.inputImagePath)
    img
  }

  override def loadBatchFileList(batchSize: Int): List[Any] = {
    val dir = new File(CLIParser.inputImageDir)
    require(dir.exists && dir.isDirectory,
      "input image directory: %s not found".format(CLIParser.inputImageDir))
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

  override def loadInputBatch(inputPaths: Any): Any = {
    val batchFile = inputPaths.asInstanceOf[List[String]]
    ImageClassifier.loadInputBatch(batchFile)
  }

  override def runSingleInference(loadedModel: Any, input: Any): Any = {
    // Running inference on single image
    val imageModel = loadedModel.asInstanceOf[ImageClassifier]
    val imgInput = input.asInstanceOf[BufferedImage]
    val output = imageModel.classifyImage(imgInput, Some(5))
    output
  }

  override def runBatchInference(loadedModel: Any, input: Any): Any = {
    val imageModel = loadedModel.asInstanceOf[ImageClassifier]
    val imgInput = input.asInstanceOf[Traversable[BufferedImage]]
    val output = imageModel.classifyImageBatch(imgInput, Some(5))
    output
  }

}
