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
// scalastyle:off
import java.awt.image.BufferedImage

import org.apache.mxnetexamples.benchmark.CLIParserBase
// scalastyle:on
import java.io.File

import org.apache.mxnet._
import org.apache.mxnet.infer._
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import java.nio.file.{Files, Paths}

import org.apache.mxnetexamples.InferBase

import scala.collection.mutable.ListBuffer

// scalastyle:off
/**
  * <p>
  * Example single shot detector (SSD) using the Infer package
  * on a ssd_resnet50_512 model.
  * @see <a href="https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer/objectdetector" target="_blank">Instructions to run this example</a>
  */
// scalastyle:on
object SSDClassifierExample {

  private val logger = LoggerFactory.getLogger(classOf[SSDClassifierExample])
  private type SSDOut = (String, Array[Float])

  def runObjectDetectionSingle(modelPathPrefix: String, inputImagePath: String,
                               context: Array[Context]):
  IndexedSeq[IndexedSeq[(String, Array[Float])]] = {
    ResourceScope.using() {
      val dType = DType.Float32
      val inputShape = Shape(1, 3, 512, 512)
      // ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
      val outputShape = Shape(1, 6132, 6)
      val inputDescriptors = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))
      val img = ImageClassifier.loadImageFromFile(inputImagePath)
      val objDetector = new ObjectDetector(modelPathPrefix, inputDescriptors, context)
      val output = objDetector.imageObjectDetect(img, Some(3))

      output
    }
  }

  def runObjectDetectionBatch(modelPathPrefix: String, inputImageDir: String,
                              context: Array[Context]):
  IndexedSeq[IndexedSeq[(String, Array[Float])]] = {
    ResourceScope.using() {
      val dType = DType.Float32
      val inputShape = Shape(1, 3, 512, 512)
      // ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
      val outputShape = Shape(1, 6132, 6)
      val inputDescriptors = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))
      val objDetector = new ObjectDetector(modelPathPrefix, inputDescriptors, context)
      // Loading batch of images from the directory path
      val batchFiles = generateBatches(inputImageDir, 20)
      var outputList = IndexedSeq[IndexedSeq[(String, Array[Float])]]()

      for (batchFile <- batchFiles) {
        val imgList = ImageClassifier.loadInputBatch(batchFile)
        // Running inference on batch of images loaded in previous step
        outputList ++= objDetector.imageBatchObjectDetect(imgList, Some(5))
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
    val parser : CmdLineParser = new CmdLineParser(inst)
    parser.parseArgument(args.toList.asJava)
    val mdprefixDir = inst.modelPathPrefix
    val imgPath = inst.inputImagePath
    val imgDir = inst.inputImageDir
    if (!checkExist(Array(mdprefixDir + "-symbol.json", imgDir, imgPath))) {
      logger.error("Model or input image path does not exist")
      sys.exit(1)
    }

    var context = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }

    try {
      val inputShape = Shape(1, 3, 512, 512)
      val outputShape = Shape(1, 6132, 6)

      val width = inputShape(2)
      val height = inputShape(3)
      var outputStr : String = "\n"

      val output = runObjectDetectionSingle(mdprefixDir, imgPath, context)


      for (ele <- output) {
        for (i <- ele) {
          outputStr += "Class: " + i._1 + "\n"
          val arr = i._2
          outputStr += "Probabilties: " + arr(0) + "\n"
          val coord = Array[Float](
            arr(1) * width, arr(2) * height,
            arr(3) * width, arr(4) * height
          )
          outputStr += "Coord:" + coord.mkString(",") + "\n"
        }
      }
      logger.info(outputStr)

      val outputList = runObjectDetectionBatch(mdprefixDir, imgDir, context)

      outputStr = "\n"
      for (idx <- outputList.indices) {
        outputStr += "*** Image " + (idx + 1) + "***" + "\n"
        for (i <- outputList(idx)) {
          outputStr += "Class: " + i._1 + "\n"
          val arr = i._2
          outputStr += "Probabilties: " + arr(0) + "\n"
          val coord = Array[Float](
            arr(1) * width, arr(2) * height,
            arr(3) * width, arr(4) * height
          )
          outputStr += "Coord:" + coord.mkString(",") + "\n"
        }
      }
      logger.info(outputStr)

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
    sys.exit(0)
  }


  def checkExist(arr : Array[String]) : Boolean = {
    var exist : Boolean = true
    for (item <- arr) {
      if (!(Files.exists(Paths.get(item)))) {
        logger.error("Cannot find: " + item)
        exist = false
      }
    }
    exist
  }

}

class CLIParser extends CLIParserBase {
  @Option(name = "--model-path-prefix", usage = "the input model directory and prefix of the model")
  val modelPathPrefix: String = "/model/ssd_resnet50_512"
  @Option(name = "--input-image", usage = "the input image")
  val inputImagePath: String = "/images/dog.jpg"
  @Option(name = "--input-dir", usage = "the input batch of images directory")
  val inputImageDir: String = "/images/"
}

class SSDClassifierExample(CLIParser: CLIParser)
  extends InferBase {
  override def loadModel(context: Array[Context], batchInference: Boolean = false): Any = {
    val dType = DType.Float32
    val batchSize = if (batchInference) CLIParser.batchSize else 1
    val inputShape = Shape(batchSize, 3, 512, 512)
    val inputDescriptors = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))
    new ObjectDetector(CLIParser.modelPathPrefix, inputDescriptors, context)
  }
  override def loadSingleData(): Any = {
    val img = ImageClassifier.loadImageFromFile(CLIParser.inputImagePath)
    img
  }

  override def runSingleInference(loadedModel: Any, input: Any): Any = {
    val detector = loadedModel.asInstanceOf[ObjectDetector]
    val imgInput = input.asInstanceOf[BufferedImage]
    detector.imageObjectDetect(imgInput)
  }

  override def loadInputBatch(inputPaths: Any): Any = {
    val batchFile = inputPaths.asInstanceOf[List[String]]
    ImageClassifier.loadInputBatch(batchFile)
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

  override def runBatchInference(loadedModel: Any, input: Any): Any = {
    val model = loadedModel.asInstanceOf[ObjectDetector]
    val imgInput = input.asInstanceOf[Traversable[BufferedImage]]
    val output = model.imageBatchObjectDetect(imgInput, Some(5))
    output
  }
}
