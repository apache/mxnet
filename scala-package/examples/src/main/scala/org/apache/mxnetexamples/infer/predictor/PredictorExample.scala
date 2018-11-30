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

import scala.io
import org.apache.mxnet._
import org.apache.mxnet.infer.Predictor
import org.apache.mxnetexamples.benchmark.CLIParserBase
import org.kohsuke.args4j.{CmdLineParser, Option}

import scala.collection.JavaConverters._

object PredictorExample {

  def loadModel(modelPathPrefix : String, inputDesc : IndexedSeq[DataDesc],
                context : Context, epoch : Int): Predictor = {
    new Predictor(modelPathPrefix, inputDesc, context, Some(epoch))
  }

  def doInference(predictor : Predictor, imageND : NDArray): IndexedSeq[NDArray] = {
    predictor.predictWithNDArray(IndexedSeq(imageND))
  }

  def preProcess(imagePath: String, h: Int, w: Int) : NDArray = {
    var img = Image.imRead(imagePath)
    img = Image.imResize(img, h, w)
    // HWC -> CHW
    img = NDArray.api.transpose(img, Some(Shape(2, 0, 1)))
    img = NDArray.api.expand_dims(img, 0)
    img.asType(DType.Float32)
  }

  def postProcess(modelPathPrefix : String, result : Array[Float]) : String = {
    val dirPath = modelPathPrefix.substring(0, 1 + modelPathPrefix.lastIndexOf(File.separator))
    val d = new File(dirPath)
    require(d.exists && d.isDirectory, s"directory: $dirPath not found")
    val f = io.Source.fromFile(dirPath + "synset.txt")
    val s = f.getLines().toIndexedSeq
    val maxIdx = result.zipWithIndex.maxBy(_._1)._2
    printf(s"Predict Result ${s(maxIdx)} with prob ${result(maxIdx)}\n")
    s(maxIdx)
  }

  def main(args : Array[String]): Unit = {
    val inst = new CLIParser
    val parser: CmdLineParser = new CmdLineParser(inst)

    parser.parseArgument(args.toList.asJava)

    var context = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }

    val imgWidth = 224
    val imgHeight = 224

    val inputDesc = IndexedSeq(new DataDesc("data", Shape(1, 3, imgHeight, imgWidth),
      DType.Float32, Layout.NCHW))

    val predictor = loadModel(inst.modelPathPrefix, inputDesc, context, 0)
    val img = preProcess(inst.inputImagePath, imgHeight, imgWidth)
    val result = doInference(predictor, img)(0).toArray
    postProcess(inst.modelPathPrefix, result)
  }

}

class CLIParser extends CLIParserBase{
  @Option(name = "--model-path-prefix", usage = "the input model directory")
  val modelPathPrefix: String = "/resnet-152/resnet-152"
  @Option(name = "--input-image", usage = "the input image")
  val inputImagePath: String = "/images/kitten.jpg"
}
