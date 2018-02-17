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

package ml.dmlc.mxnetexamples.neuralstyle.end2end

import org.slf4j.LoggerFactory
import org.kohsuke.args4j.{CmdLineParser, Option}
import scala.collection.JavaConverters._
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Context

/**
 * @author Depeng Liang
 */
object BoostInference {

  private val logger = LoggerFactory.getLogger(classOf[BoostInference])

  def main(args: Array[String]): Unit = {
    val stce = new BoostInference
    val parser: CmdLineParser = new CmdLineParser(stce)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(stce.modelPath != null
          && stce.inputImage != null
          && stce.outputPath != null)

      val dShape = Shape(1, 3, 480, 640)
      val clipNorm = 1.0f * dShape.product
      val ctx = if (stce.gpu == -1) Context.cpu() else Context.gpu(stce.gpu)

      // generator
      val gens = Array(
          GenV4.getModule("g0", dShape, ctx, isTrain = false),
          GenV3.getModule("g1", dShape, ctx, isTrain = false),
          GenV3.getModule("g2", dShape, ctx, isTrain = false),
          GenV4.getModule("g3", dShape, ctx, isTrain = false)
      )
      gens.zipWithIndex.foreach { case (gen, i) =>
        gen.loadParams(s"${stce.modelPath}/$i/v3_0002-0026000.params")
      }

      val contentNp =
        DataProcessing.preprocessContentImage(s"${stce.inputImage}", dShape, ctx)
      var data = Array(contentNp)
      for (i <- 0 until gens.length) {
        gens(i).forward(data.takeRight(1))
        val newImg = gens(i).getOutputs()(0)
        data :+= newImg
        DataProcessing.saveImage(newImg, s"${stce.outputPath}/out_${i}.jpg", stce.guassianRadius)
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

class BoostInference {
  @Option(name = "--model-path", usage = "the save model path")
  private val modelPath: String = null
  @Option(name = "--input-image", usage = "the style image")
  private val inputImage: String = null
  @Option(name = "--output-path", usage = "the output result path")
  private val outputPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = -1
  @Option(name = "--guassian-radius", usage = "the gaussian blur filter radius")
  private val guassianRadius: Int = 2
}
