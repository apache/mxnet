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
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.DataBatch
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Executor
import ml.dmlc.mxnet.optimizer.SGD
import java.io.File
import javax.imageio.ImageIO
import scala.util.Random
import ml.dmlc.mxnet.optimizer.Adam

/**
 * @author Depeng Liang
 */
object BoostTrain {

  private val logger = LoggerFactory.getLogger(classOf[BoostTrain])

  def getTvGradExecutor(img: NDArray, ctx: Context, tvWeight: Float): Executor = {
    // create TV gradient executor with input binded on img
    if (tvWeight <= 0.0f) null

    val nChannel = img.shape(1)
    val sImg = Symbol.Variable("img")
    val sKernel = Symbol.Variable("kernel")
    val channels = Symbol.SliceChannel()(sImg)(Map("num_outputs" -> nChannel))
    val out = Symbol.Concat()((0 until nChannel).map { i =>
      Symbol.Convolution()()(Map("data" -> channels.get(i), "weight" -> sKernel,
                    "num_filter" -> 1, "kernel" -> "(3,3)", "pad" -> "(1,1)",
                    "no_bias" -> true, "stride" -> "(1,1)"))
    }.toArray: _*)() * tvWeight
    val kernel = {
      val tmp = NDArray.empty(Shape(1, 1, 3, 3), ctx)
      tmp.set(Array[Float](0, -1, 0, -1, 4, -1, 0, -1, 0))
      tmp / 8.0f
    }
    out.bind(ctx, Map("img" -> img, "kernel" -> kernel))
  }

  def main(args: Array[String]): Unit = {
    val stin = new BoostTrain
    val parser: CmdLineParser = new CmdLineParser(stin)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(stin.dataPath != null
          && stin.vggModelPath != null
          && stin.saveModelPath != null
          && stin.styleImage != null)
      // params
      val vggParams = NDArray.load2Map(stin.vggModelPath)
      val styleWeight = 1.2f
      val contentWeight = 10f
      val dShape = Shape(1, 3, 384, 384)
      val clipNorm = 0.05f * dShape.product
      val modelPrefix = "v3"
      val ctx = if (stin.gpu == -1) Context.cpu() else Context.gpu(stin.gpu)

      // init style
      val styleNp = DataProcessing.preprocessStyleImage(stin.styleImage, dShape, ctx)
      var styleMod = Basic.getStyleModule("style", dShape, ctx, vggParams)
      styleMod.forward(Array(styleNp))
      val styleArray = styleMod.getOutputs().map(_.copyTo(Context.cpu()))
      styleMod.dispose()
      styleMod = null

      // content
      val contentMod = Basic.getContentModule("content", dShape, ctx, vggParams)

      // loss
      val (loss, gScale) = Basic.getLossModule("loss", dShape, ctx, vggParams)
      val extraArgs = (0 until styleArray.length)
                                  .map( i => s"target_gram_$i" -> styleArray(i)).toMap
      loss.setParams(extraArgs)
      var gradArray = Array[NDArray]()
      for (i <- 0 until styleArray.length) {
        gradArray = gradArray :+ (NDArray.ones(Shape(1), ctx) * (styleWeight / gScale(i)))
      }
      gradArray = gradArray :+ (NDArray.ones(Shape(1), ctx) * contentWeight)

      // generator
      val gens = Array(
          GenV4.getModule("g0", dShape, ctx),
          GenV3.getModule("g1", dShape, ctx),
          GenV3.getModule("g2", dShape, ctx),
          GenV4.getModule("g3", dShape, ctx)
      )
      gens.foreach { gen =>
        val opt = new SGD(learningRate = 1e-4f,
                          momentum = 0.9f,
                          wd = 5e-3f,
                          clipGradient = 5f)
        gen.initOptimizer(opt)
      }

      var filelist = new File(stin.dataPath).list().toList
      val numImage = filelist.length
      logger.info(s"Dataset size: $numImage")

      val tvWeight = 1e-2f

      val startEpoch = 0
      val endEpoch = 3

      for (k <- 0 until gens.length) {
        val path = new File(s"${stin.saveModelPath}/$k")
        if (!path.exists()) path.mkdir()
      }

      // train
      for (i <- startEpoch until endEpoch) {
        filelist = Random.shuffle(filelist)
        for (idx <- filelist.indices) {
          var dataArray = Array[NDArray]()
          var lossGradArray = Array[NDArray]()
          val data =
            DataProcessing.preprocessContentImage(s"${stin.dataPath}/${filelist(idx)}", dShape, ctx)
          dataArray = dataArray :+ data
          // get content
          contentMod.forward(Array(data))
          // set target content
          loss.setParams(Map("target_content" -> contentMod.getOutputs()(0)))
          // gen_forward
          for (k <- 0 until gens.length) {
            gens(k).forward(dataArray.takeRight(1))
            dataArray = dataArray :+ gens(k).getOutputs()(0)
            // loss forward
            loss.forward(dataArray.takeRight(1))
            loss.backward(gradArray)
            lossGradArray = lossGradArray :+ loss.getInputGrads()(0)
          }
          val grad = NDArray.zeros(data.shape, ctx)
          for (k <- gens.length - 1 to 0 by -1) {
            val tvGradExecutor = getTvGradExecutor(gens(k).getOutputs()(0), ctx, tvWeight)
            tvGradExecutor.forward()
            grad += lossGradArray(k) + tvGradExecutor.outputs(0)
            val gNorm = NDArray.norm(grad)
            if (gNorm.toScalar > clipNorm) {
              grad *= clipNorm / gNorm.toScalar
            }
            gens(k).backward(Array(grad))
            gens(k).update()
            gNorm.dispose()
            tvGradExecutor.dispose()
          }
          grad.dispose()
          if (idx % 20 == 0) {
            logger.info(s"Epoch $i: Image $idx")
            for (k <- 0 until gens.length) {
              val n = NDArray.norm(gens(k).getInputGrads()(0))
              logger.info(s"Data Norm : ${n.toScalar / dShape.product}")
              n.dispose()
            }
          }
          if (idx % 1000 == 0) {
            for (k <- 0 until gens.length) {
              gens(k).saveParams(
                  s"${stin.saveModelPath}/$k/${modelPrefix}_" +
                  s"${"%04d".format(i)}-${"%07d".format(idx)}.params")
            }
          }
          data.dispose()
        }
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

class BoostTrain {
  @Option(name = "--data-path", usage = "the input train data path")
  private val dataPath: String = null
  @Option(name = "--vgg--model-path", usage = "the pretrained model to use: ['vgg']")
  private val vggModelPath: String = null
  @Option(name = "--save--model-path", usage = "the save model path")
  private val saveModelPath: String = null
  @Option(name = "--style-image", usage = "the style image")
  private val styleImage: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = -1
}
