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

package org.apache.mxnetexamples.neuralstyle

import java.io.File

import com.sksamuel.scrimage.{Image, Pixel}
import com.sksamuel.scrimage.filter.GaussianBlurFilter
import com.sksamuel.scrimage.nio.JpegWriter
import org.apache.mxnet._
import org.apache.mxnet.optimizer.Adam
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

/**
  * An Implementation of the paper A Neural Algorithm of Artistic Style
  */
object NeuralStyle {
  case class NSExecutor(executor: Executor, data: NDArray, dataGrad: NDArray)

  private val logger = LoggerFactory.getLogger(classOf[NeuralStyle])

  def preprocessContentImage(path: String, longEdge: Int, ctx: Context): NDArray = {
    val img = Image(new File(path))
    logger.info(s"load the content image, size = ${(img.height, img.width)}")
    val factor = longEdge.toFloat / Math.max(img.height, img.width)
    val (newHeight, newWidth) = ((img.height * factor).toInt, (img.width * factor).toInt)
    val resizedImg = img.scaleTo(newWidth, newHeight)
    val sample = NDArray.empty(Shape(1, 3, newHeight, newWidth), ctx)
    val datas = {
      val rgbs = resizedImg.iterator.toArray.map { p =>
        (p.red, p.green, p.blue)
      }
      val r = rgbs.map(_._1 - 123.68f)
      val g = rgbs.map(_._2 - 116.779f)
      val b = rgbs.map(_._3 - 103.939f)
      r ++ g ++ b
    }
    sample.set(datas)
    logger.info(s"resize the content image to ${(newHeight, newWidth)}")
    sample
  }

  def preprocessStyleImage(path: String, shape: Shape, ctx: Context): NDArray = {
    val img = Image(new File(path))
    val resizedImg = img.scaleTo(shape(3), shape(2))
    val sample = NDArray.empty(Shape(1, 3, shape(2), shape(3)), ctx)
    val datas = {
      val rgbs = resizedImg.iterator.toArray.map { p =>
        (p.red, p.green, p.blue)
      }
      val r = rgbs.map(_._1 - 123.68f)
      val g = rgbs.map(_._2 - 116.779f)
      val b = rgbs.map(_._3 - 103.939f)
      r ++ g ++ b
    }
    sample.set(datas)
    sample
  }

  def clip(array: Array[Float]): Array[Float] = array.map { a =>
    if (a < 0) 0f
    else if (a > 255) 255f
    else a
  }

  def postprocessImage(img: NDArray): Image = {
    val datas = img.toArray
    val spatialSize = img.shape(2) * img.shape(3)
    val r = clip(datas.take(spatialSize).map(_ + 123.68f))
    val g = clip(datas.drop(spatialSize).take(spatialSize).map(_ + 116.779f))
    val b = clip(datas.takeRight(spatialSize).map(_ + 103.939f))
    val pixels = for (i <- 0 until spatialSize)
      yield Pixel(r(i).toInt, g(i).toInt, b(i).toInt, 255)
    Image(img.shape(3), img.shape(2), pixels.toArray)
  }

  def saveImage(img: NDArray, filename: String, radius: Int): Unit = {
    logger.info(s"save output to $filename")
    val out = postprocessImage(img)
    val gauss = GaussianBlurFilter(radius).op
    val result = Image(out.width, out.height)
    gauss.filter(out.awt, result.awt)
    result.output(filename)(JpegWriter())
  }

  def styleGramSymbol(inputSize: (Int, Int), style: Symbol): (Symbol, List[Int]) = {
    val (_, outputShape, _) = style.inferShape(
      Map("data" -> Shape(1, 3, inputSize._1, inputSize._2)))
    var gramList = List[Symbol]()
    var gradScale = List[Int]()
    for (i <- 0 until style.listOutputs().length) {
      val shape = outputShape(i)
      val x = Symbol.api.Reshape(data = Some(style.get(i)),
        target_shape = Some(Shape(shape(1), shape(2) * shape(3))))
      val gram = Symbol.api.FullyConnected(data = Some(x), weight = Some(x),
        no_bias = Some(true), num_hidden = shape(1))
      x.dispose()
      gramList = gramList :+ gram
      gradScale = gradScale :+ (shape(1) * shape(2) * shape(3) * shape(1))
    }
    (Symbol.Group(gramList: _*), gradScale)
  }

  def getLoss(gram: Symbol, content: Symbol): (Symbol, Symbol) = {
    var gramLoss = ListBuffer[Symbol]()
    for (i <- 0 until gram.listOutputs().length) {
      val gvar = Symbol.Variable(s"target_gram_$i")
      Symbol.api.square(data = Some(gvar - gram.get(i)))
      gramLoss += Symbol.api.sum(
        Some(Symbol.api.square(data = Some(gvar - gram.get(i))))
      )
      gvar.dispose()
    }
    gram.dispose()
    val cvar = Symbol.Variable("target_content")
    val contentLoss = Symbol.api.sum(
      Some(Symbol.api.square(Some(cvar - content)))
    )
    (Symbol.Group(gramLoss: _*), contentLoss)
  }

  def getTvGradExecutor(img: NDArray, ctx: Context, tvWeight: Float): scala.Option[Executor] = {
    // create TV gradient executor with input binded on img
    if (tvWeight <= 0.0f) None

    val nChannel = img.shape(1)
    val sImg = Symbol.Variable("img")
    val sKernel = Symbol.Variable("kernel")
    val channels = Symbol.api.SliceChannel(data = Some(sImg), num_outputs = nChannel)
    val result = (0 until nChannel).map { i =>
      Symbol.api.Convolution(data = Some(channels.get(i)), weight = Some(sKernel),
        num_filter = 1, kernel = Shape(3, 3), pad = Some(Shape(1, 1)), no_bias = Some(true),
        stride = Some(Shape(1, 1)))
    }.toArray
    val out = Symbol.api.Concat(result, result.length) * tvWeight
    val kernel = {
      val tmp = NDArray.empty(Shape(1, 1, 3, 3), ctx)
      tmp.set(Array[Float](0, -1, 0, -1, 4, -1, 0, -1, 0))
      tmp / 0.8f
    }
    Some(out.bind(ctx, Map("img" -> img, "kernel" -> kernel)))
  }

  def twoNorm(array: Array[Float]): Float = {
    Math.sqrt(array.map(x => x * x).sum.toDouble).toFloat
  }

  //scalastyle:off
  def runTraining(model : String, contentImage : String, styleImage: String, dev : Context,
                  modelPath : String, outputDir : String, styleWeight : Float,
                  contentWeight : Float, tvWeight : Float, gaussianRadius : Int,
                  lr: Float, maxNumEpochs: Int, maxLongEdge: Int,
                  saveEpochs : Int, stopEps: Float) : Unit = {
    NDArrayCollector.auto().withScope {
      val contentNp = preprocessContentImage(contentImage, maxLongEdge, dev)
      val styleNp = preprocessStyleImage(styleImage, contentNp.shape, dev)
      val size = (contentNp.shape(2), contentNp.shape(3))

      val (style, content) = ModelVgg19.getSymbol
      val (gram, gScale) = styleGramSymbol(size, style)
      var modelExecutor = ModelVgg19.getExecutor(gram, content, modelPath, size, dev)

      modelExecutor.data.set(styleNp)
      modelExecutor.executor.forward()

      val styleArray = modelExecutor.style.map(_.copyTo(Context.cpu()))
      modelExecutor.data.set(contentNp)
      modelExecutor.executor.forward()
      val contentArray = modelExecutor.content.copyTo(Context.cpu())

      // delete the executor
      modelExecutor.argDict.foreach(ele => ele._2.dispose())
      modelExecutor.content.dispose()
      modelExecutor.data.dispose()
      modelExecutor.dataGrad.dispose()
      modelExecutor.style.foreach(_.dispose())
      modelExecutor.executor.dispose()
      modelExecutor = null

      val (styleLoss, contentLoss) = getLoss(gram, content)
      modelExecutor = ModelVgg19.getExecutor(
        styleLoss, contentLoss, modelPath, size, dev)

      val gradArray = {
        var tmpGA = Array[NDArray]()
        for (i <- 0 until styleArray.length) {
          modelExecutor.argDict(s"target_gram_$i").set(styleArray(i))
          tmpGA = tmpGA :+ NDArray.ones(Shape(1), dev) * (styleWeight / gScale(i))
        }
        tmpGA :+ NDArray.ones(Shape(1), dev) * contentWeight
      }

      modelExecutor.argDict("target_content").set(contentArray)

      // train
      val img = Random.uniform(-0.1f, 0.1f, contentNp.shape, dev)
      val lrFS = new FactorScheduler(step = 10, factor = 0.9f)

      saveImage(contentNp, s"${outputDir}/input.jpg", gaussianRadius)
      saveImage(styleNp, s"${outputDir}/style.jpg", gaussianRadius)

      val optimizer = new Adam(
        learningRate = lr,
        wd = 0.005f,
        lrScheduler = lrFS)
      val optimState = optimizer.createState(0, img)

      logger.info(s"start training arguments")

      var oldImg = img.copyTo(dev)
      val clipNorm = img.shape.toVector.reduce(_ * _)
      val tvGradExecutor = getTvGradExecutor(img, dev, tvWeight)
      var eps = 0f
      var trainingDone = false
      var e = 0
      while (e < maxNumEpochs && !trainingDone) {
        modelExecutor.data.set(img)
        modelExecutor.executor.forward()
        modelExecutor.executor.backward(gradArray)

        val gNorm = NDArray.norm(modelExecutor.dataGrad).toScalar
        if (gNorm > clipNorm) {
          modelExecutor.dataGrad.set(modelExecutor.dataGrad * (clipNorm / gNorm))
        }
        tvGradExecutor match {
          case Some(executor) => {
            executor.forward()
            optimizer.update(0, img,
              modelExecutor.dataGrad + executor.outputs(0),
              optimState)
          }
          case None =>
            optimizer.update(0, img, modelExecutor.dataGrad, optimState)
        }
        eps = (NDArray.norm(oldImg - img) / NDArray.norm(img)).toScalar
        oldImg.set(img)
        logger.info(s"epoch $e, relative change $eps")

        if (eps < stopEps) {
          logger.info("eps < args.stop_eps, training finished")
          trainingDone = true
        }
        if ((e + 1) % saveEpochs == 0) {
          saveImage(img, s"${outputDir}/tmp_${e + 1}.jpg", gaussianRadius)
        }
        e = e + 1
      }
      saveImage(img, s"${outputDir}/out.jpg", gaussianRadius)
      logger.info("Finish fit ...")
    }
  }

  def main(args: Array[String]): Unit = {
    val alle = new NeuralStyle
    val parser: CmdLineParser = new CmdLineParser(alle)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(alle.contentImage != null && alle.styleImage != null
        && alle.modelPath != null && alle.outputDir != null)

      val dev = if (alle.gpu >= 0) Context.gpu(alle.gpu) else Context.cpu(0)
      runTraining(alle.model, alle.contentImage, alle.styleImage, dev, alle.modelPath,
        alle.outputDir, alle.styleWeight, alle.contentWeight, alle.tvWeight,
        alle.gaussianRadius, alle.lr, alle.maxNumEpochs, alle.maxLongEdge,
        alle.saveEpochs, alle.stopEps)
    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class NeuralStyle {
  @Option(name = "--model", usage = "the pretrained model to use: ['vgg']")
  private val model: String = "vgg19"
  @Option(name = "--content-image", usage = "the content image")
  private val contentImage: String = null
  @Option(name = "--style-image", usage = "the style image")
  private val styleImage: String = null
  @Option(name = "--model-path", usage = "the model file path")
  private val modelPath: String = null
  @Option(name = "--stop-eps", usage = "stop if the relative chanage is less than eps")
  private val stopEps: Float = 0.0005f
  @Option(name = "--content-weight", usage = "the weight for the content image")
  private val contentWeight: Float = 20f
  @Option(name = "--style-weight", usage = "the weight for the style image")
  private val styleWeight: Float = 1f
  @Option(name = "--tv-weight", usage = "the magtitute on TV loss")
  private val tvWeight: Float = 0.01f
  @Option(name = "--max-num-epochs", usage = "the maximal number of training epochs")
  private val maxNumEpochs: Int = 1000
  @Option(name = "--max-long-edge", usage = "resize the content image")
  private val maxLongEdge: Int = 600
  @Option(name = "--lr", usage = "the initial learning rate")
  private val lr: Float = 10f
  @Option(name = "--gpu", usage = "which gpu card to use, -1 means using cpu")
  private val gpu: Int = 0
  @Option(name = "--output-dir", usage = "the output directory")
  private val outputDir: String = null
  @Option(name = "--save-epochs", usage = "save the output every n epochs")
  private val saveEpochs: Int = 50
  @Option(name = "--gaussian-radius", usage = "the gaussian blur filter radius")
  private val gaussianRadius: Int = 1
}
