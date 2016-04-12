package ml.dmlc.mxnet.examples.neuralstyle

import ml.dmlc.mxnet._
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import com.sksamuel.scrimage.Image
import java.io.File
import com.sksamuel.scrimage.Pixel
import com.sksamuel.scrimage.filter.GaussianBlurFilter
import com.sksamuel.scrimage.nio.JpegWriter
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.optimizer.Adam

/**
 * An Implementation of the paper A Neural Algorithm of Artistic Style
 * by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
 * @author Depeng Liang
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

  def styleGramExecutor(inputShape: Shape, ctx: Context): NSExecutor = {
    // symbol
    val data = Symbol.Variable("conv")
    val rsData = Symbol.Reshape()(Map("data" -> data,
      "target_shape" -> s"(${inputShape(1)},${inputShape(2) * inputShape(3)})"))
    val weight = Symbol.Variable("weight")
    val rsWeight = Symbol.Reshape()(Map("data" -> weight,
      "target_shape" -> s"(${inputShape(1)},${inputShape(2) * inputShape(3)})"))
    val fc = Symbol.FullyConnected()(Map("data" -> rsData, "weight" -> rsWeight,
      "no_bias" -> true, "num_hidden" -> inputShape(1)))
    // executor
    val conv = NDArray.zeros(inputShape, ctx)
    val grad = NDArray.zeros(inputShape, ctx)
    val args = Map("conv" -> conv, "weight" -> conv)
    val gradMap = Map("conv" -> grad)
    val reqs = Map("conv" -> "write", "weight" -> "null")
    val executor = fc.bind(ctx, args, gradMap, reqs, Nil, null)
    NSExecutor(executor, conv, grad)
  }

  def twoNorm(array: Array[Float]): Float = {
    Math.sqrt(array.map(x => x * x).sum.toDouble).toFloat
  }

  def main(args: Array[String]): Unit = {
    val alle = new NeuralStyle
    val parser: CmdLineParser = new CmdLineParser(alle)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(alle.contentImage != null && alle.styleImage != null
        && alle.modelPath != null && alle.outputDir != null)

      val dev = if (alle.gpu >= 0) Context.gpu(alle.gpu) else Context.cpu(0)
      val contentNp = preprocessContentImage(alle.contentImage, alle.maxLongEdge, dev)
      val styleNp = preprocessStyleImage(alle.styleImage, contentNp.shape, dev)
      val size = (contentNp.shape(2), contentNp.shape(3))

      val modelExecutor = ModelVgg19.getModel(alle.modelPath, size, dev)
      val gramExecutor = modelExecutor.style.map(arr => styleGramExecutor(arr.shape, dev))

      // get style representation
      val styleArray = gramExecutor.map { gram =>
        NDArray.zeros(gram.executor.outputs(0).shape, dev)
      }
      modelExecutor.data.set(styleNp)
      modelExecutor.executor.forward()

      for(i <- 0 until modelExecutor.style.length) {
        modelExecutor.style(i).copyTo(gramExecutor(i).data)
        gramExecutor(i).executor.forward()
        gramExecutor(i).executor.outputs(0).copyTo(styleArray(i))
      }

      // get content representation
      val contentArray = NDArray.zeros(modelExecutor.content.shape, dev)
      val contentGrad = NDArray.zeros(modelExecutor.content.shape, dev)
      modelExecutor.data.set(contentNp)
      modelExecutor.executor.forward()
      modelExecutor.content.copyTo(contentArray)

      // train
      val img = Random.uniform(-0.1f, 0.1f, contentNp.shape, dev)
      val lr = new FactorScheduler(step = 10, factor = 0.9f)

      saveImage(contentNp, s"${alle.outputDir}/input.jpg", alle.guassianRadius)
      saveImage(styleNp, s"${alle.outputDir}/style.jpg", alle.guassianRadius)

       val optimizer = new Adam(
          learningRate = alle.lr,
          wd = 0.005f,
          clipGradient = 10,
          lrScheduler = lr)
      val optimState = optimizer.createState(0, img)

      logger.info(s"start training arguments $alle")

      var oldImg = img.copy()
      var gradArray: Array[NDArray] = null
      var eps = 0f
      var trainingDone = false
      var e = 0
      while (e < alle.maxNumEpochs && !trainingDone) {
        modelExecutor.data.set(img)
        modelExecutor.executor.forward()

        // style gradient
        for (i <- 0 until modelExecutor.style.length) {
          val gram = gramExecutor(i)
          gram.data.set(modelExecutor.style(i))
          gram.executor.forward()
          gram.executor.backward(gram.executor.outputs(0) - styleArray(i))
          val tmp = gram.data.shape(1) * gram.data.shape(1) *
            gram.data.shape(2) * gram.data.shape(3)
          gram.dataGrad.set(gram.dataGrad / tmp.toFloat)
          gram.dataGrad.set(gram.dataGrad * alle.styleWeight)
        }

        // content gradient
        contentGrad.set((modelExecutor.content - contentArray) * alle.contentWeight)

        // image gradient
        gradArray = gramExecutor.map(_.dataGrad) :+ contentGrad
        modelExecutor.executor.backward(gradArray)

        optimizer.update(0, img, modelExecutor.dataGrad, optimState)
        eps = twoNorm((oldImg - img).toArray) / twoNorm(img.toArray)
        oldImg.set(img)
        logger.info(s"epoch $e, relative change $eps")

        if (eps < alle.stopEps) {
          logger.info("eps < args.stop_eps, training finished")
          trainingDone = true
        }
        if ((e + 1) % alle.saveEpochs == 0) {
          saveImage(img, s"${alle.outputDir}/tmp_${e + 1}.jpg", alle.guassianRadius)
        }
        e = e + 1
      }
      saveImage(img, s"${alle.outputDir}/out.jpg", alle.guassianRadius)
      logger.info("Finish fit ...")
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
  @Option(name = "--guassian-radius", usage = "the gaussian blur filter radius")
  private val guassianRadius: Int = 1
}
