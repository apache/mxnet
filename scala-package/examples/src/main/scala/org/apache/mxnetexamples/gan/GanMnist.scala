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

package org.apache.mxnetexamples.gan

import org.apache.mxnet.{Context, CustomMetric, DataBatch, IO, NDArray, NDArrayCollector, Shape, Symbol, Xavier}
import org.apache.mxnet.optimizer.Adam
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

object GanMnist {

  private val logger = LoggerFactory.getLogger(classOf[GanMnist])

  // a deconv layer that enlarges the feature map
  def deconv2D(data: Symbol, iShape: Shape, oShape: Shape,
               kShape: (Int, Int), name: String, stride: (Int, Int) = (2, 2)): Symbol = {
    val targetShape = Shape(oShape(oShape.length - 2), oShape(oShape.length - 1))
    val net = Symbol.api.Deconvolution(data = Some(data), kernel = Shape(kShape._1, kShape._2),
      stride = Some(Shape(stride._1, stride._2)), target_shape = Some(targetShape),
      num_filter = oShape(0), no_bias = Some(true), name = name)
    net
  }

  def deconv2DBnRelu(data: Symbol, prefix: String, iShape: Shape,
                     oShape: Shape, kShape: (Int, Int), eps: Float = 1e-5f + 1e-12f): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.api.BatchNorm(name = s"${prefix}_bn", data = Some(net),
      fix_gamma = Some(true), eps = Some(eps))
    net = Symbol.api.Activation(data = Some(net), act_type = "relu", name = s"${prefix}_act")
    net
  }

  def deconv2DAct(data: Symbol, prefix: String, actType: String,
                  iShape: Shape, oShape: Shape, kShape: (Int, Int)): Symbol = {
    var net = deconv2D(data, iShape, oShape, kShape, name = s"${prefix}_deconv")
    net = Symbol.api.Activation(data = Some(net), act_type = "relu", name = s"${prefix}_act")
    net
  }

  def makeDcganSym(oShape: Shape, ngf: Int = 128, finalAct: String = "sigmoid",
                   eps: Float = 1e-5f + 1e-12f): (Symbol, Symbol) = {

    val code = Symbol.Variable("rand")
    var net = Symbol.api.FullyConnected(data = Some(code), num_hidden = 4 * 4 * ngf * 4,
      no_bias = Some(true), name = " g1")
    net = Symbol.api.Activation(data = Some(net), act_type = "relu", name = "gact1")
    // 4 x 4
    net = Symbol.api.Reshape(data = Some(net), shape = Some(Shape(-1, ngf * 4, 4, 4)))
    // 8 x 8
    net = deconv2DBnRelu(net, prefix = "g2",
      iShape = Shape(ngf * 4, 4, 4), oShape = Shape(ngf * 2, 8, 8), kShape = (3, 3))
    // 14x14
    net = deconv2DBnRelu(net, prefix = "g3",
      iShape = Shape(ngf * 2, 8, 8), oShape = Shape(ngf, 14, 14), kShape = (4, 4))
    // 28x28
    val gout = deconv2DAct(net, prefix = "g4", actType = finalAct, iShape = Shape(ngf, 14, 14),
      oShape = Shape(oShape.toArray.takeRight(3)), kShape = (4, 4))

    val data = Symbol.Variable("data")
    // 28 x 28
    val conv1 = Symbol.api.Convolution(data = Some(data), kernel = Shape(5, 5),
      num_filter = 20, name = "conv1")
    val tanh1 = Symbol.api.Activation(data = Some(conv1), act_type = "tanh")
    val pool1 = Symbol.api.Pooling(data = Some(tanh1), pool_type = Some("max"),
      kernel = Some(Shape(2, 2)), stride = Some(Shape(2, 2)))
    // second conv
    val conv2 = Symbol.api.Convolution(data = Some(pool1), kernel = Shape(5, 5),
      num_filter = 50, name = "conv2")
    val tanh2 = Symbol.api.Activation(data = Some(conv2), act_type = "tanh")
    val pool2 = Symbol.api.Pooling(data = Some(tanh2), pool_type = Some("max"),
      kernel = Some(Shape(2, 2)), stride = Some(Shape(2, 2)))
    var d5 = Symbol.api.Flatten(data = Some(pool2))
    d5 = Symbol.api.FullyConnected(data = Some(d5), num_hidden = 500, name = "fc1")
    d5 = Symbol.api.Activation(data = Some(d5), act_type = "tanh")
    d5 = Symbol.api.FullyConnected(data = Some(d5), num_hidden = 1, name = "fc_dloss")
    val dloss = Symbol.api.LogisticRegressionOutput(data = Some(d5), name = "dloss")

    (gout, dloss)
  }

  // Evaluation
  def ferr(label: NDArray, pred: NDArray): Float = {
    val predArr = pred.toArray.map(p => if (p > 0.5) 1f else 0f)
    val labelArr = label.toArray
    labelArr.zip(predArr).map { case (l, p) => Math.abs(l - p) }.sum / label.shape(0)
  }

  def runTraining(dataPath : String, context : Context,
                  outputPath : String, numEpoch : Int): Float = {
    val output = NDArrayCollector.auto().withScope {
      val lr = 0.0005f
      val beta1 = 0.5f
      val batchSize = 100
      val randShape = Shape(batchSize, 100)
      val dataShape = Shape(batchSize, 1, 28, 28)

      val (symGen, symDec) =
        makeDcganSym(oShape = dataShape, ngf = 32, finalAct = "sigmoid")

      val gMod = new GANModule(
        symGen,
        symDec,
        context = context,
        dataShape = dataShape,
        codeShape = randShape)

      gMod.initGParams(new Xavier(factorType = "in", magnitude = 2.34f))
      gMod.initDParams(new Xavier(factorType = "in", magnitude = 2.34f))

      gMod.initOptimizer(new Adam(learningRate = lr, wd = 0f, beta1 = beta1))

      val params = Map(
        "image" -> s"$dataPath/train-images-idx3-ubyte",
        "label" -> s"$dataPath/train-labels-idx1-ubyte",
        "input_shape" -> s"(1, 28, 28)",
        "batch_size" -> s"$batchSize",
        "shuffle" -> "True"
      )

      val mnistIter = IO.MNISTIter(params)

      val metricAcc = new CustomMetric(ferr, "ferr")

      var t = 0
      var dataBatch: DataBatch = null
      var acc = 0.0f
      for (epoch <- 0 until numEpoch) {
        mnistIter.reset()
        metricAcc.reset()
        t = 0
        while (mnistIter.hasNext) {
          dataBatch = mnistIter.next()
          NDArrayCollector.auto().withScope {
            gMod.update(dataBatch)
            gMod.dLabel.set(0f)
            metricAcc.update(Array(gMod.dLabel), gMod.outputsFake)
            gMod.dLabel.set(1f)
            metricAcc.update(Array(gMod.dLabel), gMod.outputsReal)

            if (t % 50 == 0) {
              val (name, value) = metricAcc.get
              acc = value(0)
              logger.info(s"epoch: $epoch, iter $t, metric=${value.mkString(" ")}")
              Viz.imSave("gout", outputPath, gMod.tempOutG(0), flip = true)
              val diff = gMod.tempDiffD
              val arr = diff.toArray
              val mean = arr.sum / arr.length
              val std = {
                val tmpA = arr.map(a => (a - mean) * (a - mean))
                Math.sqrt(tmpA.sum / tmpA.length).toFloat
              }
              diff.set((diff - mean) / std + 0.5f)
              Viz.imSave("diff", outputPath, diff, flip = true)
              Viz.imSave("data", outputPath, dataBatch.data(0), flip = true)
            }
          }
          dataBatch.dispose()
          t += 1
        }
      }
      acc
    }
    output
  }

  def main(args: Array[String]): Unit = {
    val anst = new GanMnist
    val parser: CmdLineParser = new CmdLineParser(anst)
    try {
      parser.parseArgument(args.toList.asJava)

      val dataPath = if (anst.mnistDataPath == null) System.getenv("MXNET_HOME")
      else anst.mnistDataPath

      assert(dataPath != null)
      val context = if (anst.gpu == -1) Context.cpu() else Context.gpu(anst.gpu)

      runTraining(dataPath, context, anst.outputPath, 100)
    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class GanMnist {
  @Option(name = "--mnist-data-path", usage = "the mnist data path")
  private val mnistDataPath: String = null
  @Option(name = "--output-path", usage = "the path to save the generated result")
  private val outputPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = -1
}
