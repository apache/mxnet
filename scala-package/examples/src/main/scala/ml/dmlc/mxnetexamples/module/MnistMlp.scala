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

package ml.dmlc.mxnetexamples.module

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.module.{FitParams, Module}
import ml.dmlc.mxnet.DataDesc._
import ml.dmlc.mxnet.optimizer.SGD
import org.kohsuke.args4j.{Option, CmdLineParser}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

object MnistMlp {
  private val logger = LoggerFactory.getLogger(classOf[MnistMlp])

  def getSymbol: Symbol = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")(data)(Map("num_hidden" -> 128))
    val act1 = Symbol.Activation(name = "relu1")(fc1)(Map("act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")(act1)(Map("num_hidden" -> 64))
    val act2 = Symbol.Activation(name = "relu2")(fc2)(Map("act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected(name = "fc3")(act2)(Map("num_hidden" -> 10))
    val softmax = Symbol.SoftmaxOutput(name = "softmax")(fc3)()
    softmax
  }

  def runIntermediateLevelApi(train: DataIter, eval: DataIter,
      cmdLine: MnistMlp, loadModelEpoch: Int = -1): Unit = {
    // Intermediate-level API
    val mod = if (loadModelEpoch == -1) {
      new Module(getSymbol)
    } else {
      logger.info("Load checkpoint from epoch {}", loadModelEpoch)
      Module.loadCheckpoint("model/mnist_mlp", loadModelEpoch, loadOptimizerStates = true)
    }
    mod.bind(dataShapes = train.provideData, labelShapes = Some(train.provideLabel))
    mod.initParams()
    mod.initOptimizer(optimizer = new SGD(learningRate = 0.01f, momentum = 0.9f))

    val metric = new Accuracy()

    for (epoch <- 0 until cmdLine.numEpoch) {
      while (train.hasNext) {
        val batch = train.next()
        mod.forward(batch)
        mod.updateMetric(metric, batch.label)
        mod.backward()
        mod.update()
      }

      mod.saveCheckpoint("model/mnist_mlp", epoch, saveOptStates = true)

      val (name, value) = metric.get
      name.zip(value).foreach { case (n, v) =>
        logger.info(s"epoch $epoch $n=$v")
      }
      metric.reset()
      train.reset()
    }
  }

  def runHighLevelApi(train: DataIter, test: DataIter, cmdLine: MnistMlp): Unit = {
    // High-level API
    train.reset()
    val mod = new Module(getSymbol)
    mod.fit(train, evalData = scala.Option(test), numEpoch = cmdLine.numEpoch)

    // prediction iterator API
    var iBatch = 0
    test.reset()
    while (test.hasNext) {
      val batch = test.next()
      val preds = mod.predict(batch)
      val predLabel: Array[Int] = NDArray.argmax_channel(preds(0)).toArray.map(_.toInt)
      val label = batch.label(0).toArray.map(_.toInt)
      val acc = predLabel.zip(label).map { case (py, y) =>
        if (py == y) 1 else 0
      }.sum / predLabel.length.toFloat
      if (iBatch % 20 == 0) {
        logger.info(s"Batch $iBatch acc: $acc")
      }
      iBatch += 1
    }

    // a dummy call just to test if the API works
    mod.predict(test)

    // perform prediction and calculate accuracy manually
    val preds = mod.predictEveryBatch(test)
    test.reset()
    var accSum = 0.0f
    var accCnt = 0
    var i = 0
    while (test.hasNext) {
      val batch = test.next()
      val predLabel: Array[Int] = NDArray.argmax_channel(preds(i)(0)).toArray.map(_.toInt)
      val label = batch.label(0).toArray.map(_.toInt)
      accSum += (predLabel zip label).map { case (py, y) =>
        if (py == y) 1 else 0
      }.sum
      accCnt += predLabel.length
      i += 1
    }
    logger.info(s"Validation Accuracy: {}", accSum / accCnt.toFloat)

    // evaluate on validation set with a evaluation metric
    val (name, value) = mod.score(test, new Accuracy).get
    logger.info("Scored {} = {}", name(0), value(0))
  }

  def main(args: Array[String]): Unit = {
    val inst = new MnistMlp
    val parser = new CmdLineParser(inst)
    try {
      parser.parseArgument(args.toList.asJava)

      val train = IO.MNISTIter(Map(
        "image" -> (inst.dataDir + "train-images-idx3-ubyte"),
        "label" -> (inst.dataDir + "train-labels-idx1-ubyte"),
        "label_name" -> "softmax_label",
        "input_shape" -> "(784,)",
        "batch_size" -> inst.batchSize.toString,
        "shuffle" -> "True",
        "flat" -> "True", "silent" -> "False", "seed" -> "10"))
      val eval = IO.MNISTIter(Map(
        "image" -> (inst.dataDir + "t10k-images-idx3-ubyte"),
        "label" -> (inst.dataDir + "t10k-labels-idx1-ubyte"),
        "label_name" -> "softmax_label",
        "input_shape" -> "(784,)",
        "batch_size" -> inst.batchSize.toString,
        "flat" -> "True", "silent" -> "False"))

      logger.info("Run intermediate level api from beginning.")
      runIntermediateLevelApi(train, eval, inst)
      logger.info("Run intermediate level api, start with last trained epoch.")
      runIntermediateLevelApi(train, eval, inst, loadModelEpoch = inst.numEpoch - 1)
      logger.info("Run high level api")
      runHighLevelApi(train, eval, inst)
    } catch {
      case ex: Exception =>
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
    }
  }
}

class MnistMlp {
  @Option(name = "--data-dir", usage = "the input data directory")
  private val dataDir: String = "mnist/"
  @Option(name = "--batch-size", usage = "the batch size for data iterator")
  private val batchSize: Int = 2
  @Option(name = "--num-epoch", usage = "number of training epoches")
  private val numEpoch: Int = 10
}
