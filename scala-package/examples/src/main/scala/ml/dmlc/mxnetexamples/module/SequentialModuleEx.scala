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
import ml.dmlc.mxnet.module.{FitParams, Module, SequentialModule}
import ml.dmlc.mxnet.DataDesc._
import ml.dmlc.mxnet.optimizer.SGD
import org.kohsuke.args4j.{Option, CmdLineParser}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._

object SequentialModuleEx {
  private val logger = LoggerFactory.getLogger(classOf[SequentialModuleEx])

  def getSeqModule(): SequentialModule = {
    val contexts = Array(Context.cpu(), Context.cpu())

    // module1
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected("fc1")()(Map("data" -> data, "num_hidden" -> 128))
    val act1 = Symbol.Activation("relu1")()(Map("data" -> fc1, "act_type" -> "relu"))

    val mod1 = new Module(act1, labelNames = null, contexts = contexts(0))

    // module2
    val data2 = Symbol.Variable("data")
    val fc2 = Symbol.FullyConnected("fc2")()(Map("data" -> data2, "num_hidden" -> 64))
    val act2 = Symbol.Activation("relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected("fc3")()(Map("data" -> act2, "num_hidden" -> 10))
    val softmax = Symbol.SoftmaxOutput("softmax")()(Map("data" -> fc3))

    val mod2 = new Module(softmax, contexts = contexts(1))

    // Container module
    val modSeq = new SequentialModule()
    modSeq.add(mod1).add(mod2, ("take_labels", true), ("auto_wiring", true))
    modSeq
  }

  def runIntermediateLevelApi(train: DataIter, eval: DataIter,
    cmdLine: SequentialModuleEx): Unit = {
    // Intermediate-level API
    val modSeq = getSeqModule()
    modSeq.bind(dataShapes = train.provideData, labelShapes = Some(train.provideLabel))
    if (cmdLine.loadModelPath != null) {
      logger.info(s"Load checkpoint from ${cmdLine.loadModelPath}")
      modSeq.loadParams(cmdLine.loadModelPath)
     } else modSeq.initParams()

     modSeq.initOptimizer(optimizer = new SGD(learningRate = cmdLine.lr, momentum = 0.9f))

    val metric = new Accuracy()

    for (epoch <- 0 until cmdLine.numEpoch) {
      while (train.hasNext) {
        val batch = train.next()
        modSeq.forward(batch)
        modSeq.updateMetric(metric, batch.label)
        modSeq.backward()
        modSeq.update()
      }

      val fname = "%s-%04d.params".format(s"${cmdLine.saveModelPath}/seqModule", epoch)
      modSeq.saveParams(fname)

      val (name, value) = metric.get
      logger.info(s"epoch $epoch $name=$value")
      metric.reset()
      train.reset()
    }
  }

  def runHighLevelApi(train: DataIter, test: DataIter, cmdLine: SequentialModuleEx): Unit = {
    // High-level API
    train.reset()
    val modSeq = getSeqModule()
    val fitParams = new FitParams
    fitParams.setOptimizer(new SGD(learningRate = cmdLine.lr, momentum = 0.9f))
    modSeq.fit(train, evalData = scala.Option(test),
        numEpoch = cmdLine.numEpoch, fitParams = fitParams)

    // prediction iterator API
    var iBatch = 0
    test.reset()
    while (test.hasNext) {
      val batch = test.next()
      val preds = modSeq.predict(batch)
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
    modSeq.predict(test)

    // perform prediction and calculate accuracy manually
    val preds = modSeq.predictEveryBatch(test)
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
    logger.info(s"Validation Accuracy: ${accSum / accCnt.toFloat}")

    // evaluate on validation set with a evaluation metric
    val (name, value) = modSeq.score(test, new Accuracy).get
    logger.info(s"Scored $name = $value")
  }

  def main(args: Array[String]): Unit = {
    val alex = new SequentialModuleEx
    val parser = new CmdLineParser(alex)
    try {
      parser.parseArgument(args.toList.asJava)
      require(alex.dataDir != null)

      val trainDataIter = IO.MNISTIter(Map(
        "image" -> s"${alex.dataDir}/train-images-idx3-ubyte",
        "label" -> s"${alex.dataDir}/train-labels-idx1-ubyte",
        "label_name" -> "softmax_label",
        "input_shape" -> "(784,)",
        "batch_size" -> alex.batchSize.toString,
        "shuffle" -> "True",
        "flat" -> "True", "silent" -> "False", "seed" -> "10"))
      val evalDataIter = IO.MNISTIter(Map(
        "image" -> s"${alex.dataDir}/t10k-images-idx3-ubyte",
        "label" -> s"${alex.dataDir}/t10k-labels-idx1-ubyte",
        "label_name" -> "softmax_label",
        "input_shape" -> "(784,)",
        "batch_size" -> alex.batchSize.toString,
        "flat" -> "True", "silent" -> "False"))

      logger.info("Run intermediate level api from beginning.")
      runIntermediateLevelApi(trainDataIter, evalDataIter, alex)
      logger.info("Run high level api")
      runHighLevelApi(trainDataIter, evalDataIter, alex)

    } catch {
      case ex: Exception =>
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
    }
  }
}

class SequentialModuleEx {
  @Option(name = "--data-dir", usage = "the input data directory")
  private val dataDir: String = null
  @Option(name = "--lr", usage = "the initial learning rate")
  private val lr: Float = 0.01f
  @Option(name = "--batch-size", usage = "the batch size for data iterator")
  private val batchSize: Int = 100
  @Option(name = "--num-epoch", usage = "number of training epoches")
  private val numEpoch: Int = 100
  @Option(name = "--save-model-path", usage = "the model saving path")
  private val saveModelPath: String = ""
  @Option(name = "--load-model-path", usage = "the model to be loaded")
  private val loadModelPath: String = null
}
