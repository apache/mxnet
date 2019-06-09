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

package org.apache.mxnetexamples.imclassification

import java.util.concurrent._

import org.apache.mxnetexamples.imclassification.models._
import org.apache.mxnetexamples.imclassification.util.Trainer
import org.apache.mxnet._
import org.apache.mxnetexamples.imclassification.datasets.{MnistIter, SyntheticDataIter}
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable

object TrainModel {
  private val logger = LoggerFactory.getLogger(classOf[TrainModel])

  /**
    * Simple model training and execution
    * @param model The model identifying string
    * @param dataPath Path to location of image data
    * @param numExamples Number of image data examples
    * @param numEpochs Number of epochs to train for
    * @param benchmark Whether to use benchmark synthetic data instead of real image data
    * @return The final validation accuracy
    */
  def test(model: String, dataPath: String, numExamples: Int = 60000,
           numEpochs: Int = 10, benchmark: Boolean = false): Float = {
    ResourceScope.using() {
      val devs = Array(Context.cpu(0))
      val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
      val (dataLoader, net) = dataLoaderAndModel("mnist", model, dataPath,
        numExamples = numExamples, benchmark = benchmark)
      val Acc = Trainer.fit(batchSize = 128, numExamples, devs = devs,
        network = net, dataLoader = dataLoader,
        kvStore = "local", numEpochs = numEpochs)
      logger.info("Finish test fit ...")
      val (_, num) = Acc.get
      num(0)
    }
  }

  /**
    * Gets dataset iterator and model symbol
    * @param dataset The dataset identifying string
    * @param model The model identifying string
    * @param dataDir Path to location of image data
    * @param numLayers The number of model layers (resnet only)
    * @param numExamples The number of examples in the dataset
    * @param benchmark Whether to use benchmark synthetic data instead of real image data
    * @return Data iterator (partially applied function) and model symbol
    */
  def dataLoaderAndModel(dataset: String, model: String, dataDir: String = "",
                         numLayers: Int = 50, numExamples: Int = 60000,
                         benchmark: Boolean = false
                        ): ((Int, KVStore) => (DataIter, DataIter), Symbol) = {
    val (imageShape, numClasses) = dataset match {
      case "mnist" => (List(1, 28, 28), 10)
      case "imagenet" => (List(3, 224, 224), 1000)
      case _ => throw new Exception("Invalid image data collection")
    }

    val List(channels, height, width) = imageShape
    val dataSize: Int = channels * height * width
    val (datumShape, net) = model match {
      case "mlp" => (List(dataSize), MultiLayerPerceptron.getSymbol(numClasses))
      case "lenet" => (List(channels, height, width), Lenet.getSymbol(numClasses))
      case "resnet" => (List(channels, height, width), Resnet.getSymbol(numClasses,
        numLayers, imageShape))
      case _ => throw new Exception("Invalid model name")
    }

    val dataLoader: (Int, KVStore) => (DataIter, DataIter) = if (benchmark) {
      (batchSize: Int, kv: KVStore) => {
        val iter = new SyntheticDataIter(numClasses, batchSize, datumShape, List(), numExamples)
        (iter, iter)
      }
    } else {
      dataset match {
        case "mnist" => MnistIter.getIterator(Shape(datumShape), dataDir)
        case _ => throw new Exception("This image data collection only supports the"
          + "synthetic benchmark iterator.  Use --benchmark to enable")
      }
    }
    (dataLoader, net)
  }

  /**
    * Runs image classification training from CLI with various options
    * @param args CLI args
    */
  def main(args: Array[String]): Unit = {
    val inst = new TrainModel
    val parser: CmdLineParser = new CmdLineParser(inst)
    try {
      ResourceScope.using() {
        parser.parseArgument(args.toList.asJava)

        val dataPath = if (inst.dataDir == null) System.getenv("MXNET_HOME")
        else inst.dataDir

        val (dataLoader, net) = dataLoaderAndModel(inst.dataset, inst.network, dataPath,
          inst.numLayers, inst.numExamples, inst.benchmark)

        val devs =
          if (inst.gpus != null) inst.gpus.split(',').map(id => Context.gpu(id.trim.toInt))
          else if (inst.cpus != null) inst.cpus.split(',').map(id => Context.cpu(id.trim.toInt))
          else Array(Context.cpu(0))

        val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
        envs.put("DMLC_ROLE", inst.role)
        if (inst.schedulerHost != null) {
          require(inst.schedulerPort > 0, "scheduler port not specified")
          envs.put("DMLC_PS_ROOT_URI", inst.schedulerHost)
          envs.put("DMLC_PS_ROOT_PORT", inst.schedulerPort.toString)
          require(inst.numWorker > 0, "Num of workers must > 0")
          envs.put("DMLC_NUM_WORKER", inst.numWorker.toString)
          require(inst.numServer > 0, "Num of servers must > 0")
          envs.put("DMLC_NUM_SERVER", inst.numServer.toString)
          logger.info("Init PS environments")
          KVStoreServer.init(envs.toMap)
        }

        if (inst.role != "worker") {
          logger.info("Start KVStoreServer for scheduler & servers")
          KVStoreServer.start()
        } else {
          Trainer.fit(batchSize = inst.batchSize, numExamples = inst.numExamples, devs = devs,
            network = net, dataLoader = dataLoader,
            kvStore = inst.kvStore, numEpochs = inst.numEpochs,
            modelPrefix = inst.modelPrefix, loadEpoch = inst.loadEpoch,
            lr = inst.lr, lrFactor = inst.lrFactor, lrFactorEpoch = inst.lrFactorEpoch,
            monitorSize = inst.monitor)
          logger.info("Finish fit ...")
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

class TrainModel {
  @Option(name = "--network", usage = "the cnn to use: ['mlp', 'lenet', 'resnet']")
  private val network: String = "mlp"
  @Option(name = "--num-layers", usage = "the number of resnet layers to use")
  private val numLayers: Int = 50
  @Option(name = "--data-dir", usage = "the input data directory")
  private val dataDir: String = "mnist/"

  @Option(name = "--dataset", usage = "the images to classify: ['mnist', 'imagenet']")
  private val dataset: String = "mnist"
  @Option(name = "--benchmark", usage = "Benchmark to use synthetic data to measure performance")
  private val benchmark: Boolean = false

  @Option(name = "--gpus", usage = "the gpus will be used, e.g. '0,1,2,3'")
  private val gpus: String = null
  @Option(name = "--cpus", usage = "the cpus will be used, e.g. '0,1,2,3'")
  private val cpus: String = null
  @Option(name = "--num-examples", usage = "the number of training examples")
  private val numExamples: Int = 60000
  @Option(name = "--batch-size", usage = "the batch size")
  private val batchSize: Int = 128
  @Option(name = "--lr", usage = "the initial learning rate")
  private val lr: Float = 0.1f
  @Option(name = "--model-prefix", usage = "the prefix of the model to load/save")
  private val modelPrefix: String = null
  @Option(name = "--num-epochs", usage = "the number of training epochs")
  private val numEpochs = 10
  @Option(name = "--load-epoch", usage = "load the model on an epoch using the model-prefix")
  private val loadEpoch: Int = -1
  @Option(name = "--kv-store", usage = "the kvstore type")
  private val kvStore = "local"
  @Option(name = "--lr-factor",
    usage = "times the lr with a factor for every lr-factor-epoch epoch")
  private val lrFactor: Float = 1f
  @Option(name = "--lr-factor-epoch", usage = "the number of epoch to factor the lr, could be .5")
  private val lrFactorEpoch: Float = 1f
  @Option(name = "--monitor", usage = "monitor the training process every N batch")
  private val monitor: Int = -1

  @Option(name = "--role", usage = "scheduler/server/worker")
  private val role: String = "worker"
  @Option(name = "--scheduler-host", usage = "Scheduler hostname / ip address")
  private val schedulerHost: String = null
  @Option(name = "--scheduler-port", usage = "Scheduler port")
  private val schedulerPort: Int = 0
  @Option(name = "--num-worker", usage = "# of workers")
  private val numWorker: Int = 1
  @Option(name = "--num-server", usage = "# of servers")
  private val numServer: Int = 1
}

