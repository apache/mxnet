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

package ml.dmlc.mxnetexamples.imclassification

import ml.dmlc.mxnet._
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.collection.JavaConverters._

object TrainMnist {
  private val logger = LoggerFactory.getLogger(classOf[TrainMnist])

  // multi-layer perceptron
  def getMlp: Symbol = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")()(Map("data" -> data, "num_hidden" -> 128))
    val act1 = Symbol.Activation(name = "relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")()(Map("data" -> act1, "num_hidden" -> 64))
    val act2 = Symbol.Activation(name = "relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected(name = "fc3")()(Map("data" -> act2, "num_hidden" -> 10))
    val mlp = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc3))
    mlp
  }

  // LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
  // Haffner. "Gradient-based learning applied to document recognition."
  // Proceedings of the IEEE (1998)
  def getLenet: Symbol = {
    val data = Symbol.Variable("data")
    // first conv
    val conv1 = Symbol.Convolution()()(
      Map("data" -> data, "kernel" -> "(5, 5)", "num_filter" -> 20))
    val tanh1 = Symbol.Activation()()(Map("data" -> conv1, "act_type" -> "tanh"))
    val pool1 = Symbol.Pooling()()(Map("data" -> tanh1, "pool_type" -> "max",
                                       "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    // second conv
    val conv2 = Symbol.Convolution()()(
      Map("data" -> pool1, "kernel" -> "(5, 5)", "num_filter" -> 50))
    val tanh2 = Symbol.Activation()()(Map("data" -> conv2, "act_type" -> "tanh"))
    val pool2 = Symbol.Pooling()()(Map("data" -> tanh2, "pool_type" -> "max",
                                       "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    // first fullc
    val flatten = Symbol.Flatten()()(Map("data" -> pool2))
    val fc1 = Symbol.FullyConnected()()(Map("data" -> flatten, "num_hidden" -> 500))
    val tanh3 = Symbol.Activation()()(Map("data" -> fc1, "act_type" -> "tanh"))
    // second fullc
    val fc2 = Symbol.FullyConnected()()(Map("data" -> tanh3, "num_hidden" -> 10))
    // loss
    val lenet = Symbol.SoftmaxOutput(name = "softmax")()(Map("data" -> fc2))
    lenet
  }

  def getIterator(dataShape: Shape)
    (dataDir: String, batchSize: Int, kv: KVStore): (DataIter, DataIter) = {
    val flat = if (dataShape.size == 3) "False" else "True"

    val train = IO.MNISTIter(Map(
      "image" -> (dataDir + "train-images-idx3-ubyte"),
      "label" -> (dataDir + "train-labels-idx1-ubyte"),
      "label_name" -> "softmax_label",
      "input_shape" -> dataShape.toString,
      "batch_size" -> batchSize.toString,
      "shuffle" -> "True",
      "flat" -> flat,
      "num_parts" -> kv.numWorkers.toString,
      "part_index" -> kv.`rank`.toString))

    val eval = IO.MNISTIter(Map(
      "image" -> (dataDir + "t10k-images-idx3-ubyte"),
      "label" -> (dataDir + "t10k-labels-idx1-ubyte"),
      "label_name" -> "softmax_label",
      "input_shape" -> dataShape.toString,
      "batch_size" -> batchSize.toString,
      "flat" -> flat,
      "num_parts" -> kv.numWorkers.toString,
      "part_index" -> kv.`rank`.toString))

    (train, eval)
  }


  def main(args: Array[String]): Unit = {
    val inst = new TrainMnist
    val parser: CmdLineParser = new CmdLineParser(inst)
    try {
      parser.parseArgument(args.toList.asJava)

      val dataPath = if (inst.dataDir == null) System.getenv("MXNET_DATA_DIR")
        else inst.dataDir

      val (dataShape, net) =
        if (inst.network == "mlp") (Shape(784), getMlp)
        else (Shape(1, 28, 28), getLenet)

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
        ModelTrain.fit(dataDir = inst.dataDir,
          batchSize = inst.batchSize, numExamples = inst.numExamples, devs = devs,
          network = net, dataLoader = getIterator(dataShape),
          kvStore = inst.kvStore, numEpochs = inst.numEpochs,
          modelPrefix = inst.modelPrefix, loadEpoch = inst.loadEpoch,
          lr = inst.lr, lrFactor = inst.lrFactor, lrFactorEpoch = inst.lrFactorEpoch,
          monitorSize = inst.monitor)
        logger.info("Finish fit ...")
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

class TrainMnist {
  @Option(name = "--network", usage = "the cnn to use: ['mlp', 'lenet']")
  private val network: String = "mlp"
  @Option(name = "--data-dir", usage = "the input data directory")
  private val dataDir: String = "mnist/"
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
