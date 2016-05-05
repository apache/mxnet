package ml.dmlc.mxnet.spark

import ml.dmlc.mxnet.Callback.Speedometer
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.spark.io.LabeledPointIter
import ml.dmlc.mxnet._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable
import scala.collection.JavaConverters._

/**
 * MXNet Training On Spark
 * @author Yizhi Liu
 */
class MXNet extends Serializable {
  private val logger: Logger = LoggerFactory.getLogger(classOf[MXNet])
  private val params: MXNetParams = new MXNetParams

  def setBatchSize(batchSize: Int): this.type = {
    params.batchSize = batchSize
    this
  }

  def setDimension(dimension: Shape): this.type = {
    params.dimension = dimension
    this
  }

  def setNetwork(network: Symbol): this.type = {
    // TODO: params.network = network
    this
  }

  def setContext(ctx: Context): this.type = {
    params.context = ctx
    this
  }

  def setNumWorker(numWorker: Int): this.type = {
    params.numWorker = numWorker
    this
  }

  def setNumServer(numServer: Int): this.type = {
    params.numServer = numServer
    this
  }

  def setLabelName(labelName: String): this.type = {
    params.labelName = labelName
    this
  }

  // TODO: upload to a shared storage from driver
  def setExecutorClasspath(classpath: String): this.type = {
    params.classpath = classpath
    this
  }

  def setJava(java: String): this.type = {
    params.javabin = java
    this
  }

  def train(sc: SparkContext, data: RDD[LabeledPoint]): Unit = {
    val trainData = {
      if (params.numWorker > data.partitions.length) {
        logger.info("repartitioning training set to {} partitions", params.numWorker)
        data.repartition(params.numWorker)
      } else if (params.numWorker < data.partitions.length) {
        logger.info("repartitioning training set to {} partitions", params.numWorker)
        data.coalesce(params.numWorker)
      } else {
        data
      }
    }

    logger.debug("Dimension: {}", params.dimension)
    val numExamples = trainData.count().toInt
    logger.debug("numExamples: {}", numExamples)

    val schedulerIP = utils.Network.ipAddress
    val schedulerPort = utils.Network.availablePort
    // TODO: check ip & port available
    logger.info("Starting scheduler on {}:{}", schedulerIP, schedulerPort)
    val scheduler = new ParameterServer(params.classpath, role = "scheduler",
      rootUri = schedulerIP, rootPort = schedulerPort,
      numServer = params.numServer, numWorker = params.numWorker, java = params.javabin)
    require(scheduler.startProcess(), "Failed to start ps scheduler process")

    //val broadcastParams = sc.broadcast(params)
    sc.parallelize(1 to params.numServer, params.numServer).foreachPartition { p =>
      logger.info("Starting server ...")
      val server = new ParameterServer(params.classpath,
        role = "server",
        rootUri = schedulerIP, rootPort = schedulerPort,
        numServer = params.numServer,
        numWorker = params.numWorker,
        java = params.javabin)
      require(server.startProcess(), "Failed to start ps server process")
    }

    val job = trainData.mapPartitions { partition =>
      val dataIter = new LabeledPointIter(
        partition, params.dimension,
        params.batchSize,
        labelName = params.labelName)

      logger.info("Launching worker ...")
      logger.info("Batch {}", params.batchSize)
      KVStoreServer.init(ParameterServer.buildEnv(role = "worker",
        rootUri = schedulerIP, rootPort = schedulerPort,
        numServer = params.numServer,
        numWorker = params.numWorker))
      val kv = KVStore.create("dist_sync")

      val optimizer: Optimizer = new SGD(learningRate = 0.01f,
        momentum = 0.9f, wd = 0.00001f)

      logger.debug("Define model")
      val model = new FeedForward(ctx = Context.cpu(),
        // TODO: symbol = params.network,
        symbol = MXNet.getMlp,
        numEpoch = 10,
        optimizer = optimizer,
        initializer = new Xavier(factorType = "in", magnitude = 2.34f),
        argParams = null,
        auxParams = null,
        beginEpoch = 0,
        epochSize = numExamples / params.batchSize / kv.numWorkers)
      logger.info("Start training ...")
      model.fit(trainData = dataIter,
        evalData = null,
        evalMetric = new Accuracy(),
        kvStore = kv)

      logger.info("Training finished")
      kv.dispose()
      Iterator(new MXNetModel(model))
    }.cache()

    job.foreachPartition(() => _)

    logger.info("Waiting for scheduler ...")
    scheduler.waitFor()
  }
}

object MXNet {
  private val logger: Logger = LoggerFactory.getLogger(classOf[MXNet])
  def main(args: Array[String]): Unit = {
    val cmdLine = new CommandLine
    val parser: CmdLineParser = new CmdLineParser(cmdLine)
    try {
      parser.parseArgument(args.toList.asJava)
      cmdLine.checkArguments()

      val conf = new SparkConf().setAppName("MXNet")
      val sc = new SparkContext(conf)

      val trainRaw = sc.textFile(cmdLine.input)
      val trainData = trainRaw.map { s =>
        val parts = s.split(' ')
        val label = java.lang.Double.parseDouble(parts(0))
        val features = Vectors.dense(parts(1).trim().split(',').map(java.lang.Double.parseDouble))
        LabeledPoint(label, features)
      }

      val mxnet = new MXNet()
        .setBatchSize(128)
        .setLabelName("softmax_label")
        .setContext(Context.cpu())
        //.setDimension(Shape(1, 28, 28))
        .setDimension(Shape(784))
        .setNetwork(getMlp)
        .setNumServer(cmdLine.numServer)
        .setNumWorker(cmdLine.numWorker)
        .setExecutorClasspath(cmdLine.classpaths)
        .setJava(cmdLine.java)
      mxnet.train(sc, trainData)

      sc.stop()
    } catch {
      case e: Throwable =>
        logger.error(e.getMessage, e)
        sys.exit(-1)
    }
  }

  private class CommandLine {
    @Option(name = "--input", usage = "Input training file.")
    val input: String = null
    @Option(name = "--jars", usage = "Jars for running MXNet on other nodes.")
    val jars: String = null
    @Option(name = "--num-server", usage = "PS server number")
    val numServer: Int = 1
    @Option(name = "--num-worker", usage = "PS worker number")
    val numWorker: Int = 1
    @Option(name = "--java", usage = "Java bin")
    val java: String = "java"

    def checkArguments(): Unit = {
      require(input != null, "Undefined input path")
      require(numServer > 0, s"Invalid number of servers: $numServer")
      require(numWorker > 0, s"Invalid number of workers: $numWorker")
    }

    def classpaths = {
      if (jars == null) null
      else jars.replace(",", ":")
    }
  }

  def getMlp: Symbol = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected(name = "fc1")(Map("data" -> data, "num_hidden" -> 128))
    val act1 = Symbol.Activation(name = "relu1")(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected(name = "fc2")(Map("data" -> act1, "num_hidden" -> 64))
    val act2 = Symbol.Activation(name = "relu2")(Map("data" -> fc2, "act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected(name = "fc3")(Map("data" -> act2, "num_hidden" -> 10))
    val mlp = Symbol.SoftmaxOutput(name = "softmax")(Map("data" -> fc3))
    mlp
  }

  // LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
  // Haffner. "Gradient-based learning applied to document recognition."
  // Proceedings of the IEEE (1998)
  def getLenet: Symbol = {
    val data = Symbol.Variable("data")
    // first conv
    val conv1 = Symbol.Convolution()(Map("data" -> data, "kernel" -> "(5, 5)", "num_filter" -> 20))
    val tanh1 = Symbol.Activation()(Map("data" -> conv1, "act_type" -> "tanh"))
    val pool1 = Symbol.Pooling()(Map("data" -> tanh1, "pool_type" -> "max",
      "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    // second conv
    val conv2 = Symbol.Convolution()(Map("data" -> pool1, "kernel" -> "(5, 5)", "num_filter" -> 50))
    val tanh2 = Symbol.Activation()(Map("data" -> conv2, "act_type" -> "tanh"))
    val pool2 = Symbol.Pooling()(Map("data" -> tanh2, "pool_type" -> "max",
      "kernel" -> "(2, 2)", "stride" -> "(2, 2)"))
    // first fullc
    val flatten = Symbol.Flatten()(Map("data" -> pool2))
    val fc1 = Symbol.FullyConnected()(Map("data" -> flatten, "num_hidden" -> 500))
    val tanh3 = Symbol.Activation()(Map("data" -> fc1, "act_type" -> "tanh"))
    // second fullc
    val fc2 = Symbol.FullyConnected()(Map("data" -> tanh3, "num_hidden" -> 10))
    // loss
    val lenet = Symbol.SoftmaxOutput(name = "softmax")(Map("data" -> fc2))
    lenet
  }
}
