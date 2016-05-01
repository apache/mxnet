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
object MXNet {
  private val logger: Logger = LoggerFactory.getLogger(classOf[MXNet])
  // TODO: make it configurable
  private val batchSize = 128

  def train(data: RDD[LabeledPoint]): Unit = {
    data.foreachPartition { partition =>
    }
  }

  def main(args: Array[String]): Unit = {
    val cmdLine = new CommandLine
    val parser: CmdLineParser = new CmdLineParser(cmdLine)
    try {
      parser.parseArgument(args.toList.asJava)
      cmdLine.checkArguments()

      val numWorker = cmdLine.numWorker
      val numServer = cmdLine.numServer
      val classpaths = cmdLine.classpaths
      val javabin = cmdLine.java
      val schedulerIP = utils.Network.ipAddress
      val schedulerPort = utils.Network.availablePort

      val conf = new SparkConf().setAppName("MXNet")
      val sc = new SparkContext(conf)

      val trainRaw = sc.textFile(cmdLine.input)
      val trainData = trainRaw.map { s =>
        val parts = s.split(' ')
        val label = java.lang.Double.parseDouble(parts(0))
        val features = Vectors.dense(parts(1).trim().split(',').map(java.lang.Double.parseDouble))
        LabeledPoint(label, features)
      }.repartition(numWorker)

      require(trainData.getNumPartitions == numWorker)

      val dimension = trainData.first().features.size
      logger.debug("Dimension: {}", dimension)
      val numExamples = trainData.count().toInt
      logger.debug("numExamples: {}", numExamples)

      logger.info("Starting scheduler on {}:{}", schedulerIP, schedulerPort)
      val scheduler = new ParameterServer(classpaths, role = "scheduler",
        rootUri = schedulerIP, rootPort = schedulerPort,
        numServer = numServer, numWorker = numWorker, java = javabin)
      require(scheduler.startProcess(), "Failed to start ps scheduler process")

      sc.parallelize(1 to numServer, numServer).foreachPartition { p =>
        logger.info("Starting server ...")
        val server = new ParameterServer(classpaths, role = "server",
          rootUri = schedulerIP, rootPort = schedulerPort,
          numServer = numServer, numWorker = numWorker, java = javabin)
        require(server.startProcess(), "Failed to start ps server process")
      }

      val job = trainData.mapPartitions { partition =>
        val dataIter = new LabeledPointIter(
          partition, dimension, batchSize, labelName = "softmax_label")

        logger.info("Launching worker ...")
        KVStoreServer.init(ParameterServer.buildEnv(role = "worker",
          rootUri = schedulerIP, rootPort = schedulerPort,
          numServer = numServer, numWorker = numWorker))
        val kv = KVStore.create("dist_async")

        val optimizer: Optimizer = new SGD(learningRate = 0.01f,
          momentum = 0.9f, wd = 0.00001f)

        logger.debug("Define model")
        val model = new FeedForward(ctx = Context.cpu(),
          symbol = getMlp,
          numEpoch = 10,
          optimizer = optimizer,
          initializer = new Xavier(factorType = "in", magnitude = 2.34f),
          argParams = null,
          auxParams = null,
          beginEpoch = 0,
          epochSize = numExamples / batchSize / kv.numWorkers)
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
      sc.stop()
    } catch {
      case e: Throwable =>
        logger.error(e.getMessage, e)
        sys.exit(-1)
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
}

class MXNet
