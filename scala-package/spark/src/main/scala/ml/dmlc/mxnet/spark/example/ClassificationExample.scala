package ml.dmlc.mxnet.spark.example

import ml.dmlc.mxnet.spark.MXNet
import ml.dmlc.mxnet.{Symbol, NDArray, Context, Shape}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.kohsuke.args4j.{Option, CmdLineParser}
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

class ClassificationExample
object ClassificationExample {
  private val logger: Logger = LoggerFactory.getLogger(classOf[ClassificationExample])
  def main(args: Array[String]): Unit = {
    val cmdLine = new CommandLine
    val parser: CmdLineParser = new CmdLineParser(cmdLine)
    try {
      parser.parseArgument(args.toList.asJava)
      cmdLine.checkArguments()

      val conf = new SparkConf().setAppName("MXNet")
      val sc = new SparkContext(conf)

      val network = if (cmdLine.model == "mlp") getMlp else getLenet
      val dimension = if (cmdLine.model == "mlp") Shape(784) else Shape(1, 28, 28)
      val devs =
        if (cmdLine.gpus != null) cmdLine.gpus.split(',').map(id => Context.gpu(id.trim.toInt))
        else if (cmdLine.cpus != null) cmdLine.cpus.split(',').map(id => Context.cpu(id.trim.toInt))
        else Array(Context.cpu(0))

      val mxnet = new MXNet()
        .setBatchSize(128)
        .setLabelName("softmax_label")
        .setContext(devs)
        .setDimension(dimension)
        .setNetwork(network)
        .setNumEpoch(cmdLine.numEpoch)
        .setNumServer(cmdLine.numServer)
        .setNumWorker(cmdLine.numWorker)
        .setExecutorJars(cmdLine.jars)
        .setJava(cmdLine.java)

      val trainData = parseRawData(sc, cmdLine.input)
      val start = System.currentTimeMillis
      val model = mxnet.fit(trainData)
      val timeCost = System.currentTimeMillis - start
      logger.info("Training cost {} milli seconds", timeCost)
      model.save(sc, cmdLine.output + "/model")

      logger.info("Now do validation")
      val valData = parseRawData(sc, cmdLine.inputVal)

      val brModel = sc.broadcast(model)
      val res = valData.mapPartitions { data =>
        // get real labels
        import org.apache.spark.mllib.linalg.Vector
        val points = ArrayBuffer.empty[Vector]
        val y = ArrayBuffer.empty[Float]
        while (data.hasNext) {
          val evalData = data.next()
          y += evalData.label.toFloat
          points += evalData.features
        }

        // get predicted labels
        val probArrays = brModel.value.predict(points.toIterator)
        require(probArrays.length == 1)
        val prob = probArrays(0)
        val py = NDArray.argmaxChannel(prob.get)
        require(y.length == py.size, s"${y.length} mismatch ${py.size}")

        // I'm too lazy to calculate the accuracy
        val res = Iterator((y.toArray zip py.toArray).map {
          case (y1, py1) => y1 + "," + py1 }.mkString("\n"))

        py.dispose()
        prob.get.dispose()
        res
      }
      res.saveAsTextFile(cmdLine.output + "/data")

      sc.stop()
    } catch {
      case e: Throwable =>
        logger.error(e.getMessage, e)
        sys.exit(-1)
    }
  }

  private def parseRawData(sc: SparkContext, path: String): RDD[LabeledPoint] = {
    val raw = sc.textFile(path)
    raw.map { s =>
      val parts = s.split(' ')
      val label = java.lang.Double.parseDouble(parts(0))
      val features = Vectors.dense(parts(1).trim().split(',').map(java.lang.Double.parseDouble))
      LabeledPoint(label, features)
    }
  }

  private class CommandLine {
    @Option(name = "--input", usage = "Input training file.")
    val input: String = null
    @Option(name = "--input-val", usage = "Input validation file.")
    val inputVal: String = null
    @Option(name = "--output", usage = "Output inferred result.")
    val output: String = null
    @Option(name = "--jars", usage = "Jars for running MXNet on other nodes.")
    val jars: String = null
    @Option(name = "--num-server", usage = "PS server number")
    val numServer: Int = 1
    @Option(name = "--num-worker", usage = "PS worker number")
    val numWorker: Int = 1
    @Option(name = "--num-epoch", usage = "Number of epochs")
    val numEpoch: Int = 10
    @Option(name = "--java", usage = "Java bin")
    val java: String = "java"
    @Option(name = "--model", usage = "Model definition")
    val model: String = "mlp"
    @Option(name = "--gpus", usage = "the gpus will be used, e.g. '0,1,2,3'")
    val gpus: String = null
    @Option(name = "--cpus", usage = "the cpus will be used, e.g. '0,1,2,3'")
    val cpus: String = null

    def checkArguments(): Unit = {
      require(input != null, "Undefined input path")
      require(numServer > 0, s"Invalid number of servers: $numServer")
      require(numWorker > 0, s"Invalid number of workers: $numWorker")
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
