package ml.dmlc.mxnet.examples.module

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.module.Module
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

  def main(args: Array[String]): Unit = {
    val inst = new MnistMlp
    val parser: CmdLineParser = new CmdLineParser(inst)
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

      // Intermediate-level API
      val mod = new Module(getSymbol)
      mod.bind(dataShapes = train.provideData, labelShapes = Some(train.provideLabel))
      mod.initParams()
      mod.initOptimizer(optimizer = new SGD(learningRate = 0.01f, momentum = 0.9f))

      val metric = new Accuracy()

      for (epoch <- 0 until inst.numEpoch) {
        while (train.hasNext) {
          val batch = train.next()
          mod.forward(batch)
          mod.updateMetric(metric, batch.label)
          mod.backward()
          mod.update()
        }

        val (name, value) = metric.get
        logger.info(s"epoch $epoch $name=$value")
        metric.reset()
        train.reset()
      }
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
