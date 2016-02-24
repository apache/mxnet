package ml.dmlc.mxnet.examples

import ml.dmlc.mxnet.Base.Shape
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.examples.imclassification.ModelTrain
import ml.dmlc.mxnet.optimizer.SGD
import org.slf4j.LoggerFactory

object TrainMnist {
  private val logger = LoggerFactory.getLogger(classOf[TrainMnist])

  // multi-layer perceptron
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

  def getIterator(dataShape: Shape)
    (dataDir: String, batchSize: Int, kv: KVStore): (DataIter, DataIter) = {
    val flat = if (dataShape.size == 3) "False" else "True"

    val train = IO.MNISTIter(Map(
      "image" -> (dataDir + "train-images-idx3-ubyte"),
      "label" -> (dataDir + "train-labels-idx1-ubyte"),
      "label_name" -> "softmax_label",
      "input_shape" -> s"(${dataShape.mkString(",")})",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "True",
      "flat" -> flat,
      "num_parts" -> kv.numWorkers.toString,
      "part_index" -> kv.`rank`.toString))

    val eval = IO.MNISTIter(Map(
      "image" -> (dataDir + "t10k-images-idx3-ubyte"),
      "label" -> (dataDir + "t10k-labels-idx1-ubyte"),
      "label_name" -> "softmax_label",
      "input_shape" -> s"(${dataShape.mkString(",")})",
      "batch_size" -> batchSize.toString,
      "flat" -> flat,
      "num_parts" -> kv.numWorkers.toString,
      "part_index" -> kv.`rank`.toString))

    (train, eval)
  }


  def main(args: Array[String]): Unit = {
    ModelTrain.fit(dataDir = "/Users/lewis/Workspace/source-codes/forks/mxnet/data/",
      batchSize = 128, numExamples = 60000, devs = Context.cpu(0),
      network = getMlp, dataLoader = getIterator(Vector(784)),
      kvStore = "local", numEpochs = 10, modelPrefix = null, loadEpoch = -1,
      lr = 0.1f, lrFactor = 1f, lrFactorEpoch = 1f,
      clipGradient = 0f)

    logger.info("Finish fit ...")
  }
}

class TrainMnist
