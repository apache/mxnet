package ml.dmlc.mxnet.examples

import ml.dmlc.mxnet._
import ml.dmlc.mxnet.optimizer.SGD
import org.slf4j.LoggerFactory

object TrainMnist {
  private val logger = LoggerFactory.getLogger(classOf[NDArray])

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

  def main(args: Array[String]): Unit = {
    val batchSize = 100
    // get data
    val trainDataIter = IO.MNISTIter(Map(
      "image" -> "data/train-images-idx3-ubyte",
      "label" -> "data/train-labels-idx1-ubyte",
      "input_shape" -> "(784, )",
      "label_name" -> "softmax_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "1",
      "silent" -> "0",
      "seed" -> "10"))

    val valDataIter = IO.MNISTIter(Map(
      "image" -> "data/t10k-images-idx3-ubyte",
      "label" -> "data/t10k-labels-idx1-ubyte",
      "input_shape" -> "(784, )",
      "label_name" -> "softmax_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "1",
      "silent" -> "0"))

    val model = new FeedForward(getMlp, Context.cpu(), numEpoch = 1,
      optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
    model.fit(trainDataIter, valDataIter)
    logger.info("Finish fit ...")
  }
}
