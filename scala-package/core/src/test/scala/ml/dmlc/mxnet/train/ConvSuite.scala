package ml.dmlc.mxnet.train

import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet._
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer
import scala.sys.process._

class ConvSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[ConvSuite])

  test("train mnist") {
    // symbol net
    val batchSize = 100

    val data = Symbol.Variable("data")
    val conv1 = Symbol.Convolution(Map("data" -> data, "name" -> "conv1",
                                       "num_filter" -> 32, "kernel" -> (3, 3), "stride" -> (2, 2)))
    val bn1 = Symbol.BatchNorm(Map("data" -> conv1, "name" -> "bn1"))
    val act1 = Symbol.Activation(Map("data" -> bn1, "name" -> "relu1", "act_type" -> "relu"))
    val mp1 = Symbol.Pooling(Map("data" -> act1, "name" -> "mp1",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))

    val conv2 = Symbol.Convolution(Map("data" -> mp1, "name" -> "conv2", "num_filter" -> 32,
                                       "kernel" -> (3, 3), "stride" -> (2, 2)))
    val bn2 = Symbol.BatchNorm(Map("data" -> conv2, "name" -> "bn2"))
    val act2 = Symbol.Activation(Map("data" -> bn2, "name" -> "relu2", "act_type" -> "relu"))
    val mp2 = Symbol.Pooling(Map("data" -> act2, "name" -> "mp2",
                                 "kernel" -> (2, 2), "stride" -> (2, 2), "pool_type" -> "max"))

    val fl = Symbol.Flatten(Map("data" -> mp2, "name" -> "flatten"))
    val fc2 = Symbol.FullyConnected(Map("data" -> fl, "name" -> "fc2", "num_hidden" -> 10))
    val softmax = Symbol.SoftmaxOutput(Map("data" -> fc2, "name" -> "sm"))

    val numEpoch = 1
    val model = new FeedForward(softmax, Context.cpu(), numEpoch = numEpoch,
      optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))

    // get data
    "./scripts/get_mnist_data.sh" !
    val trainDataIter = IO.MNISTIter(Map(
      "image" -> "data/train-images-idx3-ubyte",
      "label" -> "data/train-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0",
      "silent" -> "0",
      "seed" -> "10"))

    val valDataIter = IO.MNISTIter(Map(
      "image" -> "data/t10k-images-idx3-ubyte",
      "label" -> "data/t10k-labels-idx1-ubyte",
      "data_shape" -> "(1, 28, 28)",
      "label_name" -> "sm_label",
      "batch_size" -> batchSize.toString,
      "shuffle" -> "1",
      "flat" -> "0", "silent" -> "0"))

    model.fit(trainDataIter, valDataIter)
    logger.info("Finish fit ...")

    val probArrays = model.predict(valDataIter)
    assert(probArrays.length === 1)
    val prob = probArrays(0)
    logger.info("Finish predict ...")

    valDataIter.reset()
    val labels = ListBuffer.empty[NDArray]
    var evalData = valDataIter.next()
    while (evalData != null) {
      labels += evalData.label(0).copy()
      evalData = valDataIter.next()
    }
    val y = NDArray.concatenate(labels)

    val py = NDArray.argmaxChannel(prob)
    assert(y.shape === py.shape)

    var numCorrect = 0
    var numInst = 0
    for ((labelElem, predElem) <- y.toArray zip py.toArray) {
      if (labelElem == predElem) {
        numCorrect += 1
      }
      numInst += 1
    }
    val acc = numCorrect.toFloat / numInst
    logger.info(s"Final accuracy = $acc")
    assert(acc > 0.96)
  }
}
