package ml.dmlc.mxnet.examples.rnn

import ml.dmlc.mxnet.Callback.Speedometer
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.examples.rnn.BucketIo.BucketSentenceIter
import ml.dmlc.mxnet.optimizer.SGD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer

/**
 * Bucketing LSTM examples
 * @author Yizhi Liu
 */
class LstmBucketing
object LstmBucketing {
  private val logger: Logger = LoggerFactory.getLogger(classOf[LstmBucketing])

  def perplexity(label: NDArray, pred: NDArray): Float = {
    val batchSize = label.shape(0)
    // TODO: NDArray transpose
    val labelArr = Array.fill(label.size)(0)
    (0 until batchSize).foreach(row => {
      val labelRow = label.slice(row)
      labelRow.toArray.zipWithIndex.foreach { case (l, col) =>
        labelArr(col * batchSize + row) = l.toInt
      }
    })
    var loss = .0
    (0 until pred.shape(0)).foreach(i =>
      loss -= Math.log(Math.max(1e-10f, pred.slice(i).toArray(labelArr(i))))
    )
    Math.exp(loss / labelArr.length).toFloat
  }

  def main(args: Array[String]): Unit = {
    val batchSize = 32
    val buckets = Array(10, 20, 30, 40, 50, 60)
    val numHidden = 200
    val numEmbed = 200
    val numLstmLayer = 2

    val numEpoch = 25
    val learningRate = 0.01f
    val momentum = 0.0f

    val contexts = Array(Context.cpu(0))

    logger.info("Building vocab ...")
    val vocab = BucketIo.defaultBuildVocab("./data/ptb.test.txt") // TODO

    class BucketSymGen extends SymbolGenerator {
      override def generate(key: AnyRef): Symbol = {
        val seqLen = key.asInstanceOf[Int]
        Lstm.lstmUnroll(numLstmLayer, seqLen, vocab.size,
          numHidden = numHidden, numEmbed = numEmbed, numLabel = vocab.size)
      }
    }

    val initC = (0 until numLstmLayer).map(l =>
      (s"l${l}_init_c", (batchSize, numHidden))
    )
    val initH = (0 until numLstmLayer).map(l =>
      (s"l${l}_init_h", (batchSize, numHidden))
    )
    val initStates = initC ++ initH

    val dataTrain = new BucketSentenceIter("./data/ptb.test.txt", vocab,
      buckets, batchSize, initStates)
    val dataVal = new BucketSentenceIter("./data/ptb.test.txt", vocab,
      buckets, batchSize, initStates)

    logger.info("Start training ...")
    val model = FeedForward.newBuilder(new BucketSymGen())
      .setContext(contexts)
      .setNumEpoch(numEpoch)
      .setOptimizer(new SGD(learningRate = learningRate, momentum = momentum, wd = 0.00001f))
      .setInitializer(new Xavier(factorType = "in", magnitude = 2.34f))
      .setTrainData(dataTrain)
      .setEvalData(dataVal)
      .setEvalMetric(new CustomMetric(perplexity, name = "perplexity"))
      .setBatchEndCallback(new Speedometer(batchSize, 50))
      .build()
  }
}
