package ml.dmlc.mxnet.examples.rnn

import scala.io.Source
import ml.dmlc.mxnet.EvalMetric
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.EpochEndCallback
import ml.dmlc.mxnet.Model
import ml.dmlc.mxnet.Symbol
import scala.util.Random

/**
 * @author Depeng Liang
 */
object Utils {

  def readContent(path: String): String = Source.fromFile(path).mkString

  // Build  a vocabulary of what char we have in the content
  def buildVocab(path: String): Map[String, Int] = {
    val content = readContent(path).split("\n")
    var idx = 1 // 0 is left for zero padding
    var theVocab = Map[String, Int]()
    for (line <- content) {
      for (char <- line) {
        val key = s"$char"
        if (!theVocab.contains(key)) {
          theVocab = theVocab + (key -> idx)
          idx += 1
        }
      }
    }
    theVocab
  }

  // We will assign each char with a special numerical id
  def text2Id(sentence: String, theVocab: Map[String, Int]): Array[Int] = {
    val words = for (char <- sentence) yield theVocab(s"$char")
    words.toArray
  }

  // Evaluation
  def perplexity(label: NDArray, pred: NDArray): Float = {
    val shape = label.shape
    val size = shape(0) * shape(1)
    val labelT = {
      val tmp = label.toArray.grouped(shape(1)).toArray
      val result = Array.fill[Float](size)(0f)
      var idx = 0
      for (i <- 0 until shape(1)) {
        for (j <- 0 until shape(0)) {
          result(idx) = tmp(j)(i)
          idx += 1
        }
      }
      result
    }
    var loss = 0f
    val predArray = pred.toArray.grouped(pred.shape(1)).toArray
    for (i <- 0 until pred.shape(0)) {
      loss += -Math.log(Math.max(1e-10, predArray(i)(labelT(i).toInt)).toFloat).toFloat
    }
    loss / size
  }

  def doCheckpoint(prefix: String): EpochEndCallback = new EpochEndCallback {
    override def invoke(epoch: Int, symbol: Symbol,
                        argParams: Map[String, NDArray],
                        auxStates: Map[String, NDArray]): Unit = {
      Model.saveCheckpoint(prefix, epoch + 1, symbol, argParams, auxStates)
    }
  }

  // helper strcuture for prediction
  def makeRevertVocab(vocab: Map[String, Int]): Map[Int, String] = {
    var dic = Map[Int, String]()
    vocab.foreach { case (k, v) =>
      dic = dic + (v -> k)
    }
    dic
  }

  // make input from char
  def makeInput(char: Char, vocab: Map[String, Int], arr: NDArray): Unit = {
    val idx = vocab(s"$char")
    val tmp = NDArray.zeros(1)
    tmp.set(idx)
    arr.set(tmp)
  }

  // helper function for random sample
  def cdf(weights: Array[Float]): Array[Float] = {
    val total = weights.sum
    var result = Array[Float]()
    var cumsum = 0f
    for (w <- weights) {
      cumsum += w
      result = result :+ (cumsum / total)
    }
    result
  }

  def choice(population: Array[String], weights: Array[Float]): String = {
    assert(population.length == weights.length)
    val cdfVals = cdf(weights)
    val x = Random.nextFloat()
    var idx = 0
    var found = false
    for (i <- 0 until cdfVals.length) {
      if (cdfVals(i) >= x && !found) {
        idx = i
        found = true
      }
    }
    population(idx)
  }

  // we can use random output or fixed output by choosing largest probability
  def makeOutput(prob: Array[Float], vocab: Map[Int, String],
                 sample: Boolean = false, temperature: Float = 1f): String = {
    var idx = -1
    val char = if (sample == false) {
      idx = ((-1f, -1) /: prob.zipWithIndex) { (max, elem) =>
        if (max._1 < elem._1) elem else max
      }._2
      if (vocab.contains(idx)) vocab(idx)
      else ""
    } else {
      val fixDict = Array("") ++ (1 until vocab.size + 1).map(i => vocab(i))
      var scaleProb = prob.map(x => if (x < 1e-6) 1e-6 else if (x > 1 - 1e-6) 1 - 1e-6 else x)
      var rescale = scaleProb.map(x => Math.exp(Math.log(x) / temperature).toFloat)
      val sum = rescale.sum.toFloat
      rescale = rescale.map(_ / sum)
      choice(fixDict, rescale)
    }
    char
  }
}
