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

package org.apache.mxnetexamples.rnn

import org.apache.mxnet._
import org.apache.mxnetexamples.InferBase
import org.apache.mxnetexamples.benchmark.CLIParserBase
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

/**
 * Follows the demo, to test the char rnn:
 * https://github.com/dmlc/mxnet/blob/master/example/rnn/char-rnn.ipynb
 */
object TestCharRnn {

  private val logger = LoggerFactory.getLogger(classOf[TrainCharRnn])

  def runInferenceCharRNN(dataPath: String, modelPrefix: String, starterSentence : String): Unit = {
    ResourceScope.using() {
      // The batch size for training
      val batchSize = 32
      // We can support various length input
      // For this problem, we cut each input sentence to length of 129
      // So we only need fix length bucket
      val buckets = List(129)
      // hidden unit in LSTM cell
      val numHidden = 512
      // embedding dimension, which is, map a char to a 256 dim vector
      val numEmbed = 256
      // number of lstm layer
      val numLstmLayer = 3

      // build char vocabluary from input
      val vocab = Utils.buildVocab(dataPath)

      // load from check-point
      val (_, argParams, _) = Model.loadCheckpoint(modelPrefix, 75)

      // build an inference model
      val model = new RnnModel.LSTMInferenceModel(numLstmLayer, vocab.size + 1,
        numHidden = numHidden, numEmbed = numEmbed,
        numLabel = vocab.size + 1, argParams = argParams, dropout = 0.2f)

      // generate a sequence of 1200 chars
      val seqLength = 1200
      val inputNdarray = NDArray.zeros(1)
      val revertVocab = Utils.makeRevertVocab(vocab)

      // Feel free to change the starter sentence
      var output = starterSentence
      val randomSample = true
      var newSentence = true
      val ignoreLength = output.length()

      for (i <- 0 until seqLength) {
        if (i <= ignoreLength - 1) Utils.makeInput(output(i), vocab, inputNdarray)
        else Utils.makeInput(output.takeRight(1)(0), vocab, inputNdarray)
        val prob = model.forward(inputNdarray, newSentence)
        newSentence = false
        val nextChar = Utils.makeOutput(prob, revertVocab, randomSample)
        if (nextChar == "") newSentence = true
        if (i >= ignoreLength) output = output ++ nextChar
      }

      // Let's see what we can learned from char in Obama's speech.
      logger.info(output)
    }
  }

  def main(args: Array[String]): Unit = {
    val stcr = new CLIParser
    val parser: CmdLineParser = new CmdLineParser(stcr)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(stcr.dataPath != null && stcr.modelPrefix != null && stcr.starterSentence != null)
      runInferenceCharRNN(stcr.dataPath, stcr.modelPrefix, stcr.starterSentence)
    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class CLIParser extends CLIParserBase {
  @Option(name = "--data-path", usage = "the input train data file")
  val dataPath: String = "./data/obama.txt"
  @Option(name = "--model-prefix", usage = "the model prefix")
  val modelPrefix: String = "./model/obama"
  @Option(name = "--starter-sentence", usage = "the starter sentence")
  val starterSentence: String = "The joke"
}

class TestCharRnn(CLIParser: CLIParser) extends InferBase {

  private var vocab : Map[String, Int] = null

  override def loadModel(context: Array[Context], batchInference : Boolean = false): Any = {
    val batchSize = 32
    val buckets = List(129)
    val numHidden = 512
    val numEmbed = 256
    val numLstmLayer = 3
    val (_, argParams, _) = Model.loadCheckpoint(CLIParser.modelPrefix, 75)
    this.vocab = Utils.buildVocab(CLIParser.dataPath)
    var ctx = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      ctx = Context.gpu()
    }
    val model = new RnnModel.LSTMInferenceModel(numLstmLayer, vocab.size + 1,
      numHidden = numHidden, numEmbed = numEmbed,
      numLabel = vocab.size + 1, argParams = argParams, dropout = 0.2f, ctx = ctx)
    model
  }

  override def loadSingleData(): Any = {
    val revertVocab = Utils.makeRevertVocab(vocab)
    revertVocab
  }

  override def runSingleInference(loadedModel: Any, input: Any): Any = {
    val model = loadedModel.asInstanceOf[RnnModel.LSTMInferenceModel]
    val revertVocab = input.asInstanceOf[Map[Int, String]]
    // generate a sequence of 1200 chars
    val seqLength = 1200
    val inputNdarray = NDArray.zeros(1)
    // Feel free to change the starter sentence
    var output = CLIParser.starterSentence
    val randomSample = true
    var newSentence = true
    val ignoreLength = output.length()

    for (i <- 0 until seqLength) {
      if (i <= ignoreLength - 1) Utils.makeInput(output(i), vocab, inputNdarray)
      else Utils.makeInput(output.takeRight(1)(0), vocab, inputNdarray)
      val prob = model.forward(inputNdarray, newSentence)
      newSentence = false
      val nextChar = Utils.makeOutput(prob, revertVocab, randomSample)
      if (nextChar == "") newSentence = true
      if (i >= ignoreLength) output = output ++ nextChar
    }
    output
  }

  override def loadBatchFileList(batchSize: Int): List[Any] = null
  override def loadInputBatch(source: Any): Any = null
  override def runBatchInference(loadedModel: Any, input: Any): Any = null
}
