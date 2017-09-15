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


package ml.dmlc.mxnetexamples.rnn

import ml.dmlc.mxnet.Callback.Speedometer
import ml.dmlc.mxnet._
import BucketIo.BucketSentenceIter
import ml.dmlc.mxnet.optimizer.SGD
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._
import ml.dmlc.mxnet.module.BucketingModule
import ml.dmlc.mxnet.module.FitParams

/**
 * Bucketing LSTM examples
 * @author Yizhi Liu
 */
class LstmBucketing {
  @Option(name = "--data-train", usage = "training set")
  private val dataTrain: String = "example/rnn/ptb.train.txt"
  @Option(name = "--data-val", usage = "validation set")
  private val dataVal: String = "example/rnn/ptb.valid.txt"
  @Option(name = "--num-epoch", usage = "the number of training epoch")
  private val numEpoch: Int = 5
  @Option(name = "--gpus", usage = "the gpus will be used, e.g. '0,1,2,3'")
  private val gpus: String = null
  @Option(name = "--cpus", usage = "the cpus will be used, e.g. '0,1,2,3'")
  private val cpus: String = null
  @Option(name = "--save-model-path", usage = "the model saving path")
  private val saveModelPath: String = "model/lstm"
}

object LstmBucketing {
  private val logger: Logger = LoggerFactory.getLogger(classOf[LstmBucketing])

  def perplexity(label: NDArray, pred: NDArray): Float = {
    pred.waitToRead()
    val labelArr = label.T.toArray.map(_.toInt)
    var loss = .0
    (0 until pred.shape(0)).foreach(i =>
      loss -= Math.log(Math.max(1e-10f, pred.slice(i).toArray(labelArr(i))))
    )
    Math.exp(loss / labelArr.length).toFloat
  }

  def main(args: Array[String]): Unit = {
    val inst = new LstmBucketing
    val parser: CmdLineParser = new CmdLineParser(inst)
    try {
      parser.parseArgument(args.toList.asJava)
      val contexts =
        if (inst.gpus != null) inst.gpus.split(',').map(id => Context.gpu(id.trim.toInt))
        else if (inst.cpus != null) inst.cpus.split(',').map(id => Context.cpu(id.trim.toInt))
        else Array(Context.cpu(0))

      val batchSize = 32
      val buckets = Array(10, 20, 30, 40, 50, 60)
      val numHidden = 200
      val numEmbed = 200
      val numLstmLayer = 2

      logger.info("Building vocab ...")
      val vocab = BucketIo.defaultBuildVocab(inst.dataTrain)

      def BucketSymGen(key: AnyRef):
        (Symbol, IndexedSeq[String], IndexedSeq[String]) = {
        val seqLen = key.asInstanceOf[Int]
        val sym = Lstm.lstmUnroll(numLstmLayer, seqLen, vocab.size,
          numHidden = numHidden, numEmbed = numEmbed, numLabel = vocab.size)
        (sym, IndexedSeq("data"), IndexedSeq("softmax_label"))
      }

      val initC = (0 until numLstmLayer).map(l =>
        (s"l${l}_init_c_beta", (batchSize, numHidden))
      )
      val initH = (0 until numLstmLayer).map(l =>
        (s"l${l}_init_h_beta", (batchSize, numHidden))
      )
      val initStates = initC ++ initH

      val dataTrain = new BucketSentenceIter(inst.dataTrain, vocab,
        buckets, batchSize, initStates)
      val dataVal = new BucketSentenceIter(inst.dataVal, vocab,
        buckets, batchSize, initStates)

      val model = new BucketingModule(
        symGen = BucketSymGen,
        defaultBucketKey = dataTrain.defaultBucketKey,
        contexts = contexts)

      val fitParams = new FitParams()
      fitParams.setEvalMetric(
        new CustomMetric(perplexity, name = "perplexity"))
      fitParams.setKVStore("device")
      fitParams.setOptimizer(
        new SGD(learningRate = 0.01f, momentum = 0f, wd = 0.00001f))
      fitParams.setInitializer(new Xavier(factorType = "in", magnitude = 2.34f))
      fitParams.setBatchEndCallback(new Speedometer(batchSize, 50))

      logger.info("Start training ...")
      model.fit(
        trainData = dataTrain,
        evalData = Some(dataVal),
        numEpoch = inst.numEpoch, fitParams)
      logger.info("Finished training...")
    } catch {
      case ex: Exception =>
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
    }
  }
}
