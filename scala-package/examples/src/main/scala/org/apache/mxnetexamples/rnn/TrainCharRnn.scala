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
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import org.apache.mxnet.optimizer.Adam

/**
  * Follows the demo, to train the char rnn:
  * https://github.com/apache/mxnet/blob/v1.x/example/rnn/char-rnn.ipynb
  */
object TrainCharRnn {

  private val logger = LoggerFactory.getLogger(classOf[TrainCharRnn])

  def runTrainCharRnn(dataPath: String, saveModelPath: String,
                      ctx : Context, numEpoch : Int): Unit = {
    ResourceScope.using() {
      // The batch size for training
      val batchSize = 32
      // We can support various length input
      // For this problem, we cut each input sentence to length of 129
      // So we only need fix length bucket
      val buckets = Array(129)
      // hidden unit in LSTM cell
      val numHidden = 512
      // embedding dimension, which is, map a char to a 256 dim vector
      val numEmbed = 256
      // number of lstm layer
      val numLstmLayer = 3
      // we will show a quick demo in 2 epoch
      // learning rate
      val learningRate = 0.001f
      // we will use pure sgd without momentum
      val momentum = 0.0f

      val vocab = Utils.buildVocab(dataPath)

      // generate symbol for a length
      def symGen(seqLen: Int): Symbol = {
        Lstm.lstmUnroll(numLstmLayer, seqLen, vocab.size + 1,
          numHidden = numHidden, numEmbed = numEmbed,
          numLabel = vocab.size + 1, dropout = 0.2f)
      }

      // initalize states for LSTM
      val initC = for (l <- 0 until numLstmLayer)
        yield (s"l${l}_init_c_beta", (batchSize, numHidden))
      val initH = for (l <- 0 until numLstmLayer)
        yield (s"l${l}_init_h_beta", (batchSize, numHidden))
      val initStates = initC ++ initH

      val dataTrain = new BucketIo.BucketSentenceIter(dataPath, vocab, buckets,
        batchSize, initStates, seperateChar = "\n",
        text2Id = Utils.text2Id, readContent = Utils.readContent)

      // the network symbol
      val symbol = symGen(buckets(0))

      val datasAndLabels = dataTrain.provideDataDesc ++ dataTrain.provideLabelDesc
      val (argShapes, outputShapes, auxShapes) = symbol.inferShape(datasAndLabels)

      val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

      val argNames = symbol.listArguments()
      val argDict = argNames.zip(argShapes.map(NDArray.zeros(_, ctx))).toMap
      val auxNames = symbol.listAuxiliaryStates()
      val auxDict = auxNames.zip(auxShapes.map(NDArray.zeros(_, ctx))).toMap

      val datasAndLabelsNames = datasAndLabels.map(_.name)
      val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
        !datasAndLabelsNames.contains(name)
      }.map(x => x._1 -> NDArray.empty(x._2, ctx)).toMap

      argDict.foreach { case (name, ndArray) =>
        if (!datasAndLabelsNames.contains(name)) {
          initializer.initWeight(name, ndArray)
        }
      }

      val data = argDict("data")
      val label = argDict("softmax_label")

      val executor = symbol.bind(ctx, argDict, gradDict)

      val opt = new Adam(learningRate = learningRate, wd = 0.0001f)

      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }

      val evalMetric = new CustomMetric(Utils.perplexity, "perplexity")
      val batchEndCallback = new Callback.Speedometer(batchSize, 50)
      val epochEndCallback = Utils.doCheckpoint(s"${saveModelPath}/obama")

      for (epoch <- 0 until numEpoch) {
        // Training phase
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false
        // Iterate over training data.
        dataTrain.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && dataTrain.hasNext) {
            val dataBatch = dataTrain.next()

            data.set(dataBatch.data(0))
            label.set(dataBatch.label(0))
            executor.forward(isTrain = true)
            executor.backward()
            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
            }

            // evaluate at end, so out_cpu_array can lazy copy
            evalMetric.update(dataBatch.label, executor.outputs)

            nBatch += 1
            batchEndCallback.invoke(epoch, nBatch, evalMetric)
          }
          if (doReset) {
            dataTrain.reset()
          }
          // this epoch is done
          epochDone = true
        }
        val (name, value) = evalMetric.get
        name.zip(value).foreach { case (n, v) =>
          logger.info(s"Epoch[$epoch] Train-$n=$v")
        }
        val toc = System.currentTimeMillis
        logger.info(s"Epoch[$epoch] Time cost=${toc - tic}")

        epochEndCallback.invoke(epoch, symbol, argDict, auxDict)
      }
      executor.dispose()
    }
  }

  def main(args: Array[String]): Unit = {
    val incr = new TrainCharRnn
    val parser: CmdLineParser = new CmdLineParser(incr)
    try {
      parser.parseArgument(args.toList.asJava)
      val ctx = if (incr.gpu == -1) Context.cpu() else Context.gpu(incr.gpu)
      assert(incr.dataPath != null && incr.saveModelPath != null)
      runTrainCharRnn(incr.dataPath, incr.saveModelPath, ctx, 75)
    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class TrainCharRnn {
  @Option(name = "--data-path", usage = "the input train data file")
  private val dataPath: String = "./data/obama.txt"
  @Option(name = "--save-model-path", usage = "the model saving path")
  private val saveModelPath: String = "./model/"
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = -1
}
