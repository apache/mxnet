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

package org.apache.mxnetexamples.multitask

import java.io.File
import java.net.URL

import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import org.apache.commons.io.FileUtils
import org.apache.mxnet.{Context, DataBatch, DataDesc, DataIter, EvalMetric, Executor, NDArray, NDArrayCollector, Shape, Symbol, Xavier}
import org.apache.mxnet.DType.DType
import org.apache.mxnet.optimizer.RMSProp
import org.apache.mxnetexamples.Util

import scala.collection.immutable.ListMap
import scala.sys.process.Process

/**
 * Example of multi-task
 */
object ExampleMultiTask {
  private val logger = LoggerFactory.getLogger(classOf[ExampleMultiTask])

  def buildNetwork(): Symbol = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.api.FullyConnected(data = Some(data), num_hidden = 128)
    val act1 = Symbol.api.Activation(data = Some(fc1), act_type = "relu")
    val fc2 = Symbol.api.FullyConnected(data = Some(act1), num_hidden = 64)
    val act2 = Symbol.api.Activation(data = Some(fc2), act_type = "relu")
    val fc3 = Symbol.api.FullyConnected(data = Some(act2), num_hidden = 10)
    val sm1 = Symbol.api.SoftmaxOutput(data = Some(fc3))
    val sm2 = Symbol.api.SoftmaxOutput(data = Some(fc3))

    val softmax = Symbol.Group(sm1, sm2)

    softmax
  }

  // multi label mnist iterator
  class MultiMnistIterator(dataIter: DataIter) extends DataIter {

    @throws(classOf[NoSuchElementException])
    override def next(): DataBatch = {
      if (hasNext) {
        val batch = this.dataIter.next()
        val label = batch.label(0)
        new DataBatch(batch.data,
          IndexedSeq(label, label),
          batch.index,
          batch.pad, null, null, null)
      } else {
        throw new NoSuchElementException
      }
    }

    /**
     * reset the iterator
     */
    override def reset(): Unit = this.dataIter.reset()

    override def batchSize: Int = dataIter.batchSize

    /**
     * get data of current batch
     * @return the data of current batch
     */
    override def getData(): IndexedSeq[NDArray] = this.dataIter.getData()

    /**
     * Get label of current batch
     * @return the label of current batch
     */
    override def getLabel(): IndexedSeq[NDArray] = {
      val label = this.dataIter.getLabel()(0)
      IndexedSeq(label, label)
    }

    /**
     * the index of current batch
     * @return
     */
    override def getIndex(): IndexedSeq[Long] = this.dataIter.getIndex()

    // The name and shape of label provided by this iterator
    @deprecated
    override def provideLabel: ListMap[String, Shape] = {
      val provideLabel = this.dataIter.provideLabel.toArray
      // Different labels should be used here for actual application
      ListMap("softmax1_label" -> provideLabel(0)._2,
              "softmax2_label" -> provideLabel(0)._2)
    }

    // The name and shape of label provided by this iterator
    override def provideLabelDesc: IndexedSeq[DataDesc] = {
      val head = this.dataIter.provideLabelDesc(0)
      // Different labels should be used here for actual application
      IndexedSeq(
        new DataDesc("softmax1_label", head.shape, head.dtype, head.layout),
        new DataDesc("softmax2_label", head.shape, head.dtype, head.layout)
      )
    }

    /**
     * get the number of padding examples
     * in current batch
     * @return number of padding examples in current batch
     */
    override def getPad(): Int = this.dataIter.getPad()

    // The name and shape of data provided by this iterator
    @deprecated
    override def provideData: ListMap[String, Shape] = this.dataIter.provideData

    override def provideDataDesc: IndexedSeq[DataDesc] = this.dataIter.provideDataDesc

    override def hasNext: Boolean = this.dataIter.hasNext
  }

  class MultiAccuracy(num: Int, name: String) {
    require(num >= 1)

    private var sumMetric: Array[Float] = new Array[Float](num)
    private var numInst: Array[Int] = new Array[Int](num)

    def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = {
      require(labels.length == preds.length,
        "labels and predictions should have the same length.")
      assert(labels.length == num)

      for (i <- labels.indices) {
        val (pred, label) = (preds(i), labels(i))
        val predLabel = NDArray.api.argmax_channel(data = pred)
        require(label.shape == predLabel.shape,
          s"label ${label.shape} and prediction ${predLabel.shape}" +
          s"should have the same length.")
        for ((labelElem, predElem) <- label.toArray zip predLabel.toArray) {
          if (labelElem == predElem) {
            this.sumMetric(i) += 1
          }
        }
        this.numInst(i) += predLabel.shape(0)
        predLabel.dispose()
      }
    }

    def get(): Array[(String, Float)] = {
      (0 until num).map( i => (this.name, this.sumMetric(i) / this.numInst(i))).toArray
    }

    def reset(): Unit = {
      this.numInst = this.numInst.map(x => 0)
      this.sumMetric = this.numInst.map(x => 0f)
    }

  }

  class Speedometer(val batchSize: Int, val frequent: Int = 50) {
    private val logger = LoggerFactory.getLogger(classOf[Speedometer])
    private var init = false
    private var tic: Long = 0L
    private var lastCount: Int = 0

    def invoke(epoch: Int, count: Int, evalMetric: MultiAccuracy): Unit = {
      if (lastCount > count) {
        init = false
      }
      lastCount = count

      if (init) {
        if (count % frequent == 0) {
          val speed = frequent.toDouble * batchSize / (System.currentTimeMillis - tic) * 1000
          if (evalMetric != null) {
            val nameVals = evalMetric.get
            nameVals.foreach { case (name, value) =>
              logger.info("Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-%s=%f".format(
                  epoch, count, speed, name, value))
            }
          } else {
            logger.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec".format(epoch, count, speed))
          }
          tic = System.currentTimeMillis
        }
      } else {
        init = true
        tic = System.currentTimeMillis
      }
    }
  }

  def getTrainingData: String = {
    val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci"
    val tempDirPath = System.getProperty("java.io.tmpdir")
    val modelDirPath = tempDirPath + File.separator + "multitask/"
    Util.downloadUrl(baseUrl + "/mnist/mnist.zip",
      tempDirPath + "/multitask/mnist.zip")

    // TODO: Need to confirm with Windows
    Process("unzip " + tempDirPath + "/multitask/mnist.zip -d "
      + tempDirPath + "/multitask/") !

    modelDirPath
  }

  def train(batchSize: Int, numEpoch: Int, ctx: Context, modelDirPath: String):
  (Executor, MultiAccuracy) = {
    NDArrayCollector.auto().withScope {
      val lr = 0.001f
      val network = ExampleMultiTask.buildNetwork()
      val (trainIter, valIter) =
        Data.mnistIterator(modelDirPath, batchSize = batchSize, inputShape = Shape(784))
      val trainMultiIt = new MultiMnistIterator(trainIter)
      val valMultiIter = new MultiMnistIterator(valIter)

      val datasAndLabels = trainMultiIt.provideData ++ trainMultiIt.provideLabel

      val (argShapes, outputShapes, auxShapes)
      = network.inferShape(trainMultiIt.provideData("data"))
      val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

      val argNames = network.listArguments
      val argDict = argNames.zip(argShapes.map(NDArray.empty(_, ctx))).toMap

      val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
        !datasAndLabels.contains(name)
      }.map(x => x._1 -> NDArray.empty(x._2, ctx)).toMap

      argDict.foreach { case (name, ndArray) =>
        if (!datasAndLabels.contains(name)) {
          initializer.initWeight(name, ndArray)
        }
      }

      val data = argDict("data")
      val label1 = argDict("softmaxoutput0_label")
      val label2 = argDict("softmaxoutput1_label")
      val maxGradNorm = 0.5f
      val executor = network.bind(ctx, argDict, gradDict)

      val opt = new RMSProp(learningRate = lr, wd = 0.00001f)

      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }

      val evalMetric = new ExampleMultiTask.MultiAccuracy(num = 2, name = "multi_accuracy")
      val batchEndCallback = new ExampleMultiTask.Speedometer(batchSize, 50)

      for (epoch <- 0 until numEpoch) {
        // Training phase
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false
        // Iterate over training data.
        trainMultiIt.reset()

        while (!epochDone) {
          var doReset = true
          while (doReset && trainMultiIt.hasNext) {
            val dataBatch = trainMultiIt.next()

            data.set(dataBatch.data(0))
            label1.set(dataBatch.label(0))
            label2.set(dataBatch.label(1))

            executor.forward(isTrain = true)
            executor.backward()

            val norm = Math.sqrt(paramsGrads.map { case (idx, name, grad, optimState) =>
              val l2Norm = NDArray.api.norm(data = (grad / batchSize)).toScalar
              l2Norm * l2Norm
            }.sum).toFloat

            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              if (norm > maxGradNorm) {
                grad.set(grad.toArray.map(_ * (maxGradNorm / norm)))
                opt.update(idx, argDict(name), grad, optimState)
              } else opt.update(idx, argDict(name), grad, optimState)
            }

            // evaluate at end, so out_cpu_array can lazy copy
            evalMetric.update(dataBatch.label, executor.outputs)

            nBatch += 1
            batchEndCallback.invoke(epoch, nBatch, evalMetric)
          }
          if (doReset) {
            trainMultiIt.reset()
          }
          // this epoch is done
          epochDone = true
        }
        var nameVals = evalMetric.get
        nameVals.foreach { case (name, value) =>
          logger.info(s"Epoch[$epoch] Train-$name=$value")
        }
        val toc = System.currentTimeMillis
        logger.info(s"Epoch[$epoch] Time cost=${toc - tic}")

        evalMetric.reset()
        valMultiIter.reset()
        while (valMultiIter.hasNext) {
          val evalBatch = valMultiIter.next()

          data.set(evalBatch.data(0))
          label1.set(evalBatch.label(0))
          label2.set(evalBatch.label(1))

          executor.forward(isTrain = true)

          evalMetric.update(evalBatch.label, executor.outputs)
          evalBatch.dispose()
        }

        nameVals = evalMetric.get
        nameVals.foreach { case (name, value) =>
          logger.info(s"Epoch[$epoch] Validation-$name=$value")
        }
      }

      (executor, evalMetric)
    }
  }

  def main(args: Array[String]): Unit = {
    val lesk = new ExampleMultiTask
    val parser: CmdLineParser = new CmdLineParser(lesk)
    try {
      parser.parseArgument(args.toList.asJava)

      val batchSize = 100
      val numEpoch = 5
      val ctx = if (lesk.gpu != -1) Context.gpu(lesk.gpu) else Context.cpu()

      val modelPath = if (lesk.dataPath == null) lesk.dataPath else getTrainingData

      val (executor, evalMetric) = train(batchSize, numEpoch, ctx, modelPath)
      executor.dispose()

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class ExampleMultiTask {
  @Option(name = "--data-path", usage = "the mnist data path")
  private val dataPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = -1
}
