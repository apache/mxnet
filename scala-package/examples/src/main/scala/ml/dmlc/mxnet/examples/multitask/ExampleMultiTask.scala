package ml.dmlc.mxnet.examples.multitask

import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.DataIter
import ml.dmlc.mxnet.DataBatch
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.EvalMetric
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Xavier
import ml.dmlc.mxnet.optimizer.RMSProp

/**
 * Example of multi-task
 * @author Depeng Liang
 */
object ExampleMultiTask {
  private val logger = LoggerFactory.getLogger(classOf[ExampleMultiTask])

  def buildNetwork(): Symbol = {
    val data = Symbol.Variable("data")
    val fc1 = Symbol.FullyConnected("fc1")()(Map("data" -> data, "num_hidden" -> 128))
    val act1 = Symbol.Activation("relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
    val fc2 = Symbol.FullyConnected("fc2")()(Map("data" -> act1, "num_hidden" -> 64))
    val act2 = Symbol.Activation("relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
    val fc3 = Symbol.FullyConnected("fc3")()(Map("data" -> act2, "num_hidden" -> 10))
    val sm1 = Symbol.SoftmaxOutput("softmax1")()(Map("data" -> fc3))
    val sm2 = Symbol.SoftmaxOutput("softmax2")()(Map("data" -> fc3))

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
                                     batch.pad)
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
    override def provideLabel: Map[String, Shape] = {
      val provideLabel = this.dataIter.provideLabel.toArray
      // Different labels should be used here for actual application
      Map("softmax1_label" -> provideLabel(0)._2,
          "softmax2_label" -> provideLabel(0)._2)
    }

    /**
     * get the number of padding examples
     * in current batch
     * @return number of padding examples in current batch
     */
    override def getPad(): Int = this.dataIter.getPad()

    // The name and shape of data provided by this iterator
    override def provideData: Map[String, Shape] = this.dataIter.provideData

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
        val predLabel = NDArray.argmaxChannel(pred)
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

  def main(args: Array[String]): Unit = {
    val lesk = new ExampleMultiTask
    val parser: CmdLineParser = new CmdLineParser(lesk)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(lesk.dataPath != null)

      val batchSize = 100
      val numEpoch = 100
      val ctx = if (lesk.gpu != -1) Context.gpu(lesk.gpu) else Context.cpu()
      val lr = 0.001f
      val network = buildNetwork()
      val (trainIter, valIter) =
        Data.mnistIterator(lesk.dataPath, batchSize = batchSize, inputShape = Shape(784))
      val trainMultiIter = new MultiMnistIterator(trainIter)
      val valMultiIter = new MultiMnistIterator(valIter)

      val datasAndLabels = trainMultiIter.provideData ++ trainMultiIter.provideLabel
      val (argShapes, outputShapes, auxShapes) = network.inferShape(datasAndLabels)

      val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

      val argNames = network.listArguments()
      val argDict = argNames.zip(argShapes.map(NDArray.empty(_, ctx))).toMap
      val auxNames = network.listAuxiliaryStates()
      val auxDict = auxNames.zip(auxShapes.map(NDArray.empty(_, ctx))).toMap

      val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
        !datasAndLabels.contains(name)
      }.map(x => x._1 -> NDArray.empty(x._2, ctx) ).toMap

      argDict.foreach { case (name, ndArray) =>
        if (!datasAndLabels.contains(name)) {
          initializer.initWeight(name, ndArray)
        }
      }

      val data = argDict("data")
      val label1 = argDict("softmax1_label")
      val label2 = argDict("softmax2_label")

      val maxGradNorm = 0.5f
      val executor = network.bind(ctx, argDict, gradDict)

      val opt = new RMSProp(learningRate = lr, wd = 0.00001f)

      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }

      val evalMetric = new MultiAccuracy(num = 2, name = "multi_accuracy")
      val batchEndCallback = new Speedometer(batchSize, 50)

      for (epoch <- 0 until numEpoch) {
        // Training phase
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false
        // Iterate over training data.
        trainMultiIter.reset()

        while (!epochDone) {
          var doReset = true
          while (doReset && trainMultiIter.hasNext) {
            val dataBatch = trainMultiIter.next()

            data.set(dataBatch.data(0))
            label1.set(dataBatch.label(0))
            label2.set(dataBatch.label(1))

            executor.forward(isTrain = true)
            executor.backward()

            val norm = Math.sqrt(paramsGrads.map { case (idx, name, grad, optimState) =>
              val l2Norm = NDArray.norm(grad / batchSize).toScalar
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
            trainMultiIter.reset()
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
