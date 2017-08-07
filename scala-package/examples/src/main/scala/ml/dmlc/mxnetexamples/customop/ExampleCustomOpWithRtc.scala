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

package ml.dmlc.mxnetexamples.customop

import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.DType.DType
import ml.dmlc.mxnet.DataIter
import ml.dmlc.mxnet.DataBatch
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.EvalMetric
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Xavier
import ml.dmlc.mxnet.optimizer.RMSProp
import ml.dmlc.mxnet.CustomOp
import ml.dmlc.mxnet.CustomOpProp
import ml.dmlc.mxnet.Operator
import ml.dmlc.mxnet.optimizer.SGD
import ml.dmlc.mxnet.Accuracy
import ml.dmlc.mxnet.Callback.Speedometer
import ml.dmlc.mxnet.Rtc

/**
 * Example of CustomOp with Rtc
 * @author Depeng Liang
 */
object ExampleCustomOpWithRtc {
  private val logger = LoggerFactory.getLogger(classOf[ExampleCustomOpWithRtc])

  class Softmax(_param: Map[String, String]) extends CustomOp {

    private var fwdKernel: Rtc = null
    private var bwdKernel: Rtc = null

    override def forward(sTrain: Boolean, req: Array[String],
      inData: Array[NDArray], outData: Array[NDArray], aux: Array[NDArray]): Unit = {
      if (fwdKernel == null) {
        fwdKernel = new Rtc("softmax", Array(("x", inData(0))), Array(("y", outData(0))), """
          int i = threadIdx.x + blockIdx.x*blockDim.x;
          float max_x = x[i*x_dims[1]];
          for (int j = 1; j < x_dims[1]; ++j) {
              if (max_x < x[i*x_dims[1]+j]) {
                  max_x = x[i*x_dims[1]+j];
              }
          }
          float sum = 0.0f;
          for (int j = 0; j < x_dims[1]; ++j) {
              sum += expf(x[i*x_dims[1]+j]-max_x);
          }
          for (int j = 0; j < x_dims[1]; ++j) {
              y[i*x_dims[1]+j] = expf(x[i*x_dims[1]+j]-max_x)/sum;
          }
          """)
      }
      val blockDim = {
        val dimStr = this._param("forwardBlockDim")
        val dims = dimStr.substring(1, dimStr.length() - 1).split(",").map(_.toInt)
        (dims(0), dims(1), dims(2))
      }
      fwdKernel.push(Array(inData(0)), Array(outData(0)), blockDim, (inData(0).shape(0), 1, 1))
    }

    override def backward(req: Array[String], outGrad: Array[NDArray],
      inData: Array[NDArray], outData: Array[NDArray],
      inGrad: Array[NDArray], aux: Array[NDArray]): Unit = {
      val l = inData(1)
      val y = outData(0)
      val dx = inGrad(0)
      if (bwdKernel == null) {
        bwdKernel = new Rtc("softmax_grad", Array(("y", y), ("l", l)), Array(("dx", dx)), """
          int i = blockIdx.x;
          int j = threadIdx.x;
          int k = static_cast<int>(l[i]);
          if (j == k) {
              dx[i*dx_dims[1]+j] = y[i*dx_dims[1]+j] - 1.0f;
          } else {
              dx[i*dx_dims[1]+j] = y[i*dx_dims[1]+j];
          }
          """)
      }
      bwdKernel.push(Array(y, l), Array(dx), (y.shape(0), 1, 1), (y.shape(1), 1, 1))
    }
  }

  class SoftmaxProp(needTopGrad: Boolean = false)
    extends CustomOpProp(needTopGrad) {

    override def listArguments(): Array[String] = Array("data", "label")

    override def listOutputs(): Array[String] = Array("output")

    override def inferShape(inShape: Array[Shape]):
      (Array[Shape], Array[Shape], Array[Shape]) = {
      val dataShape = inShape(0)
      val labelShape = Shape(dataShape(0))
      val outputShape = dataShape
      (Array(dataShape, labelShape), Array(outputShape), null)
    }

    override def inferType(inType: Array[DType]):
      (Array[DType], Array[DType], Array[DType]) = {
      (inType, inType.take(1), null)
    }

    override def createOperator(ctx: String, inShapes: Array[Array[Int]],
      inDtypes: Array[Int]): CustomOp = new Softmax(this.kwargs)
  }

  Operator.register("softmax", new SoftmaxProp)

  def main(args: Array[String]): Unit = {
    val leop = new ExampleCustomOpWithRtc
    val parser: CmdLineParser = new CmdLineParser(leop)
    try {
      parser.parseArgument(args.toList.asJava)
      assert(leop.dataPath != null && leop.gpu >= 0)

      val ctx = Context.gpu(0)

      val dataName = Array("data")
      val labelName = Array("softmax_label")

      val data = Symbol.Variable("data")
      val label = Symbol.Variable("label")
      val fc1 = Symbol.FullyConnected("fc1")()(Map("data" -> data, "num_hidden" -> 128))
      val act1 = Symbol.Activation("relu1")()(Map("data" -> fc1, "act_type" -> "relu"))
      val fc2 = Symbol.FullyConnected("fc2")()(Map("data" -> act1, "num_hidden" -> 64))
      val act2 = Symbol.Activation("relu2")()(Map("data" -> fc2, "act_type" -> "relu"))
      val fc3 = Symbol.FullyConnected("fc3")()(Map("data" -> act2, "num_hidden" -> 10))
      val mlp = Symbol.Custom("softmax")()(Map("data" -> fc3,
        "label" -> label, "op_type" -> "softmax", "forwardBlockDim" -> "(1,1,1)"))

      val (trainIter, testIter) =
        Data.mnistIterator(leop.dataPath, batchSize = 100, inputShape = Shape(784))
      val datasAndLabels = trainIter.provideData ++ trainIter.provideLabel
      val (argShapes, outputShapes, auxShapes) = mlp.inferShape(datasAndLabels)

      val initializer = new Xavier(factorType = "in", magnitude = 2.34f)

      val argNames = mlp.listArguments()
      val argDict = argNames.zip(argShapes.map(s => NDArray.empty(s, ctx))).toMap
      val gradDict = argNames.zip(argShapes).filter { case (name, shape) =>
        !datasAndLabels.contains(name)
      }.map(x => x._1 -> NDArray.empty(x._2, ctx) ).toMap
      argDict.foreach { case (name, ndArray) =>
        if (!datasAndLabels.contains(name)) {
          initializer.initWeight(name, ndArray)
        }
      }

      val executor = mlp.bind(ctx, argDict, gradDict)
      val lr = 0.001f
      val opt = new RMSProp(learningRate = lr, wd = 0.00001f)
      val paramsGrads = gradDict.toList.zipWithIndex.map { case ((name, grad), idx) =>
        (idx, name, grad, opt.createState(idx, argDict(name)))
      }
      val evalMetric = new Accuracy
      val batchEndCallback = new Speedometer(100, 100)
      val numEpoch = 20

      for (epoch <- 0 until numEpoch) {
        val tic = System.currentTimeMillis
        evalMetric.reset()
        var nBatch = 0
        var epochDone = false

        trainIter.reset()
        while (!epochDone) {
          var doReset = true
          while (doReset && trainIter.hasNext) {
            val dataBatch = trainIter.next()
            argDict("data").set(dataBatch.data(0))
            argDict("label").set(dataBatch.label(0))
            executor.forward(isTrain = true)
            executor.backward()
            paramsGrads.foreach { case (idx, name, grad, optimState) =>
              opt.update(idx, argDict(name), grad, optimState)
            }
            evalMetric.update(dataBatch.label, executor.outputs)
            nBatch += 1
            batchEndCallback.invoke(epoch, nBatch, evalMetric)
          }
          if (doReset) {
            trainIter.reset()
          }
          epochDone = true
        }
        val (name, value) = evalMetric.get
        name.zip(value).foreach { case (n, v) =>
          logger.info(s"Epoch[$epoch] Train-accuracy=$v")
        }
        val toc = System.currentTimeMillis
        logger.info(s"Epoch[$epoch] Time cost=${toc - tic}")

        evalMetric.reset()
        testIter.reset()
        while (testIter.hasNext) {
          val evalBatch = testIter.next()
          argDict("data").set(evalBatch.data(0))
          argDict("label").set(evalBatch.label(0))
          executor.forward(isTrain = true)
          evalMetric.update(evalBatch.label, executor.outputs)
          evalBatch.dispose()
        }
        val (names, values) = evalMetric.get
        names.zip(values).foreach { case (n, v) =>
          logger.info(s"Epoch[$epoch] Validation-accuracy=$v")
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

class ExampleCustomOpWithRtc {
  @Option(name = "--data-path", usage = "the mnist data path")
  private val dataPath: String = null
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = 0
}
