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

import org.apache.commons.io.FileUtils
import org.apache.mxnet.Context
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory
import org.apache.mxnet.Symbol
import org.apache.mxnet.DataIter
import org.apache.mxnet.DataBatch
import org.apache.mxnet.NDArray
import org.apache.mxnet.Shape
import org.apache.mxnet.EvalMetric
import org.apache.mxnet.Context
import org.apache.mxnet.Xavier
import org.apache.mxnet.optimizer.RMSProp

import java.io.File
import java.net.URL

import scala.sys.process.Process
import scala.collection.immutable.ListMap
import scala.collection.immutable.IndexedSeq

import scala.collection.mutable.{ArrayBuffer, ListBuffer}


/**
  * Integration test for imageClassifier example.
  * This will run as a part of "make scalatest"
  */
class MultiTaskTest extends FunSuite with BeforeAndAfterAll {

  def getListOfFiles(dir: File): List[File] = dir.listFiles.filter(_.isFile).toList

  private val logger = LoggerFactory.getLogger(classOf[MultiTaskTest])

  logger.info("Multitask Test...")

  val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci"
  val tempDirPath = System.getProperty("java.io.tmpdir")
  val modelDirPath = tempDirPath + File.separator + "multitask/"
  val tmpFile = new File(tempDirPath + "/multitask/mnist.zip")
  if (!tmpFile.exists()) {
    FileUtils.copyURLToFile(new URL(baseUrl + "/mnist/mnist.zip"),
      tmpFile)
  }

  // TODO: Need to confirm with Windows


  Process("unzip " + tempDirPath + "/multitask/mnist.zip -d "
    + tempDirPath + "/multitask/") !

  val batchSize = 100
  val numEpoch = 10
  val ctx = Context.cpu()
  val lr = 0.001f
  val network = ExampleMultiTask.buildNetwork()
  val (trainIter, valIter) =
    Data.mnistIterator(modelDirPath, batchSize = batchSize, inputShape = Shape(784))
  val trainMultiIter = new ExampleMultiTask.MultiMnistIterator(trainIter)
  val valMultiIter = new ExampleMultiTask.MultiMnistIterator(valIter)

  val datasAndLabels = trainMultiIter.provideData ++ trainMultiIter.provideLabel

  val (argShapes, outputShapes, auxShapes) = network.inferShape(trainMultiIter.provideData("data"))
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
  evalMetric.get.foreach {case (name, value) =>
    assert(value >= 0.95f)
  }
  executor.dispose()

}
