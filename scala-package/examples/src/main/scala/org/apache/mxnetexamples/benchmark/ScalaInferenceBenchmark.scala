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

package org.apache.mxnetexamples.benchmark

import org.apache.mxnetexamples.InferBase
import org.apache.mxnetexamples.infer.imageclassifier.ImageClassifierExample
import org.apache.mxnet._
import org.apache.mxnetexamples.infer.objectdetector.SSDClassifierExample
import org.apache.mxnetexamples.rnn.TestCharRnn
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

object ScalaInferenceBenchmark {

  private val logger = LoggerFactory.getLogger(classOf[CLIParserBase])

  def loadModel(objectToRun: InferBase, context: Array[Context], batchInference : Boolean):
  Any = {
    objectToRun.loadModel(context, batchInference)
  }

  def loadDataSet(objectToRun: InferBase):
  Any = {
    objectToRun.loadSingleData()
  }

  def loadBatchDataSet(objectToRun: InferBase, batchSize: Int):
  List[Any] = {
    objectToRun.loadBatchFileList(batchSize)
  }

  def runInference(objectToRun: InferBase, loadedModel: Any, dataSet: Any, totalRuns: Int):
  List[Long] = {
    var inferenceTimes: List[Long] = List()
    for (i <- 1 to totalRuns) {
      ResourceScope.using() {
        val startTimeSingle = System.currentTimeMillis()
        objectToRun.runSingleInference(loadedModel, dataSet)
        val estimatedTimeSingle = System.currentTimeMillis() - startTimeSingle
        inferenceTimes = estimatedTimeSingle :: inferenceTimes
        logger.info("Inference time at iteration: %d is : %d \n".format(i, estimatedTimeSingle))
      }
    }

    inferenceTimes
  }

  def runBatchInference(objecToRun: InferBase, loadedModel: Any, dataSetBatches: List[Any]):
  List[Long] = {

    var inferenceTimes: List[Long] = List()
    for (batch <- dataSetBatches) {
      ResourceScope.using() {
        val loadedBatch = objecToRun.loadInputBatch(batch)
        val startTimeSingle = System.currentTimeMillis()
        objecToRun.runBatchInference(loadedModel, loadedBatch)
        val estimatedTimeSingle = System.currentTimeMillis() - startTimeSingle
        inferenceTimes = estimatedTimeSingle :: inferenceTimes
        logger.info("Batch Inference time is : %d \n".format(estimatedTimeSingle))
      }
    }

    inferenceTimes
  }

  def percentile(p: Int, seq: Seq[Long]): Long = {
    val sorted = seq.sorted
    val k = math.ceil((seq.length - 1) * (p / 100.0)).toInt
    sorted(k)
  }

  def printStatistics(inferenceTimes: List[Long], metricsPrefix: String)  {

    val times: Seq[Long] = inferenceTimes
    val p50 = percentile(50, times)
    val p99 = percentile(99, times)
    val p90 = percentile(90, times)
    val average = times.sum / (times.length * 1.0)

    logger.info("\n%s_p99 %d, %s_p90 %d, %s_p50 %d, %s_average %1.2f".format(metricsPrefix,
      p99, metricsPrefix, p90, metricsPrefix, p50, metricsPrefix, average))

  }

  def main(args: Array[String]): Unit = {

    var context = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      context = Context.gpu()
    }
    var baseCLI : CLIParserBase = null
    try {
      val exampleName = args(1)
      val exampleToBenchmark : InferBase = exampleName match {
        case "ImageClassifierExample" => {
          val imParser = new org.apache.mxnetexamples.infer.imageclassifier.CLIParser
          baseCLI = imParser
          val parsedVals = new CmdLineParser(imParser).parseArgument(args.toList.asJava)
          new ImageClassifierExample(imParser)
        }
        case "ObjectDetectionExample" => {
          val imParser = new org.apache.mxnetexamples.infer.objectdetector.CLIParser
          baseCLI = imParser
          val parsedVals = new CmdLineParser(imParser).parseArgument(args.toList.asJava)
          new SSDClassifierExample(imParser)
        }
        case "CharRnn" => {
          val imParser = new org.apache.mxnetexamples.rnn.CLIParser
          baseCLI = imParser
          val parsedVals = new CmdLineParser(imParser).parseArgument(args.toList.asJava)
          new TestCharRnn(imParser)
        }
        case _ => throw new Exception("Invalid example name to run")
      }

      logger.info("Running single inference call")
      // Benchmarking single inference call
      ResourceScope.using() {
        val loadedModel = loadModel(exampleToBenchmark, context, false)
        val dataSet = loadDataSet(exampleToBenchmark)
        val inferenceTimes = runInference(exampleToBenchmark, loadedModel, dataSet, baseCLI.count)
        printStatistics(inferenceTimes, "single_inference")
      }

      if (baseCLI.batchSize != 0) {
        logger.info("Running for batch inference call")
        // Benchmarking batch inference call
        ResourceScope.using() {
          val loadedModel = loadModel(exampleToBenchmark, context, true)
          val batchDataSet = loadBatchDataSet(exampleToBenchmark, baseCLI.batchSize)
          val inferenceTimes = runBatchInference(exampleToBenchmark, loadedModel, batchDataSet)
          printStatistics(inferenceTimes, "batch_inference")
        }
      }

    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        new CmdLineParser(baseCLI).printUsage(System.err)
        sys.exit(1)
      }
    }
  }

}

class CLIParserBase {
  @Option(name = "--example", usage = "The scala example to benchmark")
  val exampleName: String = "ImageClassifierExample"
  @Option(name = "--count", usage = "number of times to run inference")
  val count: Int = 1000
  @Option(name = "--batchSize", usage = "BatchSize to run batchinference calls", required = false)
  val batchSize: Int = 0
}
