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
import org.scalatest.FunSuite
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
class MultiTaskSuite extends FunSuite {

  test("Multitask Test") {
    val logger = LoggerFactory.getLogger(classOf[MultiTaskSuite])
    logger.info("Multitask Test...")

    val batchSize = 100
    val numEpoch = 10
    val ctx = Context.cpu()

    val modelPath = ExampleMultiTask.getTrainingData
    val (executor, evalMetric) = ExampleMultiTask.train(batchSize, numEpoch, ctx, modelPath)
    evalMetric.get.foreach { case (name, value) =>
      assert(value >= 0.95f)
    }
    executor.dispose()
  }

}
