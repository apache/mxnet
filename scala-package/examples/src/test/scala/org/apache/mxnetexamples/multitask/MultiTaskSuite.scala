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

import org.apache.mxnet._
import org.slf4j.LoggerFactory
import org.apache.mxnet.Context

import org.scalatest.FunSuite


/**
  * Integration test for Multi-task example.
  */
class MultiTaskSuite extends FunSuite {
  test("Multitask Test") {
    val logger = LoggerFactory.getLogger(classOf[MultiTaskSuite])
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      logger.info("Multitask Test...")

      ResourceScope.using() {
        val batchSize = 100
        val numEpoch = 3
        val ctx = Context.gpu()

        val modelPath = ExampleMultiTask.getTrainingData
        val (executor, evalMetric) = ExampleMultiTask.train(batchSize, numEpoch, ctx, modelPath)
        evalMetric.get.foreach { case (name, value) =>
          assert(value >= 0.95f)
        }
        executor.dispose()
      }
    } else {
      logger.info("GPU test only, skipped...")
    }
  }
}
