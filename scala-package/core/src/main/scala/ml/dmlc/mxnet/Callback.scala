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

package ml.dmlc.mxnet

import org.slf4j.{Logger, LoggerFactory}

/**
 * Callback functions that can be used to track various status during epoch.
 */
object Callback {

  class Speedometer(batchSize: Int, frequent: Int = 50) extends BatchEndCallback {
    private val logger: Logger = LoggerFactory.getLogger(classOf[Speedometer])
    private var init = false
    private var tic: Long = 0L
    private var lastCount: Int = 0

    override def invoke(epoch: Int, count: Int, evalMetric: EvalMetric): Unit = {
      if (lastCount > count) {
        init = false
      }
      lastCount = count

      if (init) {
        if (count % frequent == 0) {
          val speed = frequent.toDouble * batchSize / (System.currentTimeMillis - tic) * 1000
          if (evalMetric != null) {
            val (name, value) = evalMetric.get
            name.zip(value).foreach { case (n, v) =>
              logger.info("Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-%s=%f".format(
              epoch, count, speed, n, v))
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
}
