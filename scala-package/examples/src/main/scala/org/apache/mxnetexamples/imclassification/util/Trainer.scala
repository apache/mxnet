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

package org.apache.mxnetexamples.imclassification.util

import org.apache.mxnet.Callback.Speedometer
import org.apache.mxnet._
import org.apache.mxnet.optimizer.SGD
import org.slf4j.LoggerFactory

object Trainer {
  private val logger = LoggerFactory.getLogger(classOf[Trainer])

  /**
    * Fits a model
    * @param batchSize Number of images per training batch
    * @param numExamples Total number of image examples
    * @param devs List of device contexts to use
    * @param network The model to train
    * @param dataLoader Function to get data loaders for training and validation data
    * @param kvStore KVStore to use
    * @param numEpochs Number of times to train on each image
    * @param modelPrefix Prefix to model identification
    * @param loadEpoch Loads a saved checkpoint at this epoch when set
    * @param lr The learning rate
    * @param lrFactor Learning rate factor (see FactorScheduler)
    * @param lrFactorEpoch Learning rate factor epoch (see FactorScheduler)
    * @param clipGradient Maximum gradient during optimization
    * @param monitorSize (See Monitor)
    * @return Final accuracy
    */
  // scalastyle:off parameterNum
  def fit(batchSize: Int, numExamples: Int, devs: Array[Context],
          network: Symbol, dataLoader: (Int, KVStore) => (DataIter, DataIter),
          kvStore: String, numEpochs: Int, modelPrefix: String = null, loadEpoch: Int = -1,
          lr: Float = 0.1f, lrFactor: Float = 1f, lrFactorEpoch: Float = 1f,
          clipGradient: Float = 0f, monitorSize: Int = -1): Accuracy = {
    // kvstore
    ResourceScope.using()  {
      var kv = KVStore.create(kvStore)

      // load model
      val modelPrefixWithRank =
        if (modelPrefix == null) null
        else modelPrefix + s"-${kv.rank}"

      val (argParams, auxParams, beginEpoch) =
        if (loadEpoch >= 0) {
          require(modelPrefixWithRank != null)
          val tmp = FeedForward.load(modelPrefix, loadEpoch)
          (tmp.getArgParams, tmp.getAuxParams, loadEpoch)
        } else {
          (null, null, 0)
        }

      // save model
      val checkpoint: EpochEndCallback =
        if (modelPrefix == null) null
        else new EpochEndCallback {
          override def invoke(epoch: Int, symbol: Symbol,
                              argParams: Map[String, NDArray],
                              auxStates: Map[String, NDArray]): Unit = {
            Model.saveCheckpoint(modelPrefix, epoch + 1, symbol, argParams, auxParams)
          }
        }

      // data
      val (train, validation) = dataLoader(batchSize, kv)

      // train
      val epochSize =
        if (kvStore == "dist_sync") numExamples / batchSize / kv.numWorkers
        else numExamples / batchSize

      val lrScheduler =
        if (lrFactor < 1f) {
          new FactorScheduler(step = Math.max((epochSize * lrFactorEpoch).toInt, 1),
                              factor = lrFactor)
        } else {
          null
        }
      val optimizer: Optimizer = new SGD(learningRate = lr,
        lrScheduler = lrScheduler, clipGradient = clipGradient,
        momentum = 0.9f, wd = 0.00001f)

      // disable kvstore for single device
      if (kv.`type`.contains("local") && (devs.length == 1 || devs(0).deviceType != "gpu")) {
        kv.dispose()
        kv = null
      }

      val model = new FeedForward(ctx = devs,
                                  symbol = network,
                                  numEpoch = numEpochs,
                                  optimizer = optimizer,
                                  initializer = new Xavier(factorType = "in", magnitude = 2.34f),
                                  argParams = argParams,
                                  auxParams = auxParams,
                                  beginEpoch = beginEpoch,
                                  epochSize = epochSize)
      if (monitorSize > 0) {
        model.setMonitor(new Monitor(monitorSize))
      }
      val acc = new Accuracy()
      model.fit(trainData = train,
                evalData = validation,
                evalMetric = acc,
                kvStore = kv,
                batchEndCallback = new Speedometer(batchSize, 50),
                epochEndCallback = checkpoint)
      if (kv != null) {
        kv.dispose()
      }
      acc
    }
  }
  // scalastyle:on parameterNum
}

class Trainer

