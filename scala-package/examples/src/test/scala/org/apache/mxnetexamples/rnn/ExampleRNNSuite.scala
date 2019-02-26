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


import org.apache.mxnet.{Context, NDArrayCollector}
import org.apache.mxnetexamples.Util
import org.scalatest.{BeforeAndAfterAll, FunSuite, Ignore}
import org.slf4j.LoggerFactory

import scala.sys.process.Process

class ExampleRNNSuite extends FunSuite with BeforeAndAfterAll {
  private val logger = LoggerFactory.getLogger(classOf[ExampleRNNSuite])

  override def beforeAll(): Unit = {
    logger.info("Downloading LSTM model")
    val tempDirPath = System.getProperty("java.io.tmpdir")
    logger.info("tempDirPath: %s".format(tempDirPath))
    val baseUrl = "https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/RNN/"
    Util.downloadUrl(baseUrl + "obama.zip", tempDirPath + "/RNN/obama.zip")
    Util.downloadUrl(baseUrl + "sherlockholmes.train.txt",
      tempDirPath + "/RNN/sherlockholmes.train.txt")
    Util.downloadUrl(baseUrl + "sherlockholmes.valid.txt",
      tempDirPath + "/RNN/sherlockholmes.valid.txt")
    // TODO: Need to confirm with Windows
    Process(s"unzip $tempDirPath/RNN/obama.zip -d $tempDirPath/RNN/") !
  }

  test("Example CI: Test LSTM Bucketing") {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    var ctx = Context.cpu()
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
      System.getenv("SCALA_TEST_ON_GPU").toInt == 1) {
      ctx = Context.gpu()
    }
    if (!System.getenv().containsKey("CI")) {
      LstmBucketing.runTraining(tempDirPath + "/RNN/sherlockholmes.train.txt",
                                tempDirPath + "/RNN/sherlockholmes.valid.txt", Array(ctx), 1)
    } else {
      logger.info("Skipping test on CI...")
    }
  }

  test("Example CI: Test TrainCharRNN") {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    if (System.getenv().containsKey("SCALA_TEST_ON_GPU") &&
          System.getenv("SCALA_TEST_ON_GPU").toInt == 1 &&
          !System.getenv().containsKey("CI")) {
      val ctx = Context.gpu()
      TrainCharRnn.runTrainCharRnn(tempDirPath + "/RNN/obama.txt",
        tempDirPath, ctx, 1)
    } else {
      logger.info("CPU not supported for this test, skipped...")
    }
  }

  test("Example CI: Test Inference on CharRNN") {
    val tempDirPath = System.getProperty("java.io.tmpdir")
    val ctx = Context.gpu()
    TestCharRnn.runInferenceCharRNN(tempDirPath + "/RNN/obama.txt",
      tempDirPath + "/RNN/obama", "The joke")
  }
}
