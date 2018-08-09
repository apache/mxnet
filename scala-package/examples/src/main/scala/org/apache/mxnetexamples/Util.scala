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

package org.apache.mxnetexamples

import java.io.File
import java.net.URL

import org.apache.commons.io.FileUtils
import org.slf4j.LoggerFactory

object Util {
  /**
    * a Download wrapper with retry scheme on Scala
    * @param url the URL for the file
    * @param filePath the path to store the file
    * @param maxRetry maximum retries will take
    */
  def downloadUrl(url: String, filePath: String, maxRetry: Option[Int] = None) : Unit = {
    val tmpFile = new File(filePath)
    var retry = maxRetry.getOrElse(3)
    var success = false
    if (!tmpFile.exists()) {
      while (retry > 0 && !success) {
        try {
          FileUtils.copyURLToFile(new URL(url), tmpFile)
          success = true
        } catch {
          case e: Exception => retry -= 1
        }
      }
    } else {
      success = true
    }
   if (!success) throw new Exception(s"$url Download failed!")
  }

  /**
    * This Util is designed to manage the tests in CI
    * @param name the name of the test
    * @return runTest and number of epoch
    */
  def testManager(name: String) : (Boolean, Int) = {
    val GPUTest = Map[String, Int]("CNN" -> 10, "GAN" -> 5, "MultiTask" -> 3,
      "NSBoost" -> 10, "NSNeural" -> 80)
    val CPUTest = Set("CustomOp", "MNIST", "Infer", "Profiler")
    val GPU_Enable = System.getenv().containsKey("SCALA_TEST_INTEGRATION")
    if (GPUTest.contains(name)) {
      if (GPU_Enable) {
        val epoch = if (System.getenv("SCALA_TEST_INTEGRATION").toInt == 1) {
          1
        } else GPUTest.get(name).get
        (true, epoch)
      } else {
        (false, 0)
      }
    } else if (CPUTest.contains(name)) {
      (true, 0)
    } else {
      throw new IllegalArgumentException("Test not found, please registered in Util!")
    }
  }
}
