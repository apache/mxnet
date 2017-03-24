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

import ml.dmlc.mxnet.Base._

object Profiler {

  val mode2Int = Map("symbolic" -> 0, "all" -> 1)
  val state2Int = Map("stop" -> 0, "run" -> 1)

  /**
   * Set up the configure of profiler.
   * @param mode, optional
   *                  Indicting whether to enable the profiler, can
   *                  be "symbolic" or "all". Default is "symbolic".
   * @param fileName, optional
   *                  The name of output trace file. Default is "profile.json".
   */
  def profilerSetConfig(mode: String = "symbolic", fileName: String = "profile.json"): Unit = {
    require(mode2Int.contains(mode))
    checkCall(_LIB.mxSetProfilerConfig(mode2Int(mode), fileName))
  }

  /**
   * Set up the profiler state to record operator.
   * @param state, optional
   *                  Indicting whether to run the profiler, can
   *                  be "stop" or "run". Default is "stop".
   */
  def profilerSetState(state: String = "stop"): Unit = {
    require(state2Int.contains(state))
    checkCall(_LIB.mxSetProfilerState(state2Int(state)))
  }

  /**
   * Dump profile and stop profiler. Use this to save profile
   * in advance in case your program cannot exit normally.
   */
  def dumpProfile(): Unit = {
    checkCall(_LIB.mxDumpProfile())
  }
}
