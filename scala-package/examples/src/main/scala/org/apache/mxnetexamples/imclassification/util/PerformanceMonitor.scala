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

import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}
import java.lang.management.ManagementFactory
import java.util.Date

class PerformanceMonitor(filename: String) extends Runnable {

  val bean = ManagementFactory.getOperatingSystemMXBean
    .asInstanceOf[com.sun.management.OperatingSystemMXBean]
  val runtime = Runtime.getRuntime

  val outputfile = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)))

  val csvSchema = Array("time", "processCpuLoad", "systemCpuLoad", "usedMemory", "freeMemory",
    "totalMemory", "maxMemory", "commitedVirtualMemory")
  outputfile.write(csvSchema.mkString(",") + "\n")


  /**
    * Cleans up after thread has stopped monitoring
    */
  def finish(): Unit = {
    outputfile.close()
  }

  /**
    * Runs a periodic measurement recording lines of performance and writing (buffered) to an output CSV file
    */
  override def run(): Unit = {
    val time = new Date().getTime
    val processCpuLoad = bean.getProcessCpuLoad
    val systemCpuLoad = bean.getSystemCpuLoad
    val usedMemory = runtime.totalMemory - runtime.freeMemory
    val freeMemory = runtime.freeMemory
    val totalMemory = runtime.totalMemory
    val maxMemory = runtime.maxMemory
    val commitedVirtualMemory = bean.getCommittedVirtualMemorySize

    val row = Array(time, processCpuLoad, systemCpuLoad, usedMemory, freeMemory, totalMemory,
      maxMemory, commitedVirtualMemory)
    outputfile.write(row.mkString(",") + "\n")
  }

}
