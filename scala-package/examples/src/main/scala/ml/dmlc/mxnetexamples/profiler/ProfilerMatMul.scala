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

package ml.dmlc.mxnetexamples.profiler

import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Profiler
import java.io.File
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Random

/**
 * @author Depeng Liang
 */
object ProfilerMatMul {
  private val logger = LoggerFactory.getLogger(classOf[ProfilerMatMul])

  def main(args: Array[String]): Unit = {
    val erul = new ProfilerMatMul
    val parser: CmdLineParser = new CmdLineParser(erul)
    try {
      parser.parseArgument(args.toList.asJava)
      val ctx = if (erul.gpu >= 0) Context.gpu(erul.gpu) else Context.cpu()

      val path = s"${erul.outputPath}${File.separator}${erul.profilerName}"
      Profiler.profilerSetConfig(mode = erul.profilerMode, fileName = path)
      logger.info(s"profile file save to $path")

      val A = Symbol.Variable("A")
      val B = Symbol.Variable("B")
      val C = Symbol.dot()(A, B)()

      val executor = C.simpleBind(ctx, "write",
          Map("A" -> Shape(4096, 4096), "B" -> Shape(4096, 4096)))

      val a = Random.uniform(-1.0f, 1.0f, shape = Shape(4096, 4096))
      val b = Random.uniform(-1.0f, 1.0f, shape = Shape(4096, 4096))

      a.copyTo(executor.argDict("A"))
      b.copyTo(executor.argDict("B"))

      val flag = false
      logger.info(s"execution begin")
      var t0 = 0L
      var t1 = 0L
      for (i <- 0 until erul.iterNum) {
        if (i == erul.beginProfilingIter) {
          t0 = System.currentTimeMillis()
          Profiler.profilerSetState("run")
        }
        if (i == erul.endProfilingIter) {
          t1 = System.currentTimeMillis()
          Profiler.profilerSetState("stop")
        }
        executor.forward()
        executor.outputs(0).waitToRead()
      }
      logger.info(s"execution end")
      val duration = t1 - t0
      logger.info(s"duration: ${duration / 1000f}s")
      logger.info(s"${duration.toFloat / erul.iterNum}ms/operator")
    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class ProfilerMatMul {
  @Option(name = "--profiler-mode", usage = "the profiler mode, can be \"symbolic\" or \"all\".")
  private val profilerMode: String = "symbolic"
  @Option(name = "--output-path", usage = "the profile file output directory.")
  private val outputPath: String = "."
  @Option(name = "--profile-filename", usage = "the profile file name.")
  private val profilerName: String = "profile_matmul_20iter.json"
  @Option(name = "--iter-num", usage = "iterate number.")
  private val iterNum: Int = 100
  @Option(name = "--begin-profiling-iter'", usage = "specific iterate to start the profiler.")
  private val beginProfilingIter: Int = 50
  @Option(name = "--end-profiling-iter'", usage = "specific iterate to stop the profiler.")
  private val endProfilingIter: Int = 70
  @Option(name = "--gpu", usage = "which gpu card to use, default is -1, means using cpu")
  private val gpu: Int = -1
}
