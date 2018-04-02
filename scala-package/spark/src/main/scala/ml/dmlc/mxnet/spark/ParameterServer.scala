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

package ml.dmlc.mxnet.spark

import java.io.{IOException, InputStream, OutputStream}
import java.util.concurrent.atomic.AtomicReference

import ml.dmlc.mxnet.KVStoreServer
import org.kohsuke.args4j.{Option, CmdLineParser}
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable
import scala.collection.JavaConverters._

/**
 * Start ps scheduler/server in a new process
 */
private[mxnet] object ParameterServer {
  private val logger: Logger = LoggerFactory.getLogger(classOf[ParameterServer])
  def main(args: Array[String]): Unit = {
    val cmdLine = new CommandLine
    val parser: CmdLineParser = new CmdLineParser(cmdLine)
    try {
      parser.parseArgument(args.toList.asJava)
      cmdLine.checkArguments()
      KVStoreServer.init(buildEnv(
        cmdLine.role, cmdLine.rootUri, cmdLine.rootPort,
        cmdLine.numServer, cmdLine.numWorker))
      KVStoreServer.start(dieIfOthersGoOutTimeout = cmdLine.timeout)
    } catch {
      case e: Throwable =>
        logger.error(e.getMessage, e)
        sys.exit(-1)
    }
  }

  def buildEnv(role: String, rootUri: String, rootPort: Int,
               numServer: Int, numWorker: Int): Map[String, String] = {
    val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
    envs.put("DMLC_ROLE", role)
    envs.put("DMLC_PS_ROOT_URI", rootUri)
    envs.put("DMLC_PS_ROOT_PORT", rootPort.toString)
    envs.put("DMLC_NUM_SERVER", numServer.toString)
    envs.put("DMLC_NUM_WORKER", numWorker.toString)
    envs.toMap
  }

  private class CommandLine {
    @Option(name = "--role", usage = "PS role")
    val role: String = null
    @Option(name = "--root-uri", usage = "PS scheduler address")
    val rootUri: String = null
    @Option(name = "--root-port", usage = "PS scheduler port")
    val rootPort: Int = -1
    @Option(name = "--num-server", usage = "PS server number")
    val numServer: Int = 1
    @Option(name = "--num-worker", usage = "PS worker number")
    val numWorker: Int = 1
    @Option(name = "--timeout", usage = "PS go out timeout")
    val timeout: Int = 0

    def checkArguments(): Unit = {
      require(role != null, "Undefined role")
      require(rootUri != null, "Undefined root uri")
      require(rootPort > 0, s"Invalid root port $rootPort")
      require(numServer > 0, s"Invalid number of servers: $numServer")
      require(numWorker > 0, s"Invalid number of workers: $numWorker")
    }
  }
}

class ParameterServer(
    classpath: String,
    role: String,
    rootUri: String,
    rootPort: Int,
    numServer: Int = 1,
    numWorker: Int = 1,
    timeout: Int = 0,
    java: String = "java",
    jvmOpts: String = "") {

  private val logger: Logger = LoggerFactory.getLogger(classOf[ParameterServer])
  private val psProcess: AtomicReference[Process] = new AtomicReference[Process]

  /**
   * A utility class to redirect the child process's stdout or stderr.
   */
  private class RedirectThread(
      in: InputStream,
      out: OutputStream,
      name: String,
      propagateEof: Boolean = false)
    extends Thread(name) {

    setDaemon(true)
    override def run() {
      val buf = new Array[Byte](1024)
      var len = in.read(buf)
      while (len != -1) {
        out.write(buf, 0, len)
        out.flush()
        len = in.read(buf)
      }
      if (propagateEof) {
        out.close()
      }
    }
  }

  private def startLoggingThreads(rootUri: String, rootPort: Int): Unit = {
    val inputStream = psProcess.get().getInputStream
    val errorStream = psProcess.get().getErrorStream
    logger.info(s"Starting InputStream-Redirecter Thread for $rootUri:$rootPort")
    new RedirectThread(inputStream, System.out, "InputStream-Redirecter", true).start()
    logger.info(s"Starting ErrorStream-Redirecter Thread for $rootUri:$rootPort")
    new RedirectThread(errorStream, System.err, "ErrorStream-Redirecter", true).start()
  }

  def startProcess(): Int = {
    val cp = if (classpath == null) "" else s"-cp $classpath"
    val cmd = s"$java $jvmOpts $cp $runningClass " +
      s"--role=$role --root-uri=$rootUri --root-port=$rootPort " +
      s"--num-server=$numServer --num-worker=$numWorker --timeout=$timeout"
    try {
      val childProcess = Runtime.getRuntime.exec(cmd)
      logger.info(s"Started process: $cmd at $rootUri:$rootPort")
      psProcess.set(childProcess)
      startLoggingThreads(rootUri, rootPort)
      psProcess.get().waitFor()
    } catch {
      case ioe: IOException =>
        ioe.printStackTrace()
        1
    } finally {
      stop()
    }
  }

  def stop() {
    if (psProcess.get != null && psProcess.get().isAlive) {
      psProcess.get.destroy()
    }
  }

  private def runningClass: String = {
    // trick to remove the last '$'
    classOf[ParameterServer].getName.replace("$", "")
  }
}
