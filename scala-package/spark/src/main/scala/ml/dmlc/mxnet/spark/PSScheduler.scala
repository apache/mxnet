package ml.dmlc.mxnet.spark

import java.io.IOException
import java.util.concurrent.atomic.AtomicReference

import ml.dmlc.mxnet.KVStoreServer
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable
import scala.collection.JavaConverters._

/**
 * Start parameter scheduler on spark driver
 * @author Yizhi Liu
 */
object PSScheduler {
  private val logger: Logger = LoggerFactory.getLogger(classOf[PSScheduler])

  def main(args: Array[String]): Unit = {
    println("Start scheduler in object PSScheduler")
    val cmdLine = new CommandLine
    val parser: CmdLineParser = new CmdLineParser(cmdLine)
    try {
      parser.parseArgument(args.toList.asJava)
      cmdLine.checkArguments()
      val envs: mutable.Map[String, String] = mutable.HashMap.empty[String, String]
      envs.put("DMLC_ROLE", cmdLine.role)
      envs.put("DMLC_PS_ROOT_URI", cmdLine.rootUri)
      envs.put("DMLC_PS_ROOT_PORT", cmdLine.rootPort.toString)
      envs.put("DMLC_NUM_SERVER", cmdLine.numServer.toString)
      envs.put("DMLC_NUM_WORKER", cmdLine.numWorker.toString)
      KVStoreServer.init(envs.toMap)
      KVStoreServer.start()
    } catch {
      case e: Throwable =>
        logger.error(e.getMessage, e)
        sys.exit(-1)
    }
  }
}

class PSScheduler(private val classpath: String,
                  private val rootUri: String,
                  private val rootPort: Int,
                  private val numServer: Int = 1,
                  private val numWorker: Int = 1,
                  private val java: String = "java",
                  private val jvmOpts: String = "") {
  private val trackerProcess: AtomicReference[Process] = new AtomicReference[Process]
  private val role = "scheduler"
  private val proj = runningClass(classOf[PSScheduler].getName)

  def startProcess(): Boolean = {
    val cmd = s"$java $jvmOpts -cp $classpath $proj " +
      s"--role=$role --root-uri=$rootUri --root-port=$rootPort " +
      s"--num-server=$numServer --num-worker=$numWorker"
    PSScheduler.logger.info(s"Start process: $cmd")
    try {
      trackerProcess.set(Runtime.getRuntime.exec(cmd))
      true
    }
    catch {
      case ioe: IOException =>
        ioe.printStackTrace()
        false
    }
  }

  def stop() {
    if (trackerProcess.get != null) {
      trackerProcess.get.destroy()
    }
  }

  private def runningClass(cls: String): String = {
    // trick to remove the last '$'
    cls.replace("$", "")
  }
}

class CommandLine {
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

  def checkArguments(): Unit = {
    require(role != null, "Undefined role")
    require(rootUri != null, "Undefined root uri")
    require(rootPort > 0, s"Invalid root port $rootPort")
    require(numServer > 0, s"Invalid number of servers: $numServer")
    require(numWorker > 0, s"Invalid number of worders: $numWorker")
  }
}
