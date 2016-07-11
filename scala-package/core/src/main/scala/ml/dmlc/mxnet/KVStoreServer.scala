package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.{Logger, LoggerFactory}

/**
 * Server node for the key value store
 * @author Yizhi Liu
 */
class KVStoreServer(private val kvStore: KVStore) {
  private val logger: Logger = LoggerFactory.getLogger(classOf[KVStoreServer])
  private val handle: KVStoreHandle = kvStore.handle
  private val controller = new KVServerControllerCallback {
    override def invoke(cmdId: Int, cmdBody: String): Unit = {
      logger.debug("Receive cmdId {}, cmdBody: {}", cmdId, cmdBody)
      if (cmdId == 0) {
        val optimizer = Serializer.getSerializer.deserialize[Optimizer](
          Serializer.decodeBase64String(cmdBody))
        kvStore.setOptimizer(optimizer)
      } else {
        logger.warn(s"Server ${kvStore.rank}, unknown command ($cmdId, $cmdBody)")
      }
    }
  }

  // run the server, whose behavior is like
  // while receive(x):
  //   if is_command x: controller(x)
  //   else if is_key_value x: updater(x)
  def run(): Unit = {
    checkCall(_LIB.mxKVStoreRunServer(handle, controller))
  }
}

object KVStoreServer {
  private val logger: Logger = LoggerFactory.getLogger(classOf[KVStoreServer])
  /**
   * Start server/scheduler according to env variables
   * @param dieIfOthersGoOutTimeout When this argument is set to an integer greater than 0
   *                                (in second),
   *                                a daemon thread will start to periodically check
   *                                whether scheduler (server side) or servers (scheduler side)
   *                                are dead. If so, die itself.
   *                                This could be useful for running mxnet on distributed
   *                                data platform,
   *                                where you do not know which node your application runs on
   *                                and in such situation
   *                                you want others die automatically once
   *                                some of the nodes goes out.
   */
  def start(dieIfOthersGoOutTimeout: Int = 0): Unit = {
    val isWorker = new RefInt
    checkCall(_LIB.mxKVStoreIsWorkerNode(isWorker))
    require(isWorker.value == 0, "cannot start kv-store server on worker node")
    val kvStore = KVStore.create("dist")
    val daemonThread: Option[Thread] =
      if (dieIfOthersGoOutTimeout > 0) {
        val daemon = new Runnable {
          override def run(): Unit = {
            var running = true
            while (running) {
              try {
                Thread.sleep(dieIfOthersGoOutTimeout.toLong * 1000)
                val numDead = kvStore.numDeadNode(KVStore.GROUP_NODE_SCHEDULER
                  + KVStore.GROUP_NODE_SERVER + KVStore.GROUP_NODE_WORKER)
                if (numDead > 0) {
                  logger.error(s"Detect $numDead dead node(s). Shutdown now.")
                  System.exit(1)
                }
              } catch {
                case e: InterruptedException => running = false
              }
            }
          }
        }
        val t = new Thread(daemon)
        t.setDaemon(true)
        t.start()
        Option(t)
      } else {
        None
      }
    val server = new KVStoreServer(kvStore)
    server.run()
    daemonThread.foreach(t => {
      t.interrupt()
      t.join()
    })
    kvStore.dispose()
  }

  def init(env: Map[String, String]): Unit = {
    val keys = env.keys.toArray
    val vals = env.values.toArray
    checkCall(_LIB.mxInitPSEnv(keys, vals))
  }
}

trait KVServerControllerCallback {
  def invoke(cmdId: Int, cmdBody: String): Unit
}
