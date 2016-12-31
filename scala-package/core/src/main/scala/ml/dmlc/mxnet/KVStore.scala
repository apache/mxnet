package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._
import org.slf4j.{LoggerFactory, Logger}

/**
 * Key value store interface of MXNet for parameter synchronization.
 * @author Yizhi Liu
 */
object KVStore {

  // group id of scheduler/server/worker
  val GROUP_NODE_SCHEDULER = 1
  val GROUP_NODE_SERVER = 2
  val GROUP_NODE_WORKER = 4

  /**
   * Create a new KVStore. <br />
   * <b>
   * WARNING: it is your responsibility to clear this object through dispose().
   * NEVER rely on the GC strategy
   * </b>
   *
   * @param name : {'local', 'dist'}
   *     The type of KVStore
   *     - local works for multiple devices on a single machine (single process)
   *     - dist works for multi-machines (multiple processes)
   * @return The created KVStore
   */
  def create(name: String = "local"): KVStore = {
    val handle = new KVStoreHandleRef
    checkCall(_LIB.mxKVStoreCreate(name, handle))
    new KVStore(handle.value)
  }
}

// scalastyle:off finalize
class KVStore(private[mxnet] val handle: KVStoreHandle) {
  private val logger: Logger = LoggerFactory.getLogger(classOf[KVStore])
  private var updaterFunc: MXKVStoreUpdater = null
  private var disposed = false

  override protected def finalize(): Unit = {
    dispose()
  }

  /**
   * Release the native memory.
   * The object shall never be used after it is disposed.
   */
  def dispose(): Unit = {
    if (!disposed) {
      _LIB.mxKVStoreFree(handle)
      disposed = true
    }
  }

  /**
   * Initialize a single or a sequence of key-value pairs into the store.
   * For each key, one must init it before push and pull.
   * Only worker 0's (rank == 0) data are used.
   * This function returns after data have been initialized successfully
   *
   * @param keys The keys.
   * @param values The values.
   */
  def init(keys: Array[Int], values: Array[NDArray]): Unit = {
    require(keys.length == values.length, "len(keys) != len(values)")
    val valuePtrs = values.map(_.handle)
    checkCall(_LIB.mxKVStoreInit(handle, keys.length, keys, valuePtrs))
  }

  def init(key: Int, value: NDArray): Unit = {
    init(Array(key), Array(value))
  }

  /**
   * Push a single or a sequence of key-value pairs into the store.
   * Data consistency:
   * 1. this function returns after adding an operator to the engine.
   * 2. push is always called after all previous push and pull on the same key are finished
   * 3. there is no synchronization between workers. One can use _barrier() to sync all workers
   *
   * @param keys Keys
   * @param values  According values
   * @param priority
   *         The priority of the push operation.
   *         The higher the priority, the faster this action is likely
   *         to be executed before other push actions.
   */
  def push(keys: Array[Int], values: Array[NDArray], priority: Int): Unit = {
    require(keys.length == values.length, "len(keys) != len(values)")
    val valuePtrs = values.map(_.handle)
    checkCall(_LIB.mxKVStorePush(handle, keys.length, keys, valuePtrs, priority))
  }

  def push(keys: Array[Int], values: Array[NDArray]): Unit = push(keys, values, 0)

  def push(key: Int, value: NDArray, priority: Int = 0): Unit = {
    push(Array(key), Array(value), priority)
  }

  def push(key: Int, values: Array[NDArray], priority: Int): Unit = {
    val keys = Array.fill(values.length)(key)
    push(keys, values, priority)
  }

  def push(key: Int, values: Array[NDArray]): Unit = {
    push(key, values, 0)
  }

  /**
   * Pull a single value or a sequence of values from the store.
   *
   * Data consistency:
   * 1. this function returns after adding an operator to the engine. But any
   *    further read on out will be blocked until it is finished.
   * 2. pull is always called after all previous push and pull on the same key are finished
   * 3. It pulls the newest value from the store.
   * @param keys Keys
   * @param outs According values
   * @param priority
   *     The priority of the push operation.
   *     The higher the priority, the faster this action is likely
   *     to be executed before other push actions.
   */
  def pull(keys: Array[Int], outs: Array[NDArray], priority: Int): Unit = {
    require(keys.length == outs.length, "len(keys) != len(outs)")
    val outPtrs = outs.map(_.handle)
    checkCall(_LIB.mxKVStorePull(handle, keys.length, keys, outPtrs, priority))
  }

  def pull(keys: Array[Int], outs: Array[NDArray]): Unit = pull(keys, outs, 0)

  def pull(key: Int, out: NDArray, priority: Int = 0): Unit = {
    pull(Array(key), Array(out), priority)
  }

  def pull(key: Int, outs: Array[NDArray], priority: Int): Unit = {
    val keys = Array.fill(outs.length)(key)
    pull(keys, outs, priority)
  }

  def pull(key: Int, outs: Array[NDArray]): Unit = {
    pull(key, outs, 0)
  }

  // Get the type of this kvstore
  def `type`: String = {
    val kvType = new RefString
    checkCall(_LIB.mxKVStoreGetType(handle, kvType))
    kvType.value
  }

  /**
   * Get the number of worker nodes
   * @return The number of worker nodes
   */
  def numWorkers: Int = {
    val size = new RefInt
    checkCall(_LIB.mxKVStoreGetGroupSize(handle, size))
    size.value
  }

  /**
   * Get the rank of this worker node
   * @return The rank of this node, which is in [0, get_num_workers())
   */
  def rank: Int = {
    val rank = new RefInt
    checkCall(_LIB.mxKVStoreGetRank(handle, rank))
    rank.value
  }

  /**
   * Register an optimizer to the store
   * If there are multiple machines, this process (should be a worker node)
   * will pack this optimizer and send it to all servers. It returns after
   * this action is done.
   *
   * @param optimizer the optimizer
   */
  def setOptimizer(optimizer: Optimizer): Unit = {
    val isWorker = new RefInt
    checkCall(_LIB.mxKVStoreIsWorkerNode(isWorker))
    if (`type`.contains("dist") && isWorker.value != 0) {
      val optSerialized = Serializer.getSerializer.serialize(optimizer)
      val cmd = Serializer.encodeBase64String(optSerialized)
      logger.debug("Send optimizer to server: {}", cmd)
      sendCommandToServers(0, cmd)
    } else {
      setUpdater(Optimizer.getUpdater(optimizer))
    }
  }

  /**
   * Set a push updater into the store.
   *
   * This function only changes the local store. Use setOptimizer for
   * multi-machines.
   *
   * @param updater  the updater function
   */
  def setUpdater(updater: MXKVStoreUpdater): Unit = {
    this.updaterFunc = updater
    checkCall(_LIB.mxKVStoreSetUpdater(handle, updaterFunc))
  }

  /**
   * Global barrier among all worker nodes
   *
   * For example, assume there are n machines, we want to let machine 0 first
   * init the values, and then pull the inited value to all machines. Before
   * pulling, we can place a barrier to guarantee that the initialization is
   * finished.
   */
  def barrier(): Unit = {
    checkCall(_LIB.mxKVStoreBarrier(handle))
  }

  def numDeadNode(nodeId: Int): Int = {
    val number = new RefInt
    checkCall(_LIB.mxKVStoreGetNumDeadNode(handle, nodeId, number))
    number.value
  }

  /**
   * Whether to do barrier when the kvstore finalizes
   * @param barrierBeforeExit
   */
  def setBarrierBeforeExit(barrierBeforeExit: Boolean): Unit = {
    val flag: Int = if (barrierBeforeExit) 1 else 0
    checkCall(_LIB.mxKVStoreSetBarrierBeforeExit(handle, flag))
  }

  /**
   * Send a command to all server nodes
   *
   * Send a command to all server nodes, which will make each server node run
   * KVStoreServer.controller
   *
   * This function returns after the command has been executed in all server nodes
   *
   * @param head the head of the command
   * @param body the body of the command
   */
  private def sendCommandToServers(head: Int, body: String): Unit = {
    checkCall(_LIB.mxKVStoreSendCommmandToServers(handle, head, body))
  }
}
// scalastyle:off finalize
