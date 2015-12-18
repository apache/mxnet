package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

/**
 * Key value store interface of MXNet for parameter synchronization.
 * @author Yizhi Liu
 */
object KVStore {
  /**
   * Create a new KVStore.
   *
   * @param name : {'local', 'dist'}
   *     The type of KVStore
   *     - local works for multiple devices on a single machine (single process)
   *     - dist works for multi-machines (multiple processes)
   * @return The created KVStore
   */
  def create(name: String = "local"): KVStore = {
    val handle = new KVStoreHandle
    checkCall(_LIB.mxKVStoreCreate(name, handle))
    new KVStore(handle)
  }
}

class KVStore(private val handle: KVStoreHandle) {
  private var updaterFunc: MXKVStoreUpdater = null

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
    val valuePtrs = values.map(_.handle.value)
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
    val valuePtrs = values.map(_.handle.value)
    checkCall(_LIB.mxKVStorePush(handle, keys.length, keys, valuePtrs, priority))
  }

  def push(keys: Array[Int], values: Array[NDArray]): Unit = push(keys, values, 0)

  def push(key: Int, value: NDArray, priority: Int = 0): Unit = {
    push(Array(key), Array(value), priority)
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
    val outPtrs = outs.map(_.handle.value)
    checkCall(_LIB.mxKVStorePull(handle, keys.length, keys, outPtrs, priority))
  }

  def pull(keys: Array[Int], outs: Array[NDArray]): Unit = pull(keys, outs, 0)

  def pull(key: Int, out: NDArray, priority: Int = 0): Unit = {
    pull(Array(key), Array(out), priority)
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
    checkCall(_LIB.mxKVStoreSetUpdater(handle, updaterFunc, null))
  }
}
