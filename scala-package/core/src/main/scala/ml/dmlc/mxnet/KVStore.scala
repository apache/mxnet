package ml.dmlc.mxnet

import ml.dmlc.mxnet.Base._

/**
 * Key value store interface of MXNet for parameter synchronization.
 *
 * @author Yizhi Liu
 */
object KVStore {
  def main(args: Array[String]): Unit = {
    val kv = KVStore.create()
    println(kv.handle.value)

    println("Setting updater")
    val updater = new MXKVStoreUpdater {
      override def update(key: Int, input: NDArray, stored: NDArray, handle: AnyRef): Unit = {
        println(s"update on key: $key")
        stored += input * 2
      }
    }
    kv._setUpdater(updater)

    val shape = Array(2, 1)
    val a = NDArray.zeros(shape)

    kv.init(Array(3), Array(NDArray.zeros(shape)+4))
    kv.pull(Array(3), Array(a))
    println(a.toArray.mkString(","))

    kv.push(Array(3), Array(NDArray.zeros(shape)+1))
    kv.pull(Array(3), Array(a))
    println(a.toArray.mkString(","))
  }
  /**
    Create a new KVStore.

    Parameters
    ----------
    name : {'local'}
        The type of KVStore
        - local works for multiple devices on a single machine (single process)
        - dist works for multi-machines (multiple processes)
    Returns
    -------
    kv : KVStore
        The created KVStore
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
    Initialize a single or a sequence of key-value pairs into the store.

        For each key, one must init it before push and pull.

        Only worker 0's (rank == 0) data are used.

        This function returns after data have been initialized successfully

        Parameters
        ----------
        key : int or sequence of int
            The keys.
        value : NDArray or sequence of NDArray
            The values.

        Examples
        --------
        >>> # init a single key-value pair
        >>> shape = (2,3)
        >>> kv = mx.kv.create('local')
        >>> kv.init(3, mx.nd.ones(shape)*2)
        >>> a = mx.nd.zeros(shape)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # init a list of key-value pairs
        >>> keys = [5, 7, 9]
        >>> kv.init(keys, [mx.nd.ones(shape)]*len(keys))
  */
  def init(keys: Array[Int], values: Array[NDArray]): Unit = {
    require(keys.length == values.length, "len(keys) != len(values)")
    val valuePtrs = values.map(_.handle.value)
    checkCall(_LIB.mxKVStoreInit(handle, keys.length, keys, valuePtrs))
  }

  /**
    Push a single or a sequence of key-value pairs into the store.

        Data consistency:

        1. this function returns after adding an operator to the engine.

        2. push is always called after all previous push and pull on the same
        key are finished

        3. there is no synchronization between workers. One can use _barrier()
        to sync all workers

        Parameters
        ----------
        key : int or list of int
            Keys

        value : NDArray or list of NDArray or list of list of NDArray
            According values

        priority : int, optional
            The priority of the push operation.
            The higher the priority, the faster this action is likely
            to be executed before other push actions.

        Examples
        --------
        >>> # push a single key-value pair
        >>> kv.push(3, mx.nd.ones(shape)*8)
        >>> kv.pull(3, out=a) # pull out the value
        >>> print a.asnumpy()
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]

        >>> # aggregate the value and the push
        >>> gpus = [mx.gpu(i) for i in range(4)]
        >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
        >>> kv.push(3, b)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]

        >>> # push a list of keys.
        >>> # single device
        >>> kv.push(keys, [mx.nd.ones(shape)]*len(keys))
        >>> b = [mx.nd.zeros(shape)]*len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1].asnumpy()
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]

        >>> # multiple devices:
        >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
        >>> kv.push(keys, b)
        >>> kv.pull(keys, out=b)
        >>> print b[1][1].asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
  */
  def push(keys: Array[Int], values: Array[NDArray], priority: Int = 0): Unit = {
    require(keys.length == values.length, "len(keys) != len(values)")
    val valuePtrs = values.map(_.handle.value)
    checkCall(_LIB.mxKVStorePush(handle, keys.length, keys, valuePtrs, priority))
  }

  /**
   * Pull a single value or a sequence of values from the store.

    Data consistency:

    1. this function returns after adding an operator to the engine. But any
    further read on out will be blocked until it is finished.

    2. pull is always called after all previous push and pull on the same
    key are finished

    3. It pulls the newest value from the store.

    Parameters
    ----------
    key : int or list of int
        Keys

    out: NDArray or list of NDArray or list of list of NDArray
        According values

    priority : int, optional
        The priority of the push operation.
        The higher the priority, the faster this action is likely
        to be executed before other push actions.

    Examples
    --------
    >>> # pull a single key-value pair
    >>> a = mx.nd.zeros(shape)
    >>> kv.pull(3, out=a)
    >>> print a.asnumpy()
    [[ 2.  2.  2.]
    [ 2.  2.  2.]]

    >>> # pull into multiple devices
    >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
    >>> kv.pull(3, out=b)
    >>> print b[1].asnumpy()
    [[ 2.  2.  2.]
    [ 2.  2.  2.]]

    >>> # pull a list of key-value pairs.
    >>> # On single device
    >>> keys = [5, 7, 9]
    >>> b = [mx.nd.zeros(shape)]*len(keys)
    >>> kv.pull(keys, out=b)
    >>> print b[1].asnumpy()
    [[ 2.  2.  2.]
    [ 2.  2.  2.]]
    >>> # On multiple devices
    >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
    >>> kv.pull(keys, out=b)
    >>> print b[1][1].asnumpy()
    [[ 2.  2.  2.]
    [ 2.  2.  2.]]
   */
  def pull(keys: Array[Int], outs: Array[NDArray], priority: Int = 0): Unit = {
    require(keys.length == outs.length, "len(keys) != len(outs)")
    val outPtrs = outs.map(_.handle.value)
    checkCall(_LIB.mxKVStorePull(handle, keys.length, keys, outPtrs, priority))
  }

  /**
   * Set a push updater into the store.

        This function only changes the local store. Use set_optimizer for
        multi-machines.

        Parameters
        ----------
        updater : function
            the updater function

        Examples
        --------
        >>> def update(key, input, stored):
        ...     print "update on key: %d" % key
        ...     stored += input * 2
        >>> kv._set_updater(update)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        >>> kv.push(3, mx.nd.ones(shape))
        update on key: 3
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 6.  6.  6.]
        [ 6.  6.  6.]]
   */
  private def _setUpdater(updater: MXKVStoreUpdater): Unit = {
    this.updaterFunc = updater
    checkCall(_LIB.mxKVStoreSetUpdater(handle, updaterFunc, null))
  }
}
