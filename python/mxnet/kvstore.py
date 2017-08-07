# coding: utf-8
""" Key value store interface of MXNet for parameter synchronization."""
from __future__ import absolute_import

import ctypes
import pickle
from .ndarray import NDArray
from .base import _LIB
from .base import check_call, c_array, c_str, string_types, mx_uint, py_str
from .base import NDArrayHandle, KVStoreHandle
from . import optimizer as opt

def _ctype_key_value(keys, vals):
    if isinstance(keys, (tuple, list)):
        assert(len(keys) == len(vals))
        c_keys = []
        c_vals = []
        for key, val in zip(keys, vals):
            c_key_i, c_val_i = _ctype_key_value(key, val)
            c_keys += c_key_i
            c_vals += c_val_i
        return (c_array(ctypes.c_char_p, c_keys), c_array(NDArrayHandle, c_vals))
    names = []
    keys = str(keys)
    if isinstance(vals, NDArray):
        names.append(c_str(keys))
        return (c_array(ctypes.c_char_p, names),
                c_array(NDArrayHandle, [vals.handle]))
    else:
        for value in vals:
            assert(isinstance(value, NDArray))
        return (c_array(ctypes.c_char_p, [c_str(keys)] * len(vals)),
                c_array(NDArrayHandle, [value.handle for value in vals]))

def _updater_wrapper(updater):
    """A wrapper for the user-defined handle."""
    def updater_handle(key, lhs_handle, rhs_handle, _):
        """ ctypes function """
        lhs = NDArray(NDArrayHandle(lhs_handle))
        rhs = NDArray(NDArrayHandle(rhs_handle))
        updater(key, lhs, rhs)
    return updater_handle


class KVStore(object):
    """A key-value store for synchronization of values, over multiple devices."""
    def __init__(self, handle):
        """Initializes a new KVStore.

        Parameters
        ----------
        handle : KVStoreHandle
            `KVStore` handle of C API.
        """
        assert isinstance(handle, KVStoreHandle)
        self.handle = handle
        self._updater = None
        self._updater_func = None

    def __del__(self):
        check_call(_LIB.MXKVStoreFree(self.handle))

    def init(self, key, value):
        """ Initializes a single or a sequence of key-value pairs into the store.

        For each key, one must `init` it before calling `push` or `pull`.
        When multiple workers invoke `init` for the same key, only
        the value supplied by worker with rank `0` is used. This function returns
        after data has been initialized successfully.

        Parameters
        ----------
        key : str or sequence of str
            The keys.
        value : NDArray or sequence of NDArray
            Values corresponding to the keys.

        Examples
        --------
        >>> # init a single key-value pair
        >>> shape = (2,3)
        >>> kv = mx.kv.create('local')
        >>> kv.init('3', mx.nd.ones(shape)*2)
        >>> a = mx.nd.zeros(shape)
        >>> kv.pull('3', out=a)
        >>> print a.asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # init a list of key-value pairs
        >>> keys = ['5', '7', '9']
        >>> kv.init(keys, [mx.nd.ones(shape)]*len(keys))
        """
        ckeys, cvals = _ctype_key_value(key, value)
        check_call(_LIB.MXKVStoreInitEx(self.handle, mx_uint(len(ckeys)), ckeys, cvals))

    def push(self, key, value, priority=0):
        """ Pushes a single or a sequence of key-value pairs into the store.

        This function returns immediately after adding an operator to the engine.
        The actual operation is executed asynchronously after all previous `push`
        and `pull` calls for the same input key(s) are finished.
        There is no synchronization between workers. One can use ``_barrier()``
        to sync all workers.

        Parameters
        ----------
        key : str or list of str
            Keys.

        value : NDArray or list of NDArray or list of list of NDArray
            Values corresponding to the keys.

        priority : int, optional
            The priority of the push operation.
            Higher priority push operations are likely to be executed before
            other push actions.

        Examples
        --------
        >>> # push a single key-value pair
        >>> kv.push('3', mx.nd.ones(shape)*8)
        >>> kv.pull('3', out=a) # pull out the value
        >>> print a.asnumpy()
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]

        >>> # aggregate the value and the push
        >>> gpus = [mx.gpu(i) for i in range(4)]
        >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
        >>> kv.push('3', b)
        >>> kv.pull('3', out=a)
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
        """
        ckeys, cvals = _ctype_key_value(key, value)
        check_call(_LIB.MXKVStorePushEx(
            self.handle, mx_uint(len(ckeys)), ckeys, cvals,
            ctypes.c_int(priority)))


    def pull(self, key, out=None, priority=0):
        """ Pulls a single value or a sequence of values from the store.

        This function returns immediately after adding an operator to the engine.
        Subsequent attempts to read from the `out` variable will be blocked until the
        pull operation completes.

        `pull` is executed asynchronously after all previous `push` and `pull` calls
        for the same input key(s) are finished.

        The returned values are gauranteed to be the latest values in the store.

        Parameters
        ----------
        key : int or list of int
            Keys.

        out: NDArray or list of NDArray or list of list of NDArray
            Values corresponding to the keys.

        priority : int, optional
            The priority of the pull operation.
            Higher priority pull operations are likely to be executed before
            other pull actions.

        Examples
        --------
        >>> # pull a single key-value pair
        >>> a = mx.nd.zeros(shape)
        >>> kv.pull('3', out=a)
        >>> print a.asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # pull into multiple devices
        >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
        >>> kv.pull('3', out=b)
        >>> print b[1].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # pull a list of key-value pairs.
        >>> # On single device
        >>> keys = ['5', '7', '9']
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
        """
        assert(out is not None)
        ckeys, cvals = _ctype_key_value(key, out)
        check_call(_LIB.MXKVStorePullEx(
            self.handle, mx_uint(len(ckeys)), ckeys, cvals,
            ctypes.c_int(priority)))

    def set_optimizer(self, optimizer):
        """ Registers an optimizer with the kvstore.

        When using a single machine, this function updates the local optimizer.
        If using multiple machines and this operation is invoked from a worker node,
        it will serialized the optimizer with pickle and send it to all servers.
        The function returns after all servers have been updated.

        Parameters
        ----------
        optimizer : Optimizer
            The new optimizer for the store

        Examples
        --------

        >>> kv = mx.kv.create()
        >>> shape = (2, 2)
        >>> weight = mx.nd.zeros(shape)
        >>> kv.init(3, weight)
        >>> # set the optimizer for kvstore as the default SGD optimizer
        >>> kv.set_optimizer(mx.optimizer.SGD())
        >>> grad = mx.nd.ones(shape)
        >>> kv.push(3, grad)
        >>> kv.pull(3, out = weight)
        >>> # weight is updated via gradient descent
        >>> weight.asnumpy()
        array([[-0.01, -0.01],
               [-0.01, -0.01]], dtype=float32)
        """
        is_worker = ctypes.c_int()
        check_call(_LIB.MXKVStoreIsWorkerNode(ctypes.byref(is_worker)))

        # pylint: disable=invalid-name
        if 'dist' in self.type and is_worker.value:
            # send the optimizer to server
            try:
                # use ASCII protocol 0, might be slower, but not a big ideal
                optim_str = pickle.dumps(optimizer, 0)
            except:
                raise
            self._send_command_to_servers(0, optim_str)
        else:
            self._set_updater(opt.get_updater(optimizer))

    @property
    def type(self):
        """ Returns the type of this kvstore.

        Returns
        -------
        type : str
            the string type
        """
        kv_type = ctypes.c_char_p()
        check_call(_LIB.MXKVStoreGetType(self.handle, ctypes.byref(kv_type)))
        return py_str(kv_type.value)

    @property
    def rank(self):
        """ Returns the rank of this worker node.

        Returns
        -------
        rank : int
            The rank of this node, which is in range [0, num_workers())
        """
        rank = ctypes.c_int()
        check_call(_LIB.MXKVStoreGetRank(self.handle, ctypes.byref(rank)))
        return rank.value

    @property
    def num_workers(self):
        """Returns the number of worker nodes.

        Returns
        -------
        size :int
            The number of worker nodes.
        """
        size = ctypes.c_int()
        check_call(_LIB.MXKVStoreGetGroupSize(self.handle, ctypes.byref(size)))
        return size.value

    def save_optimizer_states(self, fname):
        """Saves the optimizer (updater) state to a file. This is often used when checkpointing
        the model during training.

        Parameters
        ----------
        fname : str
            Path to the output states file.
        """
        assert self._updater is not None, "Cannot save states for distributed training"
        with open(fname, 'wb') as fout:
            fout.write(self._updater.get_states())

    def load_optimizer_states(self, fname):
        """Loads the optimizer (updater) state from the file.

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
        assert self._updater is not None, "Cannot save states for distributed training"
        self._updater.set_states(open(fname, 'rb').read())

    def _set_updater(self, updater):
        """Sets a push updater into the store.

        This function only changes the local store. When running on multiple machines one must
        use `set_optimizer`.

        Parameters
        ----------
        updater : function
            The updater function.

        Examples
        --------
        >>> def update(key, input, stored):
        ...     print "update on key: %d" % key
        ...     stored += input * 2
        >>> kv._set_updater(update)
        >>> kv.pull('3', out=a)
        >>> print a.asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        >>> kv.push('3', mx.nd.ones(shape))
        update on key: 3
        >>> kv.pull('3', out=a)
        >>> print a.asnumpy()
        [[ 6.  6.  6.]
        [ 6.  6.  6.]]
        """
        self._updater = updater
        _updater_proto = ctypes.CFUNCTYPE(
            None, ctypes.c_int, NDArrayHandle, NDArrayHandle, ctypes.c_void_p)
        self._updater_func = _updater_proto(_updater_wrapper(updater))
        check_call(_LIB.MXKVStoreSetUpdater(self.handle, self._updater_func, None))


    def _barrier(self):
        """Invokes global barrier among all worker nodes.

        For example, assume there are `n` machines. We would like machine `0` to first
        `init` the values and then have all the workers `pull` the initialized value.
        Before pulling, we can place invoke `_barrier()` to guarantee that the
        initialization is finished.
        """
        check_call(_LIB.MXKVStoreBarrier(self.handle))

    def _send_command_to_servers(self, head, body):
        """Sends a command to all server nodes.

        Sending command to a server node will cause that server node to invoke
        ``KVStoreServer.controller`` to execute the command.

        This function returns after the command has been executed on all server
        nodes.

        Parameters
        ----------
        head : int
            the head of the command.
        body : str
            the body of the command.
        """
        check_call(_LIB.MXKVStoreSendCommmandToServers(
            self.handle, mx_uint(head), c_str(body)))

def create(name='local'):
    """Creates a new KVStore.

    For single machine training, there are two commonly used types:

    ``local``: Copies all gradients to CPU memory and updates weights there.

    ``device``: Aggregates gradients and updates weights on GPUs. With this setting,
    the KVStore also attempts to use GPU peer-to-peer communication,
    potentially accelerating the communication.

    For distributed training, KVStore also supports a number of types:

    ``dist_sync``: Behaves similarly to ``local`` but with one major difference.
    With ``dist_sync``, batch-size now means the batch size used on each machine.
    So if there are ``n`` machines and we use batch size ``b``,
    then ``dist_sync`` behaves like ``local`` with batch size ``n * b``.

    ``dist_device_sync``: Identical to ``dist_sync`` with the difference similar
    to ``device`` vs ``local``.

    ``dist_async``: Performs asynchronous updates.
    The weights are updated whenever gradients are received from any machine.
    No two updates happen on the same weight at the same time. However, the order is not
    guaranteed.

    Parameters
    ----------
    name : {'local', 'device', 'dist_sync', 'dist_device_sync', 'dist_async'}
        The type of KVStore.
    Returns
    -------
    kv : KVStore
        The created KVStore.
    """
    if not isinstance(name, string_types):
        raise TypeError('name must be a string')
    handle = KVStoreHandle()
    check_call(_LIB.MXKVStoreCreate(c_str(name),
                                    ctypes.byref(handle)))
    return KVStore(handle)
