# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
""" Key value store interface of MXNet for parameter synchronization."""
from __future__ import absolute_import

import ctypes
import pickle
from .ndarray import NDArray
from .ndarray import _ndarray_cls
from .base import _LIB
from .base import check_call, c_array, c_str, string_types, mx_uint, py_str
from .base import NDArrayHandle, KVStoreHandle
from . import optimizer as opt

def _ctype_key_value(keys, vals):
    """
    Returns ctype arrays for the key-value args, and the whether string keys are used.
    For internal use only.
    """
    if isinstance(keys, (tuple, list)):
        assert(len(keys) == len(vals))
        c_keys = []
        c_vals = []
        use_str_keys = None
        for key, val in zip(keys, vals):
            c_key_i, c_val_i, str_keys_i = _ctype_key_value(key, val)
            c_keys += c_key_i
            c_vals += c_val_i
            use_str_keys = str_keys_i if use_str_keys is None else use_str_keys
            assert(use_str_keys == str_keys_i), "inconsistent types of keys detected."
        c_keys_arr = c_array(ctypes.c_char_p, c_keys) if use_str_keys \
                     else c_array(ctypes.c_int, c_keys)
        c_vals_arr = c_array(NDArrayHandle, c_vals)
        return (c_keys_arr, c_vals_arr, use_str_keys)

    assert(isinstance(keys, (int,) + string_types)), \
           "unexpected type for keys: " + str(type(keys))
    use_str_keys = isinstance(keys, string_types)
    if isinstance(vals, NDArray):
        c_keys = c_array(ctypes.c_char_p, [c_str(keys)]) if use_str_keys \
                 else c_array(ctypes.c_int, [keys])
        return (c_keys, c_array(NDArrayHandle, [vals.handle]), use_str_keys)
    else:
        for value in vals:
            assert(isinstance(value, NDArray))
        c_keys = c_array(ctypes.c_char_p, [c_str(keys)] * len(vals)) if use_str_keys \
                 else c_array(ctypes.c_int, [keys] * len(vals))
        return (c_keys, c_array(NDArrayHandle, [value.handle for value in vals]), use_str_keys)

def _updater_wrapper(updater):
    """A wrapper for the user-defined handle."""
    def updater_handle(key, lhs_handle, rhs_handle, _):
        """ ctypes function """
        lhs = _ndarray_cls(NDArrayHandle(lhs_handle))
        rhs = _ndarray_cls(NDArrayHandle(rhs_handle))
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
        self._str_updater_func = None

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
        key : str, int, or sequence of str or int
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
        >>> keys = [5, 7, 9]
        >>> kv.init(keys, [mx.nd.ones(shape)]*len(keys))
        """
        ckeys, cvals, use_str_keys = _ctype_key_value(key, value)
        if use_str_keys:
            check_call(_LIB.MXKVStoreInitEx(self.handle, mx_uint(len(ckeys)), ckeys, cvals))
        else:
            check_call(_LIB.MXKVStoreInit(self.handle, mx_uint(len(ckeys)), ckeys, cvals))

    def push(self, key, value, priority=0):
        """ Pushes a single or a sequence of key-value pairs into the store.

        This function returns immediately after adding an operator to the engine.
        The actual operation is executed asynchronously after all previous `push`
        and `pull` calls for the same input key(s) are finished.
        There is no synchronization between workers. One can use ``_barrier()``
        to sync all workers.

        Parameters
        ----------
        key : str, int, or sequence of str or int
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
        >>> keys = [4, 5, 6]
        >>> kv.push(keys, [mx.nd.ones(shape)]*len(keys))
        >>> b = [mx.nd.zeros(shape)]*len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1].asnumpy()
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]

        >>> # multiple devices:
        >>> keys = ['7', '8', '9']
        >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
        >>> kv.push(keys, b)
        >>> kv.pull(keys, out=b)
        >>> print b[1][1].asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        """
        ckeys, cvals, use_str_keys = _ctype_key_value(key, value)
        if use_str_keys:
            check_call(_LIB.MXKVStorePushEx(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, ctypes.c_int(priority)))
        else:
            check_call(_LIB.MXKVStorePush(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, ctypes.c_int(priority)))


    def pull(self, key, out=None, priority=0):
        """ Pulls a single value or a sequence of values from the store.

        This function returns immediately after adding an operator to the engine.
        Subsequent attempts to read from the `out` variable will be blocked until the
        pull operation completes.

        `pull` is executed asynchronously after all previous `push` and `pull` calls
        for the same input key(s) are finished.

        The returned values are gauranteed to be the latest values in the store.

        For row_sparse values, please use `row_sparse_pull` instead.

        Parameters
        ----------
        key : str, int, or sequence of str or int
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
        >>> keys = [5, 7, 9]
        >>> b = [mx.nd.zeros(shape)]*len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
        >>> # On multiple devices
        >>> keys = ['6', '8', '10']
        >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1][1].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
        """
        assert(out is not None)
        ckeys, cvals, use_str_keys = _ctype_key_value(key, out)
        if use_str_keys:
            check_call(_LIB.MXKVStorePullEx(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, ctypes.c_int(priority)))
        else:
            check_call(_LIB.MXKVStorePull(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, ctypes.c_int(priority)))

    def row_sparse_pull(self, key, out=None, priority=0, row_ids=None):
        """ Pulls a single row_sparse value or a sequence of row_sparse values from the store
         with specified row_ids.

        `row_sparse_pull` is executed asynchronously after all previous
        `push`/`pull`/`row_sparse_pull` calls for the same input key(s) are finished.

        The returned values are guaranteed to be the latest values in the store.

        Parameters
        ----------
        key : str, int, or sequence of str or int
            Keys.

        out: NDArray or list of NDArray or list of list of NDArray
            Values corresponding to the keys. The stype is expected to be row_sparse

        priority : int, optional
            The priority of the pull operation.
            Higher priority pull operations are likely to be executed before
            other pull actions.

        row_ids : NDArray or list of NDArray
            The row_ids for which to pull for each value. Each row_id is an 1D-NDArray \
            whose values don't have to be unique nor sorted.

        Examples
        --------
        >>> shape = (3, 3)
        >>> kv.init('3', mx.nd.ones(shape).tostype('row_sparse'))
        >>> a = mx.nd.zeros(shape, stype='row_sparse')
        >>> row_ids = mx.nd.array([0, 2], dtype='int64')
        >>> kv.row_sparse_pull('3', out=a, row_ids=row_ids)
        >>> print a.asnumpy()
        [[ 1.  1.  1.]
        [ 0.  0.  0.]
        [ 1.  1.  1.]]
        >>> duplicate_row_ids = mx.nd.array([2, 2], dtype='int64')
        >>> kv.row_sparse_pull('3', out=a, row_ids=duplicate_row_ids)
        >>> print a.asnumpy()
        [[ 0.  0.  0.]
        [ 0.  0.  0.]
        [ 1.  1.  1.]]
        >>> unsorted_row_ids = mx.nd.array([1, 0], dtype='int64')
        >>> kv.row_sparse_pull('3', out=a, row_ids=unsorted_row_ids)
        >>> print a.asnumpy()
        [[ 1.  1.  1.]
        [ 1.  1.  1.]
        [ 0.  0.  0.]]
        """
        assert(out is not None)
        assert(row_ids is not None)
        ckeys, cvals, use_str_keys = _ctype_key_value(key, out)
        _, crow_ids, _ = _ctype_key_value(key, row_ids)
        assert(len(crow_ids) == len(cvals)), \
               "the number of row_ids doesn't match the number of values"
        if use_str_keys:
            check_call(_LIB.MXKVStorePullRowSparseEx(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, crow_ids, ctypes.c_int(priority)))
        else:
            check_call(_LIB.MXKVStorePullRowSparse(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, crow_ids, ctypes.c_int(priority)))


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
        # set updater with int keys
        _updater_proto = ctypes.CFUNCTYPE(
            None, ctypes.c_int, NDArrayHandle, NDArrayHandle, ctypes.c_void_p)
        self._updater_func = _updater_proto(_updater_wrapper(updater))
        # set updater with str keys
        _str_updater_proto = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, NDArrayHandle, NDArrayHandle, ctypes.c_void_p)
        self._str_updater_func = _str_updater_proto(_updater_wrapper(updater))
        check_call(_LIB.MXKVStoreSetUpdaterEx(self.handle, self._updater_func,
                                              self._str_updater_func, None))


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
