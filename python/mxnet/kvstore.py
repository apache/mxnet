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

from array import array
import ctypes
import pickle
from .ndarray import NDArray
from .ndarray import _ndarray_cls
from .base import _LIB, c_str_array, c_handle_array, c_array, c_array_buf, c_str
from .base import check_call, string_types, mx_uint, py_str
from .base import NDArrayHandle, KVStoreHandle
from . import optimizer as opt
from .profiler import set_kvstore_handle

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
        c_vals_arr = c_array(ctypes.c_void_p, c_vals)
        return (c_keys_arr, c_vals_arr, use_str_keys)

    assert(isinstance(keys, (int,) + string_types)), \
           "unexpected type for keys: " + str(type(keys))
    use_str_keys = isinstance(keys, string_types)
    if isinstance(vals, NDArray):
        c_keys = c_str_array([keys]) if use_str_keys \
                 else c_array_buf(ctypes.c_int, array('i', [keys]))
        return (c_keys, c_handle_array([vals]), use_str_keys)
    else:
        for value in vals:
            assert(isinstance(value, NDArray))
        c_keys = c_str_array([keys] * len(vals)) if use_str_keys \
                 else c_array_buf(ctypes.c_int, array('i', [keys] * len(vals)))
        return (c_keys, c_handle_array(vals), use_str_keys)

def _ctype_dict(param_dict):
    """
    Returns ctype arrays for keys and values(converted to strings) in a dictionary
    """
    assert(isinstance(param_dict, dict)), \
        "unexpected type for param_dict: " + str(type(param_dict))
    c_keys = c_array(ctypes.c_char_p, [c_str(k) for k in param_dict.keys()])
    c_vals = c_array(ctypes.c_char_p, [c_str(str(v)) for v in param_dict.values()])
    return (c_keys, c_vals)

def _updater_wrapper(updater):
    """A wrapper for the user-defined handle."""
    def updater_handle(key, lhs_handle, rhs_handle, _):
        """ ctypes function """
        lhs = _ndarray_cls(NDArrayHandle(lhs_handle))
        rhs = _ndarray_cls(NDArrayHandle(rhs_handle))
        updater(key, lhs, rhs)
    return updater_handle

def _get_kvstore_server_command_type(command):
    command_types = {'kController': 0,
                     'kSetMultiPrecision': 1,
                     'kStopServer': 2,
                     'kSyncMode': 3,
                     'kSetGradientCompression': 4,
                     'kSetProfilerParams': 5}
    assert (command in command_types), "Unknown command type to send to server"
    return command_types[command]

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
        value : NDArray, RowSparseNDArray or sequence of NDArray or RowSparseNDArray
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

        >>> # init a row_sparse value
        >>> kv.init('4', mx.nd.ones(shape).tostype('row_sparse'))
        >>> b = mx.nd.sparse.zeros('row_sparse', shape)
        >>> kv.row_sparse_pull('4', row_ids=mx.nd.array([0, 1]), out=b)
        >>> print b
        <RowSparseNDArray 2x3 @cpu(0)>
        """
        ckeys, cvals, use_str_keys = _ctype_key_value(key, value)
        if use_str_keys:
            check_call(_LIB.MXKVStoreInitEx(self.handle, mx_uint(len(ckeys)), ckeys, cvals))
        else:
            check_call(_LIB.MXKVStoreInit(self.handle, mx_uint(len(ckeys)), ckeys, cvals))

    def push(self, key, value, priority=0):
        """ Pushes a single or a sequence of key-value pairs into the store.

        This function returns immediately after adding an operator to the engine.
        The actual operation is executed asynchronously. If there are consecutive
        pushes to the same key, there is no guarantee on the serialization of pushes.
        The execution of a push does not guarantee that all previous pushes are
        finished.
        There is no synchronization between workers.
        One can use ``_barrier()`` to sync all workers.

        Parameters
        ----------
        key : str, int, or sequence of str or int
            Keys.

        value : NDArray, RowSparseNDArray, list of NDArray or RowSparseNDArray,
                or list of list of NDArray or RowSparseNDArray
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
        >>> keys = ['4', '5', '6']
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

        >>> # push a row_sparse value
        >>> b = mx.nd.sparse.zeros('row_sparse', shape)
        >>> kv.init('10', mx.nd.sparse.zeros('row_sparse', shape))
        >>> kv.push('10', mx.nd.ones(shape).tostype('row_sparse'))
        >>> # pull out the value
        >>> kv.row_sparse_pull('10', row_ids=mx.nd.array([0, 1]), out=b)
        >>> print b
        <RowSparseNDArray 2x3 @cpu(0)>
        """
        ckeys, cvals, use_str_keys = _ctype_key_value(key, value)
        if use_str_keys:
            check_call(_LIB.MXKVStorePushEx(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, ctypes.c_int(priority)))
        else:
            check_call(_LIB.MXKVStorePush(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, ctypes.c_int(priority)))


    def pull(self, key, out=None, priority=0, ignore_sparse=True):
        """ Pulls a single value or a sequence of values from the store.

        This function returns immediately after adding an operator to the engine.
        Subsequent attempts to read from the `out` variable will be blocked until the
        pull operation completes.

        `pull` is executed asynchronously after all previous `pull` calls and only
        the last `push` call for the same input key(s) are finished.

        The returned values are guaranteed to be the latest values in the store.

        pull with `RowSparseNDArray` is not supported for dist kvstore.
        Please use ``row_sparse_pull`` instead.

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

        ignore_sparse: bool, optional, default True
            Whether to ignore sparse arrays in the request.

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
            check_call(_LIB.MXKVStorePullWithSparseEx(self.handle, mx_uint(len(ckeys)), ckeys,
                                                      cvals, ctypes.c_int(priority),
                                                      ctypes.c_bool(ignore_sparse)))
        else:
            check_call(_LIB.MXKVStorePullWithSparse(self.handle, mx_uint(len(ckeys)), ckeys,
                                                    cvals, ctypes.c_int(priority),
                                                    ctypes.c_bool(ignore_sparse)))

    def row_sparse_pull(self, key, out=None, priority=0, row_ids=None):
        """ Pulls a single RowSparseNDArray value or a sequence of RowSparseNDArray values \
        from the store with specified row_ids. When there is only one row_id, KVStoreRowSparsePull \
        is invoked just once and the result is broadcast to all the rest of outputs.

        `row_sparse_pull` is executed asynchronously after all previous
        `pull`/`row_sparse_pull` calls and the last `push` call for the
        same input key(s) are finished.

        The returned values are guaranteed to be the latest values in the store.

        Parameters
        ----------
        key : str, int, or sequence of str or int
            Keys.

        out: RowSparseNDArray or list of RowSparseNDArray or list of list of RowSparseNDArray
            Values corresponding to the keys. The stype is expected to be row_sparse

        priority : int, optional
            The priority of the pull operation.
            Higher priority pull operations are likely to be executed before
            other pull actions.

        row_ids : NDArray or list of NDArray
            The row_ids for which to pull for each value. Each row_id is an 1-D NDArray \
            whose values don't have to be unique nor sorted.

        Examples
        --------
        >>> shape = (3, 3)
        >>> kv.init('3', mx.nd.ones(shape).tostype('row_sparse'))
        >>> a = mx.nd.sparse.zeros('row_sparse', shape)
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
        if isinstance(row_ids, NDArray):
            row_ids = [row_ids]
        assert(isinstance(row_ids, list)), \
            "row_ids should be NDArray or list of NDArray"
        first_out = out
        # whether row_ids are the same
        single_rowid = False
        if len(row_ids) == 1 and isinstance(out, list):
            single_rowid = True
            first_out = [out[0]]
        ckeys, cvals, use_str_keys = _ctype_key_value(key, first_out)
        _, crow_ids, _ = _ctype_key_value(key, row_ids)
        assert(len(crow_ids) == len(cvals)), \
               "the number of row_ids doesn't match the number of values"
        if use_str_keys:
            check_call(_LIB.MXKVStorePullRowSparseEx(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, crow_ids, ctypes.c_int(priority)))
        else:
            check_call(_LIB.MXKVStorePullRowSparse(
                self.handle, mx_uint(len(ckeys)), ckeys, cvals, crow_ids, ctypes.c_int(priority)))
        # the result can be copied to other devices without invoking row_sparse_pull
        # if the indices are the same
        if single_rowid:
            for out_i in out[1:]:
                out[0].copyto(out_i)

    def set_gradient_compression(self, compression_params):
        """ Specifies type of low-bit quantization for gradient compression \
         and additional arguments depending on the type of compression being used.

        2bit Gradient Compression takes a positive float `threshold`.
        The technique works by thresholding values such that positive values in the
        gradient above threshold will be set to threshold. Negative values whose absolute
        values are higher than threshold, will be set to the negative of threshold.
        Values whose absolute values are less than threshold will be set to 0.
        By doing so, each value in the gradient is in one of three states. 2bits are
        used to represent these states, and every 16 float values in the original
        gradient can be represented using one float. This compressed representation
        can reduce communication costs. The difference between these thresholded values and
        original values is stored at the sender's end as residual and added to the
        gradient in the next iteration.

        When kvstore is 'local', gradient compression is used to reduce communication
        between multiple devices (gpus). Gradient is quantized on each GPU which
        computed the gradients, then sent to the GPU which merges the gradients. This
        receiving GPU dequantizes the gradients and merges them. Note that this
        increases memory usage on each GPU because of the residual array stored.

        When kvstore is 'dist', gradient compression is used to reduce communication
        from worker to sender. Gradient is quantized on each worker which
        computed the gradients, then sent to the server which dequantizes
        this data and merges the gradients from each worker. Note that this
        increases CPU memory usage on each worker because of the residual array stored.
        Only worker to server communication is compressed in this setting.
        If each machine has multiple GPUs, currently this GPU to GPU or GPU to CPU communication
        is not compressed. Server to worker communication (in the case of pull)
        is also not compressed.

        To use 2bit compression, we need to specify `type` as `2bit`.
        Only specifying `type` would use default value for the threshold.
        To completely specify the arguments for 2bit compression, we would need to pass
        a dictionary which includes `threshold` like:
        {'type': '2bit', 'threshold': 0.5}

        Parameters
        ----------
        compression_params : dict
            A dictionary specifying the type and parameters for gradient compression.
            The key `type` in this dictionary is a
            required string argument and specifies the type of gradient compression.
            Currently `type` can be only `2bit`
            Other keys in this dictionary are optional and specific to the type
            of gradient compression.
        """
        if ('device' in self.type) or ('dist' in self.type): # pylint: disable=unsupported-membership-test
            ckeys, cvals = _ctype_dict(compression_params)
            check_call(_LIB.MXKVStoreSetGradientCompression(self.handle,
                                                            mx_uint(len(compression_params)),
                                                            ckeys, cvals))
        else:
            raise Exception('Gradient compression is not supported for this type of kvstore')

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
        if 'dist' in self.type and is_worker.value: # pylint: disable=unsupported-membership-test
            # send the optimizer to server
            try:
                # use ASCII protocol 0, might be slower, but not a big ideal
                optim_str = py_str(pickle.dumps(optimizer, 0))
            except:
                raise
            cmd = _get_kvstore_server_command_type('kController')
            self._send_command_to_servers(cmd, optim_str)
            if optimizer.multi_precision:
                cmd = _get_kvstore_server_command_type('kSetMultiPrecision')
                self._send_command_to_servers(cmd, '')
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

    def save_optimizer_states(self, fname, dump_optimizer=False):
        """Saves the optimizer (updater) state to a file. This is often used when checkpointing
        the model during training.

        Parameters
        ----------
        fname : str
            Path to the output states file.
        dump_optimizer : bool, default False
            Whether to also save the optimizer itself. This would also save optimizer
            information such as learning rate and weight decay schedules.
        """
        assert self._updater is not None, "Cannot save states for distributed training"
        with open(fname, 'wb') as fout:
            fout.write(self._updater.get_states(dump_optimizer))

    def load_optimizer_states(self, fname):
        """Loads the optimizer (updater) state from the file.

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
        assert self._updater is not None, "Cannot load states for distributed training"
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
    name : {'local', 'device', 'nccl', 'dist_sync', 'dist_device_sync', 'dist_async'}
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
    kv = KVStore(handle)
    set_kvstore_handle(kv.handle)
    return kv
