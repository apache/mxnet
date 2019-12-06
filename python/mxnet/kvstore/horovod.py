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
""" Horovod backend for MXNet KVStore"""
from __future__ import absolute_import

from array import array
import ctypes
import pickle
from ..ndarray import NDArray
from ..ndarray import _ndarray_cls
from ..base import _LIB, c_str_array, c_handle_array, c_array, c_array_buf, c_str
from ..base import check_call, string_types, mx_uint, py_str
from ..base import NDArrayHandle
from .. import optimizer as opt
from .base import _ctype_key_value, _ctype_dict, AbstractKVStore

__all__ = ['Horovod']


class Horovod(AbstractKVStore):
    """Horovod backend for MXNet KVStore interface."""

    def __init__(self):
        """Initializes a new KVStore."""
        try:
            import horovod.mxnet as hvd
            self.handle = hvd
        except ImportError as err:
            print('Did not find horovod library. Please install horovod first')
            raise err

    def broadcast(self, key, value, out, priority=0):
        """ Broadcast the value NDArray at rank 0 to all ranks' out. If out is None,
        the result is stored in `value`.

        Parameters
        ----------
        key : str, int, or sequence of str or int
            Keys.

        value : NDArray, list of NDArray, or list of list of NDArray
            Values corresponding to the keys.

        out : NDArray, list of NDArray, or list of list of NDArray
            Values corresponding to the keys.

        Examples
        --------
        >>> # broadcast a single key-value pair
        >>> shape = (2,3)
        >>> kv = mx.kv.create('local')
        >>> a = mx.nd.zeros(shape)
        >>> kv.broadcast('3', mx.nd.ones(shape)*2, out=a)
        >>> print a.asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        """
        if isinstance(key, list):
            assert len(key) == len(value)
            for k, v in zip(key, value):
                self._broadcast(k, v, out, priority=priority)
        else:
            self._broadcast(k, v, out, priority=priority)

    def _broadcast(self, key, value, out, priority=0):
        """ Broadcast the value NDArray at rank 0 to all ranks' out. If out is None,
        the result is stored in `value`.

        Parameters
        ----------
        key : str or int
            Keys.

        value : NDArray, list of NDArray
            Values corresponding to the keys.

        out : NDArray, list of NDArray
            Values corresponding to the keys.
        """
        result = self.handle.broadcast(value, root_rank=0, name=key, priority=priority)
        if isinstance(out, list):
            for o in output:
                result.copyto(o)
        else:
            result.copyto(out)


    def pushpull(self, key, value, out=None, priority=0):
        """ Performs push and pull a single value or a sequence of values from the store.

        This function is coalesced form of push and pull operations. This function returns
        immediately after adding an operator to the engine. Subsequent attempts to read
        from the `out` variable will be blocked until the pull operation completes.

        `value` is pushed to the kvstore server for the specified keys and the updated
        values are pulled from the server to `out`. If `out` is not specified the pulled
        values are written to `value`. The returned values are guaranteed to be the latest
        values in the store.

        Parameters
        ----------
        key : str, int, or sequence of str or int
            Keys.

        value : NDArray, list of NDArray, or list of list of NDArray
            Values corresponding to the keys.

        out: NDArray or list of NDArray or list of list of NDArray
            Values corresponding to the keys.

        priority : int, optional
            The priority of the operation.
            Higher priority operations are likely to be executed before other actions.

        Examples
        --------
        >>> # pushpull a single key-value pair
        >>> kv.pushpull('3', mx.nd.ones(shape)*8, out=a)
        >>> print a.asnumpy()
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]

        >>> # pushpull a list of keys.
        >>> # single device
        >>> keys = ['4', '5', '6']
        >>> b = [mx.nd.zeros(shape)]*len(keys)
        >>> kv.pushpull(keys, [mx.nd.ones(shape)]*len(keys), out=b)
        >>> print b[1].asnumpy()
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]

        """
        if isinstance(key, list):
            assert len(key) == len(value)
            for k, v in zip(key, value):
                self._pushpull(key, value, out=out, priority=priority)
        else:
            self._pushpull(key, value, out=out, priority=priority)

    def _pushpull(self, key, value, out=None, priority=0):
        """ Performs push and pull a single value or a sequence of values from the store.

        This function is coalesced form of push and pull operations. This function returns
        immediately after adding an operator to the engine. Subsequent attempts to read
        from the `out` variable will be blocked until the pull operation completes.

        `value` is pushed to the kvstore server for the specified keys and the updated
        values are pulled from the server to `out`. If `out` is not specified the pulled
        values are written to `value`. The returned values are guaranteed to be the latest
        values in the store.

        Parameters
        ----------
        key : str, or int
            Keys.

        value : NDArray, list of NDArray
            Values corresponding to the keys.

        out: NDArray or list of NDArray
            Values corresponding to the keys.

        priority : int, optional
            The priority of the operation.
            Higher priority operations are likely to be executed before other actions.
        """
        if out is None:
            self.handle.allreduce_(value, average=False, name=key, priority=priority)
        else:
            result = self.handle.allreduce(value, average=False, name=key, priority=priority)
            if isinstance(out, list):
                for o in output:
                    result.copyto(o)
            else:
                result.copyto(out)

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
        """
        raise NotImplementedError()

    @property
    def type(self):
        """ Returns the type of this kvstore.

        Returns
        -------
        type : str
            the string type
        """
        return 'horovod'

    @property
    def rank(self):
        """ Returns the rank of this worker node.

        Returns
        -------
        rank : int
            The rank of this node, which is in range [0, num_workers())
        """
        return self.handle.rank()

    @property
    def num_workers(self):
        """Returns the number of worker nodes.

        Returns
        -------
        size :int
            The number of worker nodes.
        """
        return self.handle.size()

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
        raise NotImplementedError()

    def load_optimizer_states(self, fname):
        """Loads the optimizer (updater) state from the file.

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
        raise NotImplementedError()
