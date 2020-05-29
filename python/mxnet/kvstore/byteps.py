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
""" BytePS backend for MXNet KVStore"""
from __future__ import absolute_import

from ..ndarray import NDArray
from .base import KVStoreBase

__all__ = ['BytePS']


@KVStoreBase.register
class BytePS(KVStoreBase):
    """BytePS backend for MXNet KVStore interface."""

    def __init__(self):
        """Initializes a new KVStore."""
        try:
            import byteps.mxnet as bps
            self.handle = bps
        except ModuleNotFoundError as err:
            print('Did not find BytePS library. Please install BytePS first')
            raise err
        except ImportError as err:
            print('Did not find BytePS library. Please install BytePS first')
            raise err
        self.handle.init()

    def broadcast(self, key, value, out, priority=0):
        """ Broadcast the value NDArray at rank 0 to all ranks' out. If out is None,
        the result is stored in `value`.

        Parameters
        ----------
        key : str, or int
            The keys.
        value : NDArray, or list of NDArray
            Values corresponding to the key.
        out : NDArray, or lise of NDArray
            Values corresponding to the keys.

        Examples
        --------
        >>> # broadcast a single key-value pair
        >>> shape = (2,3)
        >>> kv = mx.kv.create('byteps')
        >>> a = mx.nd.zeros(shape)
        >>> kv.broadcast('3', mx.nd.ones(shape)*2, out=a)
        >>> print a.asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
        """
        # do not accept list or tuple for key/value
        assert isinstance(key, (str, int))

        # unpack the list if it contains just one NDArray
        value = value[0] if isinstance(
            value, list) and len(value) == 1 else value
        assert isinstance(
            value, NDArray), "The type of value can only be NDArray or list of NDArray which has only one element."
        assert value.context.device_type == 'gpu', "Byteps KVStore only support GPU context for broadcast value."

        # optimzation when out = value or out = [value]
        if isinstance(out, (list, tuple)) and len(out) == 1:
            inplace = value is out[0]
        else:
            inplace = value is out

        if inplace:
            broadcast_value = value
        else:
            broadcast_value = value.copy()
        # for non-root-rank, assign value with 0, thus the result of pushpull will be
        # equal to the value of root-rank, thus implementing broadcast.
        root_rank = 0
        if self.rank != root_rank:
            broadcast_value.__imul__(0)
        self.handle.byteps_declare_tensor(str(key))
        self.handle.byteps_push_pull(broadcast_value, version=0, priority=priority,
                                     name=str(key), is_average=False)
        # Make sure tensors pushed to MXNet engine get processed such that all
        # workers are synced before starting training.
        broadcast_value.wait_to_read()

        out = out if isinstance(out, list) else [out]
        for o in out:
            broadcast_value.copyto(o)

    def pushpull(self, key, value, out=None, priority=0):
        """ Performs push and pull a single value from the store.
        This function is coalesced form of push and pull operations.
        `value` is pushed to the kvstore server for the specified keys and the aggregated
        values are pulled from the server to `out`. If `out` is not specified the pulled
        values are written to `value`.

        Parameters
        ----------
        key : str, or int
            The key.
        value : NDArray, or list of NDArray
            Values corresponding to the key.
        out: NDArray, or list of NDArray
            Values corresponding to the key.
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
        """
        # the most common operation operates on one NDArray as `value`, and
        # `out` is set to None, for inplace pushpull.

        assert isinstance(key, (str, int))

        # unpack the list if it contains just one NDArray
        value = value[0] if isinstance(
            value, list) and len(value) == 1 else value
        assert isinstance(
            value, NDArray), "The type of value can only be NDArray or list of NDArray which has only one element."
        assert value.context.device_type == 'gpu', "Byteps KVStore only support GPU context for pushpull value"

        # optimzation when out = value or out = [value]
        if isinstance(out, (list, tuple)) and len(out) == 1:
            inplace = value is out[0]
        else:
            inplace = value is out

        if inplace:
            pushpull_value = value
        else:
            pushpull_value = value.copy()

        self.handle.byteps_declare_tensor(str(key))
        self.handle.byteps_push_pull(pushpull_value, version=0, priority=priority,
                                     name=str(key), is_average=False)

        if out is not None:
            out = out if isinstance(out, list) else [out]
            for o in out:
                pushpull_value.copyto(o)

    @staticmethod
    def is_capable(capability):
        """Queries if the KVStore type supports certain capability, such as optimizer algorithm,
        gradient compression, sparsity, etc.
        As byteps server does not store weight, this function will return false for any capabilities.

        Parameters
        ----------
        capability: str
            The capability to query

        Returns
        -------
        result : bool
            Whether the capability is supported or not.
        """
        return False

    @property
    def type(self):
        """ Returns the type of this kvstore.

        Returns
        -------
        type : str
            the string type
        """
        return 'byteps'

    @property
    def local_rank(self):
        """ Returns the local rank of this worker on the node.

        Returns
        -------
        rank : int
            The local rank of this node, which is in range [0, num_workers_on_current_node())
        """
        return self.handle.local_rank()

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

    def set_optimizer(self, optimizer):
        """
        Not Implement yet.

        Parameters
        ----------
        optimizer : KVStoreBase
            The new optimizer for the store
        """
        raise NotImplementedError()

    def save_optimizer_states(self, fname, dump_optimizer=False):
        """
        Not Implement yet.

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
        """
        Not Implement yet.

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
        raise NotImplementedError()
