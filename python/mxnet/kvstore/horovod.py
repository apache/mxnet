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
from __future__ import absolute_import
from .base import KVStoreBase

__all__ = ['Horovod']


@KVStoreBase.register
class Horovod(KVStoreBase):
    """A communication backend using Horovod."""

    def __init__(self):
        import horovod.mxnet as hvd
        hvd.init()

    @property
    def type(self):
        return 'horovod'

    def broadcast(self, key, value, out=None, priority=0):
        """ Broadcast the `value` NDArray at rank 0 to all ranks

        Parameters
        ----------
        key : str, or int
            The key.

        value : NDArray
            The value corresponding to the key to broadcast. If `out` is not specified,
            `value` NDArray will be updated in-place.

        out : NDArray, list of NDArray
            Output tensor that receives value broadcasted from root process
            If not specified, output will be written to `value`

        priority : int, optional
            The priority of the operation.
            Higher priority operations are likely to be executed before other actions.

        Examples
        --------
        >>> # broadcast a value in-place
        >>> shape = (2,3)
        >>> kv = mx.kv.create('horovod')
        >>> a = mx.nd.ones(shape)
        >>> kv.broadcast('1', value=a)
        >>> print(a.asnumpy())
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]

        >>> a = mx.nd.ones(shape)
        >>> b = mx.nd.zeros(shape)
        >>> kv.broadcast('2', value=a, out=b)
        >>> print(b.asnumpy)
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]
        """
        import horovod.mxnet as hvd

        if out is None:
            hvd.broadcast_(tensor=value, root_rank=0, name=key, priority=priority)
        else:
            out[:] = hvd.broadcast(tensor=value, root_rank=0, name=key, priority=priority)

    def pushpull(self, key, value, out=None, priority=0):
        """ Performs allreduce on a single tensor or a list of tensor objects

        This function performs in-place summation of the input tensor over all the processes.

        The name `pushpull` is a generic term. In Horovod, its action is implemented via
        ring allreduce. Each operation is identified by the 'key'; if `key` is not provided, an
        incremented auto-generated name is used. The tensor type and shape must be
        the same on all processes for a given name. The reduction will not start until all processes
        are ready to send and receive the tensor.

        Parameters
        ----------
        key : str, int, or sequence of str or int
            Keys used to uniquely tag an operation.

        value : NDArray
            Tensor value on one process to be summed. If `out` is not specified, the `value` will
            be modified in-place

        out: NDArray
            Output tensor after allreduce. If not specified, the input tensor `value` will be
            modified in-place.

        priority : int, optional
            The priority of the operation.
            Higher priority operations are likely to be executed before other actions.

        Examples
        --------
        >>> # perform in-place allreduce on tensor a
        >>> shape = (2, 3)
        >>> nworker = kv.num_workers # assume there are 8 processes
        >>> a = mx.nd.ones(shape)
        >>> kv.pushpull('1', a)
        >>> print(a.asnumpy())
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]

        >>> # perform allreduce on tensor a and output to b
        >>> a = mx.nd.ones(shape)
        >>> kv.pushpull('2', a, out=b)
        >>> print(b.asnumpy())
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]
        """
        import horovod.mxnet as hvd

        if out is None:
            hvd.allreduce_(value, average=False, name=key, priority=priority)
        else:
            out[:] = hvd.allreduce(value, average=False, name=key, priority=priority)

    def set_optimizer(self, optimizer):
        pass

    @staticmethod
    def is_capable(capability):
        pass

    def save_optimizer_states(self, fname, dump_optimizer=False):
        pass

    def load_optimizer_states(self, fname):
        pass

    @property
    def rank(self):
        import horovod.mxnet as hvd
        return hvd.rank()

    @property
    def local_rank(self):
        import horovod.mxnet as hvd
        return hvd.local_rank()

    @property
    def num_workers(self):
        import horovod.mxnet as hvd
        return hvd.size()
