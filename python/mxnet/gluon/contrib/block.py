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
# pylint: disable= arguments-differ
"""Base contrib container class."""
__all__ = ['SparseBlock']

from ... import nd
from ..block import Block

class SparseBlock(Block):
    """`SparseBlock` only supports forwarding with NDArray.
    """
    def __init__(self, prefix=None, params=None):
        super(SparseBlock, self).__init__(prefix=prefix, params=params)

    def forward(self, x, *args):
        """Defines the forward computation. Arguments has to be :py:class:`NDArray`."""
        assert isinstance(x, nd.NDArray), \
            "SparseBlock requires the first argument to forward be NDArray, " \
            "but got %s"%type(x)
        with x.context as ctx:
            params = {}
            for i, j in self._reg_params.items():
                if j._stype == 'default':
                    params[i] = j.data(ctx)
                else:
                    params['%s_param'%i] = j
        return self.sparse_forward(x, *args, **params)

    def sparse_forward(self, x, *args, **kwargs):
        """Overrides to define sparse forward computation for this `SparseBlock`.

        Note that the *args for :py:meth:`SparseBlock.sparse_forward` is a list of
        :py:class:`NDArray`s and :py:class:`Parameter`s. If the storage type of any
        Parameter is sparse, the Parameter is passed as :py:class:`Parameter` by itself.
        Otherwise, the Parameter is passed as a :py:class:`NDArray`.

        When overridding sparse_forward, typically one needs to invoke
        :py:meth:`Parameter.row_sparse_data` to access the data of the Parameter.

        Parameters
        ----------
        x : NDArray
            The first input tensor.
        *args : list of Parameter or NDArray
            Additional input tensors or sparse parameters.
        """
        raise NotImplementedError
