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
"""Base container class for contrib neural network models."""
__all__ = ['SparseBlock']

from .. import Block

class SparseBlock(Block):
    """`SparseBlock` only supports forwarding with NDArray.
    """
    def __init__(self, prefix=None, params=None):
        super(SparseBlock, self).__init__(prefix=prefix, params=params)

    def forward(self, x, *args):
        """Defines the forward computation. Arguments has to be :py:class:`NDArray`."""
        assert isinstance(x, NDArray), \
            "SparseBlock requires the first argument to forward be NDArray, " \
            "but got %s"%type(x)
        with x.context as ctx:
            params = {}
            for name, param in self._reg_params.items():
                if param._stype != 'default':
                    params[name] = param
                else:
                    params[name] = param.data(ctx)
            ret = self.sparse_forward(x, *args, **params)
            for name, param in params.items():
                if param._dirty:
                    raise RuntimeError("sparse Parameter '%s' was not pulled during "
                                       "sparse_forward(). Please call param.row_sparse_data() "
                                       "with indices to pull sparse parameters in "
                                       "sparse_forward()" % name)

    def sparse_forward(self, F, x, *args, **kwargs):
        """Overrides to define sparse forward computation for this `SparseBlock`.

        Parameters
        ----------
        x : NDArray
            The first input tensor.
        *args : list of Parameter or NDArray
            Additional input tensors or sparse parameters.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError
