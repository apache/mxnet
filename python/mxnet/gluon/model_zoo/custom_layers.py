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
"""Custom neural network layers in model_zoo."""
__all__ = ['HybridConcurrent', 'Identity']

from ..block import Block, HybridBlock
from ..utils import _indent

class HybridConcurrent(HybridBlock):
    """Lays `HybridBlock`s concurrently.

    Example::

        net = HybridConcurrent()
        # use net's name_scope to give child Blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
            net.add(Identity())
    """
    def __init__(self, concat_dim, prefix=None, params=None):
        super(HybridConcurrent, self).__init__(prefix=prefix, params=params)
        self.concat_dim = concat_dim

    def add(self, block):
        """Adds block on top of the stack."""
        self.register_child(block)

    def hybrid_forward(self, F, x):
        out = []
        for block in self._children:
            out.append(block(x))
        out = F.concat(*out, dim=self.concat_dim)
        return out

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block.__repr__(), 2))
                            for key, block in enumerate(self._children)
                            if isinstance(block, Block)])
        return s.format(name=self.__class__.__name__,
                        modstr=modstr)


class Identity(HybridBlock):
    """Block that passes through the input directly.

    This layer is often used in conjunction with HybridConcurrent
    block for residual connection.

    Example::

        net = HybridConcurrent()
        # use net's name_scope to give child Blocks appropriate names.
        with net.name_scope():
            net.add(nn.Dense(10, activation='relu'))
            net.add(nn.Dense(20))
            net.add(Identity())
    """
    def __init__(self, prefix=None, params=None):
        super(Identity, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x):
        return x
