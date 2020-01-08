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
# pylint: disable=wildcard-import
"""Stochastic block class."""
__all__ = ['StochasticBlock', 'StochasticSequential']

from functools import wraps
from ...block import HybridBlock
from ...utils import _indent


class StochasticBlock(HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(StochasticBlock, self).__init__(prefix=prefix, params=params)
        self._losses = []
        self._losscache = []
        self._count = 0

    def add_loss(self, loss):
        self._count += 1
        self._losscache.append(loss)

    @staticmethod
    def collectLoss(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            # Loss from hybrid_forward
            func_out = func(self, *args, **kwargs)
            collected_loss = self._losscache
            self._losscache = []
            return (func_out, collected_loss)

        return inner

    def __call__(self, *args):
        """Calls forward. Only accepts positional arguments."""
        for hook in self._forward_pre_hooks.values():
            hook(self, args)
        out = self.forward(*args)
        self._losses.extend(out[1])
        for hook in self._forward_hooks.values():
            hook(self, args, out)
        # if _mx_npx.is_np_array():
        #         _check_all_np_ndarrays(out)
        return out[0]

    @property
    def losses(self):
        return self._losses


class StochasticSequential(StochasticBlock):
    def __init__(self, prefix=None, params=None):
        super(StochasticSequential, self).__init__(prefix=prefix, params=params)

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in blocks:
            self.register_child(block)

    @StochasticBlock.collectLoss
    def hybrid_forward(self, F, x):
        for block in self._children.values():
            x = block(x)
            if hasattr(block, '_losses'):
                self.add_loss(block.losses)
        return x

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block.__repr__(), 2))
                            for key, block in self._children.items()])
        return s.format(name=self.__class__.__name__,
                        modstr=modstr)

    def __getitem__(self, key):
        layers = list(self._children.values())[key]
        if isinstance(layers, list):
            net = type(self)(prefix=self._prefix)
            with net.name_scope():
                net.add(*layers)
            return net
        else:
            return layers

    def __len__(self):
        return len(self._children)
