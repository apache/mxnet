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
"""Stochastic block class."""
__all__ = ['StochasticBlock', 'StochasticSequential']

from functools import wraps
from ...block import HybridBlock
from ...nn import HybridSequential
from ...utils import _indent


class StochasticBlock(HybridBlock):
    """`StochasticBlock` extends `HybridBlock` to support accumulating loss
    in the forward phase, which is extremely useful in building Bayesian Neural Network,
    where the loss function is composed of a classification loss and a KL loss.

    """

    def __init__(self, prefix=None, params=None):
        super(StochasticBlock, self).__init__(prefix=prefix, params=params)
        self._losses = []
        self._losscache = []

    def add_loss(self, loss):
        self._losscache.append(loss)

    @staticmethod
    def collectLoss(func):
        """To accumulate loss during the forward phase, one could first decorate
        hybrid_forward with `StochasticBlock.collectLoss,
        and then collect the loss tensor `x` by calling self.add_loss(x).
        For example, in the following forward function,
        we generate samples from a Gaussian parameterized by `loc` and `scale` and
        accumulate the KL-divergence between it and its prior into the block's loss storage.:
        @StochasticBlock.collectLoss
        def hybrid_forward(self, F, loc, scale):
            qz = mgp.Normal(loc, scale)
            # prior
            pz = mgp.Normal(F.np.zeros_like(loc), F.np.ones_like(scale))
            self.add_loss(mgp.kl_divergence(qz, pz))
            return qz.sample()
        """
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
        self._losses = []
        # pylint: disable=no-value-for-parameter
        out = self.forward(*args)  # out[0]: net output, out[1]: collected loss
        self._losses.extend(out[1])
        for hook in self._forward_hooks.values():
            hook(self, args, out)
        return out[0]

    @property
    def losses(self):
        return self._losses

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError


class StochasticSequential(StochasticBlock):
    """Stack StochasticBlock sequentially.
    """
    def __init__(self, prefix=None, params=None):
        super(StochasticSequential, self).__init__(prefix=prefix, params=params)
        self._layers = []

    def add(self, *blocks):
        """Adds block on top of the stack."""
        for block in blocks:
            self._layers.append(block)
            self.register_child(block)

    @StochasticBlock.collectLoss
    def hybrid_forward(self, F, x, *args):
        for block in self._children.values():
            x = block()(x, *args)
            args = []
            if isinstance(x, (tuple, list)):
                args = x[1:]
                x = x[0]
        if args:
            x = tuple([x] + list(args))
        for block in self._layers:
            if hasattr(block, '_losses'):
                self.add_loss(block._losses)
        return x

    def __repr__(self):
        s = '{name}(\n{modstr}\n)'
        modstr = '\n'.join(['  ({key}): {block}'.format(key=key,
                                                        block=_indent(block().__repr__(), 2))
                            for key, block in self._children.items()])
        return s.format(name=self.__class__.__name__, modstr=modstr)

    def __getitem__(self, key):
        layers = list(self._children.values())[key]
        if isinstance(layers, list):
            net = type(self)(prefix=self._prefix)
            with net.name_scope():
                net.add(*(l() for l in layers))
            return net
        else:
            return layers()

    def __len__(self):
        return len(self._children)

