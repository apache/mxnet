# coding: utf-8
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

# pylint: disable=W0223
"""DCASGD optimizer."""
from __future__ import absolute_import
from ..ndarray import (zeros, clip, square)
from .optimizer import Optimizer, register

__all__ = ['DCASGD']


@register
class DCASGD(Optimizer):
    """The DCASGD optimizer.

    This class implements the optimizer described in *Asynchronous Stochastic Gradient Descent
    with Delay Compensation for Distributed Deep Learning*,
    available at https://arxiv.org/abs/1609.08326.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    learning_rate : float, default 0.1
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    momentum : float, optional
       The momentum value.
    lamda : float, optional
       Scale DC value.
    use_fused_step : bool, default False
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.1, momentum=0.0, lamda=0.04,
                 use_fused_step=False, **kwargs):
        super(DCASGD, self).__init__(learning_rate=learning_rate,
                                     use_fused_step=use_fused_step,
                                     **kwargs)
        self.momentum = momentum
        self.weight_previous = {}
        self.lamda = lamda

    def create_state(self, index, weight):
        if self.momentum == 0.0:
            return None, weight.copy()  # previous weight
        else:
            return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # momentum
                    weight.copy())  # previous weight

    def step(self, indices, weights, grads, states):
        """Perform an optimization step using gradients and states.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)

            # preprocess grad
            grad *= self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            grad += wd * weight

            # update mom, previous_weight
            mom, previous_weight = state

            d = square(grad)
            d *= weight - previous_weight
            d *= self.lamda
            d += grad

            if mom is not None:
                mom[:] *= self.momentum
                mom[:] -= lr * d
            else:
                assert (self.momentum == 0.0)
                mom = d
                mom *= -lr
            previous_weight[:] = weight

            # update weight
            weight[:] += mom
