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
"""Adam optimizer."""
from __future__ import absolute_import
import math
from ..ndarray import (zeros, clip, sqrt, square)
from ..ndarray import adam_update
from .optimizer import Optimizer, register

__all__ = ['Adam']


@register
class Adam(Optimizer):
    """The Adam optimizer.

    This class implements the optimizer described in *Adam: A Method for
    Stochastic Optimization*, available at http://arxiv.org/abs/1412.6980.

    If the storage types of grad is ``row_sparse``, and ``lazy_update`` is True, \
    **lazy updates** at step t are applied by::

        for row in grad.indices:
            rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient) + wd * weight[row]
            m[row] = beta1 * m[row] + (1 - beta1) * rescaled_grad[row]
            v[row] = beta2 * v[row] + (1 - beta2) * (rescaled_grad[row]**2)
            lr = learning_rate * sqrt(1 - beta2**t) / (1 - beta1**t)
            w[row] = w[row] - lr * m[row] / (sqrt(v[row]) + epsilon)

    The lazy update only updates the mean and var for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all indices.
    Compared with the original update, it can provide large improvements in model training
    throughput for some applications. However, it provides slightly different semantics than
    the original update, and may lead to different empirical results.

    Otherwise, **standard updates** at step t are applied by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient) + wd * weight
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        lr = learning_rate * sqrt(1 - beta2**t) / (1 - beta1**t)
        w = w - lr * m / (sqrt(v) + epsilon)

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    For details of the update algorithm, see :class:`~mxnet.ndarray.adam_update`.

    Parameters
    ----------
    learning_rate : float, default 0.001
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    beta1 : float, default 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, default 1e-8
        Small value to avoid division by 0.
    lazy_update : bool, default False
       Default is False. If True, lazy updates are applied \
       if the storage types of weight and grad are both ``row_sparse``.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 lazy_update=False, use_fused_step=True, **kwargs):
        super(Adam, self).__init__(use_fused_step=use_fused_step,
                                   learning_rate=learning_rate,
                                   **kwargs)
        if not self.use_fused_step:
            assert not lazy_update,\
                'When use_fused_step is set to False, lazy_update has to be turned off.'
        self.lazy_update = lazy_update
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
        stype = weight.stype if self.lazy_update else 'default'
        return (zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype))  # variance

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
            t = self._index_update_count[index]

            # preprocess grad
            grad *= self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, - self.clip_gradient, self.clip_gradient)
            grad += wd * weight

            coef1 = 1. - self.beta1**t
            coef2 = 1. - self.beta2**t
            lr *= math.sqrt(coef2) / coef1

            # update mean and var
            mean, var = state
            mean[:] *= self.beta1
            mean[:] += (1. - self.beta1) * grad
            var[:] *= self.beta2
            var[:] += (1. - self.beta2) * square(grad)

            # update weight
            d = mean / (sqrt(var) + self.epsilon)
            weight[:] -= lr * d

    def fused_step(self, indices, weights, grads, states):
        """Perform a fused optimization step using gradients and states.
        Fused kernel is used for update.

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
            t = self._index_update_count[index]

            coef1 = 1. - self.beta1**t
            coef2 = 1. - self.beta2**t

            lr *= math.sqrt(coef2)/coef1

            kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                      'rescale_grad': self.rescale_grad}
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient

            mean, var = state

            # update weight with fused kernel
            adam_update(weight, grad, mean, var, out=weight,
                        lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)
