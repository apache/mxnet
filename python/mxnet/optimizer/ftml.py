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
"""FTML optimizer."""
from __future__ import absolute_import
from ..ndarray import (zeros, clip, sqrt, square)
from ..ndarray import ftml_update
from .optimizer import Optimizer, register

__all__ = ['FTML']


@register
class FTML(Optimizer):
    """The FTML optimizer.

    This class implements the optimizer described in
    *FTML - Follow the Moving Leader in Deep Learning*,
    available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.

    Denote time step by t. The optimizer updates the weight by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient) + wd * weight
        v = beta2 * v + (1 - beta2) * square(rescaled_grad)
        d_t = (1 - power(beta1, t)) / lr * (square_root(v / (1 - power(beta2, t))) + epsilon)
        z = beta1 * z + (1 - beta1) * rescaled_grad - (d_t - beta1 * d_(t-1)) * weight
        weight = - z / d_t

    For details of the update algorithm, see :class:`~mxnet.ndarray.ftml_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    learning_rate : float, default 0.0025
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    beta1 : float, default 0.6
        0 < beta1 < 1. Generally close to 0.5.
    beta2 : float, default 0.999
        0 < beta2 < 1. Generally close to 1.
    epsilon : float, default 1e-8
        Small value to avoid division by 0.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.0025, beta1=0.6, beta2=0.999, epsilon=1e-8,
                 use_fused_step=True, **kwargs):
        super(FTML, self).__init__(learning_rate=learning_rate,
                                   use_fused_step=use_fused_step,
                                   **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype), # d_0
                zeros(weight.shape, weight.context, dtype=weight.dtype), # v_0
                zeros(weight.shape, weight.context, dtype=weight.dtype)) # z_0

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

            # update d, v, z
            d, v, z = state

            v[:] *= self.beta2
            v[:] += (1. - self.beta2) * square(grad)
            sigma = - self.beta1 * d
            d[:] = sqrt(v / coef2) + self.epsilon
            d[:] *= coef1 / lr
            sigma += d
            z[:] *= self.beta1
            z[:] += (1. - self.beta1) * grad
            z[:] -= sigma * weight

            # update weight
            weight[:] = - z / d

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

            kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                      'rescale_grad': self.rescale_grad, 't': t}
            if self.clip_gradient:
                kwargs['clip_grad'] = self.clip_gradient

            d, v, z = state

            # update weight with fused kernel
            ftml_update(weight, grad, d, v, z, out=weight, lr=lr, wd=wd, **kwargs)
