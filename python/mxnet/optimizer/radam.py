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
"""RAdam optimizer."""
from __future__ import absolute_import
from .optimizer import Optimizer, register
from ..ndarray import (zeros, clip, sqrt, square, full, NDArray)

__all__ = ['RAdam']


@register
class RAdam(Optimizer):
    """The RAdam optimizer.

    This class implements the optimizer described in *On the Variance of the Adaptive Learning Rate and Beyond*,
    available at https://arxiv.org/pdf/1908.03265.pdf.

    Updates are applied by::

        grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1)
        p = p_inf - (2 * step * beta2) / (1 - beta2) 

    If p >= 5::

        lr_a = sqrt((1 - beta2) / (v + epsilon))
        r = sqrt(((p - 4) * (p - 2) * p_inf) / ((p_inf - 4) * (p_inf - 2) * p))
        w = w - (lr * m_hat * r * lr_a)

    If p < 5::

        w = w - (lr * m_hat)

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

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
    use_fused_step : bool, default False
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_fused_step=False, **kwargs):
        super(RAdam, self).__init__(use_fused_step=use_fused_step,
                                    learning_rate=learning_rate,
                                    **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state(self, index, weight):
        """state creation function."""
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

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

            bias_correction1 = 1 - self.beta1 ** t
            bias_correction2 = 1 - self.beta2 ** t

            # preprocess grad
            grad *= self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, - self.clip_gradient, self.clip_gradient)
            grad += wd * weight 

            # update mean and var
            mean, var = state
            mean[:] *= self.beta1
            mean[:] += (1. - self.beta1) * grad
            var[:] *= self.beta2
            var[:] += (1. - self.beta2) * square(grad)

            bias_corrected_mean = mean / bias_correction1

            # maximum length of the approximated SMA
            rho_inf = 2 / (1 - self.beta2) - 1
            # compute the length of the approximated SMA
            rho_t = rho_inf - 2 * step * (self.beta2 ** step) / bias_correction2

            #update weight
            if rho_t >= 5:
                # compute the variance rectification term and update parameters accordingly
                rect = sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                adaptive_lr = sqrt(bias_correction2) / (sqrt(var) + self.epsilon) 
                weight[:] += bias_corrected_mean * lr * adaptive_lr * rect * -1.0
            else:
                weight[:] += bias_corrected_mean * lr * -1.0
