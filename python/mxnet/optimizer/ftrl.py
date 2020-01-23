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
"""FTRL optimizer."""
from __future__ import absolute_import
from ..ndarray import (zeros, clip, sqrt, square, sign, maximum, abs as NDabs)
from ..ndarray import ftrl_update
from .optimizer import Optimizer, register

__all__ = ['Ftrl']


#pylint: disable=invalid-name
#pylint: disable=line-too-long
@register
class Ftrl(Optimizer):
    """The Ftrl optimizer.

    Referenced from *Ad Click Prediction: a View from the Trenches*, available at
    http://dl.acm.org/citation.cfm?id=2488200.

    eta :
        .. math::
           \\eta_{t,i} = \\frac{learningrate}{\\beta+\\sqrt{\\sum_{s=1}^tg_{s,i}^2}}

    The optimizer updates the weight by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
        n += rescaled_grad**2
        w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)

    If the storage types of weight, state and grad are all ``row_sparse``, \
    **sparse updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
            z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
            n[row] += rescaled_grad[row]**2
            w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)

    The sparse update only updates the z and n for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    For details of the update algorithm, see :class:`~mxnet.ndarray.ftrl_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    learning_rate : float, default 0.1
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    lamda1 : float, default 0.01
        L1 regularization coefficient.
    beta : float, default 1.0
        Per-coordinate learning rate correlation parameter.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """

    def __init__(self, learning_rate=0.1, lamda1=0.01, beta=1.,
                 use_fused_step=True, **kwargs):
        super(Ftrl, self).__init__(learning_rate=learning_rate,
                                   use_fused_step=use_fused_step,
                                   **kwargs)
        self.lamda1 = lamda1
        self.beta = beta

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, stype=weight.stype),  # z
                zeros(weight.shape, weight.context, stype=weight.stype))  # n

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
                grad = clip(grad, - self.clip_gradient, self.clip_gradient)

            # update z, n
            z, n = state

            sigma = - sqrt(n)
            n[:] += square(grad)
            denom = sqrt(n)
            sigma += denom
            sigma /= lr
            z[:] += grad - sigma * weight

            # update weight
            denom += self.beta
            denom /= lr
            denom += wd
            d = sign(z) * maximum(NDabs(z) - self.lamda1, 0)
            weight[:] = - d / denom

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

            kwargs = {'lamda1': self.lamda1, 'beta': self.beta, 'rescale_grad': self.rescale_grad}
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient

            # update weight with fused kernel
            z, n = state
            ftrl_update(weight, grad, z, n, out=weight, lr=lr, wd=wd, **kwargs)
