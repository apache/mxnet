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
"""AdaDelta optimizer."""
from __future__ import absolute_import
from ..ndarray import (zeros, clip, sqrt, square)
from .optimizer import Optimizer, register

__all__ = ['AdaDelta']


@register
class AdaDelta(Optimizer):
    """The AdaDelta optimizer.

    This class implements AdaDelta, an optimizer described in  *ADADELTA: An adaptive
    learning rate method*, available at https://arxiv.org/abs/1212.5701.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad, clip_gradient) + wd * weight
        acc_grad = rho * acc_grad + (1. - rho) * grad * grad
        delta = sqrt(acc_delta + epsilon) / sqrt(acc_grad + epsilon) * grad
        acc_delta = rho * acc_delta + (1. - rho) * delta * delta
        weight -= learning_rate * delta

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    learning_rate : float, default 1.0
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    rho: float, default 0.9
        Decay rate for both squared gradients and delta.
    epsilon : float, default 1e-6
        Small value to avoid division by 0.
    use_fused_step : bool, default False
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=1.0, rho=0.9, epsilon=1e-6, use_fused_step=False, **kwargs):
        super(AdaDelta, self).__init__(learning_rate=learning_rate,
                                       use_fused_step=use_fused_step,
                                       **kwargs)
        self.rho = rho
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context),  # accumulated g
                zeros(weight.shape, weight.context))  # accumulated delta

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
            grad += wd * weight

            acc_g, acc_delta = state

            # update g, delta
            acc_g[:] *= self.rho
            acc_g[:] += (1. - self.rho) * square(grad)
            current_delta = sqrt(acc_delta + self.epsilon)
            current_delta /= sqrt(acc_g + self.epsilon)
            current_delta *= grad
            acc_delta[:] *= self.rho
            acc_delta[:] += (1. - self.rho) * square(current_delta)

            # update weight
            weight[:] -= lr * current_delta
