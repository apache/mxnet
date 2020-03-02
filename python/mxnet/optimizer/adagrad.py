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
"""AdaGrad optimizer"""
from __future__ import absolute_import
from ..ndarray import (zeros, clip, sqrt, square)
from ..ndarray import sparse
from .optimizer import Optimizer, register

__all__ = ['AdaGrad']


@register
class AdaGrad(Optimizer):
    """AdaGrad optimizer.

    This class implements the AdaGrad optimizer described in *Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization*, and available at
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad, clip_gradient) + wd * weight
        history += square(grad)
        weight -= learning_rate * grad / (sqrt(history) + epsilon)

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    See Also
    ----------
    :meth:`mxnet.ndarray.sparse.adagrad_update`.

    Parameters
    ----------
    learning_rate : float, default 0.01
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    epsilon : float, default 1e-6
        Small value to avoid division by 0.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False or grad is not sparse, step is called,
        otherwise, fused_step is called.

    """
    def __init__(self, learning_rate=0.01, epsilon=1e-6, use_fused_step=True, **kwargs):
        super(AdaGrad, self).__init__(learning_rate=learning_rate,
                                      use_fused_step=use_fused_step,
                                      **kwargs)
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return zeros(weight.shape, weight.context, stype=weight.stype)  # history

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

            # update history
            history = state
            history[:] += square(grad)
            d = grad / (sqrt(history) + self.epsilon)

            # update weight
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
            is_sparse = grad.stype == 'row_sparse'

            if is_sparse:
                self._update_count(index)
                lr = self._get_lr(index)
                wd = self._get_wd(index)
                kwargs = {'epsilon': self.epsilon, 'rescale_grad': self.rescale_grad}
                if self.clip_gradient:
                    kwargs['clip_gradient'] = self.clip_gradient

                history = state

                # When grad is sparse, update weight with fused kernel
                sparse.adagrad_update(weight, grad, history, out=weight, lr=lr, wd=wd, **kwargs)
            else:
                # When the grad is not sparse, the func step is called to update weight and state
                self.step([index], [weight], [grad], [state])
