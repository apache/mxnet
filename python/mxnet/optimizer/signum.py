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
"""Signum optimizer."""
from __future__ import absolute_import
from ..ndarray import (zeros, clip)
from ..ndarray import (signsgd_update, signum_update)
from .optimizer import Optimizer, register

__all__ = ['Signum']


@register
class Signum(Optimizer):
    r"""The Signum optimizer that takes the sign of gradient or momentum.

    The optimizer updates the weight by::

        rescaled_grad = rescale_grad * clip(grad, clip_gradient) + wd * weight
        state = momentum * state + (1-momentum)*rescaled_grad
        weight = (1 - lr * wd_lh) * weight - lr * sign(state)

    References
    ----------
    Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli & Anima Anandkumar. (2018).
    signSGD: Compressed Optimisation for Non-Convex Problems. In ICML'18.

    See: https://arxiv.org/abs/1802.04434

    For details of the update algorithm see
    :class:`~mxnet.ndarray.signsgd_update` and :class:`~mxnet.ndarray.signum_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    learning_rate : float, default 0.01
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    momentum : float, optional
       The momentum value.
    wd_lh : float, optional
       The amount of decoupled weight decay regularization, see details in the original paper at:\
       https://arxiv.org/abs/1711.05101
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9, wd_lh=0.0, use_fused_step=True, **kwargs):
        super(Signum, self).__init__(learning_rate=learning_rate,
                                     use_fused_step=use_fused_step,
                                     **kwargs)
        self.momentum = momentum
        self.wd_lh = wd_lh

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=weight.stype)
        return momentum

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

            if state is not None:
                # preprocess grad
                grad *= self.rescale_grad
                if self.clip_gradient is not None:
                    grad = clip(grad, - self.clip_gradient, self.clip_gradient)
                grad += wd * weight

                # update mom
                mom = state
                mom[:] *= self.momentum
                mom[:] -= (1 - self.momentum) * grad

                # update weight
                weight[:] *= 1 - lr * self.wd_lh
                weight[:] += lr * ((mom > 0) - (mom < 0))
            else:
                # update weight
                weight[:] *= 1 - lr * (wd + self.wd_lh)
                weight[:] -= lr * ((grad > 0) - (grad < 0))

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

            kwargs = {'rescale_grad': self.rescale_grad}
            if self.momentum > 0:
                kwargs['momentum'] = self.momentum
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient

            # update weight with fused kernel
            if state is not None:
                if self.wd_lh:
                    kwargs['wd_lh'] = self.wd_lh
                signum_update(weight, grad, state, out=weight,
                              lr=lr, wd=wd, **kwargs)
            else:
                wd += self.wd_lh
                signsgd_update(weight, grad, out=weight,
                               lr=lr, wd=wd, **kwargs)
