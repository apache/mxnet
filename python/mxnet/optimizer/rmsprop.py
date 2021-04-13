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
"""RMSProp optimizer."""
from __future__ import absolute_import
from ..ndarray import (zeros, clip, sqrt, square)
from ..ndarray import (rmsprop_update, rmspropalex_update)
from .optimizer import Optimizer, register

__all__ = ['RMSProp']


@register
class RMSProp(Optimizer):
    """The RMSProp optimizer.

    Two versions of RMSProp are implemented:

    If ``centered=False``, we follow
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf by
    Tieleman & Hinton, 2012.
    For details of the update algorithm see :class:`~mxnet.ndarray.rmsprop_update`.

    If ``centered=True``, we follow http://arxiv.org/pdf/1308.0850v5.pdf (38)-(45)
    by Alex Graves, 2013.
    For details of the update algorithm see :class:`~mxnet.ndarray.rmspropalex_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    learning_rate : float, default 0.001
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    rho: float, default 0.9
        A decay factor of moving average over past squared gradient.
    momentum: float, default 0.9
        Heavy ball momentum factor. Only used if `centered`=``True``.
    epsilon : float, default 1e-8
        Small value to avoid division by 0.
    centered : bool, default False
        Flag to control which version of RMSProp to use.::

            True: will use Graves's version of `RMSProp`,
            False: will use Tieleman & Hinton's version of `RMSProp`.

    clip_weights : float, optional
        Clips weights into range ``[-clip_weights, clip_weights]``.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.001, rho=0.9, momentum=0.9,
                 epsilon=1e-8, centered=False, clip_weights=None,
                 use_fused_step=True, **kwargs):
        super(RMSProp, self).__init__(learning_rate=learning_rate,
                                      use_fused_step=use_fused_step,
                                      **kwargs)
        self.rho = rho
        self.momentum = momentum
        self.centered = centered
        self.epsilon = epsilon
        self.clip_weights = clip_weights

    def create_state(self, index, weight):
        if self.centered:
            return (
                zeros(weight.shape, weight.context, stype=weight.stype),  # mean
                zeros(weight.shape, weight.context, stype=weight.stype),  # var
                zeros(weight.shape, weight.context, stype=weight.stype))  # mom
        else:
            return zeros(weight.shape, weight.context, stype=weight.stype)  # var

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

            if not self.centered:
                # update var
                var = state
                var[:] *= self.rho
                var[:] += (1 - self.rho) * square(grad)

                # update weight
                d = grad / (sqrt(var) + self.epsilon)
                weight[:] -= lr * d
            else:
                # update mean, var, mom
                mean, var, mom = state
                mean[:] *= self.rho
                mean[:] += (1 - self.rho) * grad
                var[:] *= self.rho
                var[:] += (1 - self.rho) * square(grad)
                mom[:] *= self.momentum
                mom[:] -= lr * grad / sqrt(var - square(mean) + self.epsilon)

                # update weight
                weight[:] += mom

            if self.clip_weights:
                clip(weight, -self.clip_weights, self.clip_weights, out=weight)

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

            kwargs = {'rho': self.rho, 'epsilon': self.epsilon,
                      'rescale_grad': self.rescale_grad}
            if self.centered:
                kwargs['momentum'] = self.momentum
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            if self.clip_weights:
                kwargs['clip_weights'] = self.clip_weights

            # update weight with fused kernel
            if not self.centered:
                var = state
                rmsprop_update(weight, grad, var, out=weight, lr=lr, wd=wd, **kwargs)
            else:
                mean, var, mom = state
                rmspropalex_update(weight, grad, mean, var, mom, out=weight,
                                   lr=lr, wd=wd, **kwargs)
