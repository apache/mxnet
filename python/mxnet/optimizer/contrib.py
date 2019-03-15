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

# pylint: disable=too-many-lines
"""Contrib optimizers."""
from ..ndarray import (NDArray, clip, contrib, mean, sqrt, square, zeros)
from .optimizer import Optimizer

# convenience wrapper for Optimizer.Register
register = Optimizer.register  # pylint: disable=invalid-name

__all__ = ['GroupAdaGrad']


@register
class GroupAdaGrad(Optimizer):
    """Adagrad optimizer with row-wise learning rates.

    This class implements the AdaGrad optimizer described in *Adaptive
    Subgradient Methods for Online Learning and Stochastic Optimization*, and
    available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf but
    uses only a single learning rate for every row of the parameter array.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad, clip_gradient)
        history += mean(square(grad), axis=1, keepdims=True)
        div = grad / sqrt(history + float_stable_eps)
        weight -= div * lr

    Weights are updated lazily if the gradient is sparse.

    For details of the update algorithm see
    :class:`~mxnet.ndarray.contrib.group_adagrad_update`.

    This optimizer accepts the following parameters in addition to those
    accepted by :class:`.Optimizer`. Weight decay is not supported.

    Parameters
    ----------
    eps: float, optional
        Initial value of the history accumulator. Avoids division by 0.

    """

    def __init__(self, eps=1e-5, **kwargs):
        super(GroupAdaGrad, self).__init__(**kwargs)
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        assert len(weight.shape) == 2
        history = zeros(
            (weight.shape[0], 1), weight.context, stype=weight.stype)
        return history

    def update(self, index, weight, grad, state):
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        assert wd == 0, 'Weight decay is not supported for GroupAdaGrad'

        is_sparse = grad.stype == 'row_sparse'
        if is_sparse:
            kwargs = {
                'epsilon': self.float_stable_eps,
                'rescale_grad': self.rescale_grad
            }
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            contrib.group_adagrad_update(
                weight,
                grad,
                state,
                out=weight,
                lr=lr,
                **kwargs)
        else:
            grad = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            state[:] += mean(square(grad), axis=1, keepdims=True)
            div = lr * grad / sqrt(state + self.float_stable_eps)
            weight[:] -= div
