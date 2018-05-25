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
from ..ndarray import (NDArray, clip, contrib, full, mean, norm, sparse, sqrt,
                       square, zeros)
from .optimizer import Optimizer

# convenience wrapper for Optimizer.Register
register = Optimizer.register  # pylint: disable=invalid-name

__all__ = ['ProximalGroupAdaGrad']


@register
class ProximalGroupAdaGrad(Optimizer):
    """Proximal Adagrad optimizer with row-wise learning rates.

    This class implements the AdaGrad optimizer described in *Adaptive
    Subgradient Methods for Online Learning and Stochastic Optimization*, and
    available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf but
    uses only a single learning rate for every row of the parameter array.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad, clip_gradient)
        history += mean(square(grad), axis=1, keepdims=True)
        div = grad / sqrt(history + float_stable_eps)
        weight -= div * lr

    If `l2_regularization_strength > 0` a proximal operator is used to optimize
    with group lasso objective. Weights are updated lazily if the gradient is
    sparse. In particular, before using a set of weights for a forward pass,
    you may want to ensure that the lazily accumulated group lasso
    regularization is applied. This can be achieved by creating a sparse
    gradient array that contains explicit 0 data for the indices to be updated:

        fake_grad = mx.nd.sparse.row_sparse_array(
            (mx.nd.zeros((len(indices), dim)), indices))
        weight.grad()[:] = fake_grad
        weight.data()._fresh_grad = True
        trainer._optimizer._index_update_count[0] -= 1
        trainer._optimizer.num_update -= 1
        trainer.step(batch_size=1)

    For details of the update algorithm see
    :class:`~mxnet.ndarray.contrib.proximal_group_adagrad_update`.

    This optimizer accepts the following parameters in addition to those
    accepted by :class:`.Optimizer`. Weight decay is not supported.

    Parameters
    ----------
    l2_regularization_strength : float
       Strength of group lasso L2 regularization.
    eps: float, optional
        Initial value of the history accumulator. Avoids division by 0.

    """

    def __init__(self, l2_regularization_strength=0.0, eps=1e-5, **kwargs):
        super(ProximalGroupAdaGrad, self).__init__(**kwargs)
        self.l2_regularization_strength = l2_regularization_strength
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        assert len(weight.shape) == 2
        history = zeros(
            (weight.shape[0], 1), weight.context, stype=weight.stype)
        last_update = None
        if self.l2_regularization_strength > 0:
            last_update = full(
                shape=(weight.shape[0], ),
                val=self.num_update,
                ctx=weight.context)
        else:
            last_update = zeros(1, ctx=weight.context)
        return (history, last_update)

    def update(self, index, weight, grad, state):
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        assert wd == 0, 'Weight decay is not supported for ProximalGroupAdaGrad'

        is_sparse = grad.stype == 'row_sparse'
        history = state[0]
        last_update = state[1]
        if is_sparse:
            kwargs = {
                'epsilon': self.float_stable_eps,
                'rescale_grad': self.rescale_grad
            }
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            if self.l2_regularization_strength:
                kwargs['l2_regularization_strength'] = \
                    self.l2_regularization_strength
            contrib.proximal_group_adagrad_update(
                weight,
                grad,
                history,
                out=weight,
                last_update=last_update,
                lr=lr,
                current_update=self.num_update,
                **kwargs)
        elif self.l2_regularization_strength > 0:
            grad = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            history[:] += mean(square(grad), axis=1, keepdims=True)
            div = lr * grad / sqrt(history + self.float_stable_eps)
            num_skipped = (self.num_update - last_update).expand_dims(1)
            scaled_l2 = lr / sqrt(history + self.float_stable_eps) \
                * self.l2_regularization_strength * num_skipped
            nrm = norm(weight - div, ord=2, axis=1, keepdims=True)
            weight[:] = (weight - div) * (1 - scaled_l2 / nrm)
            weight[:] *= nrm > scaled_l2
            last_update[:] = self.num_update
        else:
            grad = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            history[:] += mean(square(grad), axis=1, keepdims=True)
            div = lr * grad / sqrt(history + self.float_stable_eps)
            weight[:] -= div
