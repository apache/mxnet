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
"""AdamW optimizer."""
import math
import os
import numpy as np
from .optimizer import Optimizer, register
from ..ndarray import (zeros, clip, sqrt, square, full, NDArray)
from ..ndarray.contrib import mp_adamw_update, adamw_update,\
    multi_mp_adamw_update, multi_adamw_update


__all__ = ['AdamW']


@register
class AdamW(Optimizer):
    """The AdamW optimizer.

    This class implements the optimizer described in *Decoupled Weight Decay Regularization*,
     available at https://arxiv.org/pdf/1711.05101.pdf.

    Updates are applied by::

        grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        lr = learning_rate * sqrt(1 - beta2**t) / (1 - beta1**t)
        w = w - lr * (m / (sqrt(v) + epsilon) + wd * w)


    Also, we can turn of the bias correction term and the updates are as follows::

        grad = clip(grad * rescale_grad, clip_gradient) + wd * weight
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        lr = learning_rate
        w = w - lr * (m / (sqrt(v) + epsilon) + wd * w)

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
    epsilon : float, default 1e-6
        Small value to avoid division by 0.
    correct_bias : bool, default True
       Can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository).
       Default True.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 correct_bias=True, use_fused_step=True, **kwargs):
        super().__init__(use_fused_step=use_fused_step,
                         learning_rate=learning_rate,
                         **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.correct_bias = correct_bias
        self.aggregate_num = max(1, min(50,
                                        int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', '4'))))

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

            # preprocess grad
            grad *= self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, - self.clip_gradient, self.clip_gradient)
            if self.correct_bias:
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
            # add wd
            if wd > 0:
                weight[:] -= lr * wd * weight

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
        multi_precision = self.multi_precision and weights[0].dtype == np.float16
        aggregate = self.aggregate_num > 1
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        for w_i, g_i in zip(weights, grads):
            assert(isinstance(w_i, NDArray))
            assert(isinstance(g_i, NDArray))
            aggregate = (aggregate and
                         w_i.stype == 'default' and
                         g_i.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)
        if self.correct_bias:
            new_lrs = []
            for idx, lr in zip(indices, lrs):
                t = self._index_update_count[idx]
                coef1 = 1. - self.beta1 ** t
                coef2 = 1. - self.beta2 ** t
                new_lrs.append(lr * math.sqrt(coef2) / coef1)
            lrs = new_lrs
        if not isinstance(self.rescale_grad, NDArray):
            self.rescale_grad = full(shape=(1,), val=self.rescale_grad, ctx=weights[0].context)
        else:
            self.rescale_grad = self.rescale_grad.as_in_context(weights[0].context)
        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            current_index = 0
            while current_index < len(indices):
                sidx = current_index
                eidx = min(current_index + self.aggregate_num, len(indices))
                if not multi_precision:
                    mean, var = list(zip(*states[sidx:eidx]))
                    multi_adamw_update(weights[sidx:eidx],
                                       grads[sidx:eidx],
                                       mean, var,
                                       out=weights[sidx:eidx],
                                       size=len(weights[sidx:eidx]),
                                       lrs=list(np.ones(len(weights[sidx:eidx]))),
                                       wds=wds[sidx:eidx],
                                       etas=lrs[sidx:eidx],
                                       **kwargs)
                else:
                    mean_var = list(zip(*states[sidx:eidx]))[0]
                    tmean_var = list(zip(*mean_var))
                    mean = tmean_var[0]
                    var = tmean_var[1]
                    multi_mp_adamw_update(weights[sidx:eidx],
                                          grads[sidx:eidx],
                                          mean, var,
                                          list(zip(*states[sidx:eidx]))[1],
                                          out=weights[sidx:eidx],
                                          size=len(weights[sidx:eidx]),
                                          lrs=list(np.ones(len(weights[sidx:eidx]))),
                                          wds=wds[sidx:eidx],
                                          etas=lrs[sidx:eidx],
                                          **kwargs)
                current_index += self.aggregate_num
        else:
            for w_i, g_i, s_i, lr, wd in zip(weights, grads, states, lrs, wds):
                if not multi_precision:
                    mean, var = s_i
                    adamw_update(w_i, g_i, mean, var, out=w_i,
                                 lr=1, wd=wd, eta=lr, **kwargs)
                else:
                    mean, var = s_i[0]
                    mp_adamw_update(w_i, g_i, mean, var, s_i[1], out=w_i,
                                    lr=1, wd=wd, eta=lr, **kwargs)
