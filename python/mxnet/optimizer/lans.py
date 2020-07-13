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
"""LANS optimizer."""
from __future__ import absolute_import
import numpy
from ..ndarray import (zeros, clip, sqrt, where, square, ones_like,
                       maximum, minimum)
from ..ndarray.contrib import (multi_lans_update, multi_mp_lans_update)
from .optimizer import Optimizer, register

__all__ = ['LANS']


@register
class LANS(Optimizer):
    """LANS Optimizer.

    Referenced from 'Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes'
    (http://arxiv.org/abs/2006.13484)

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
    lower_bound : float, default None
        Lower limit of norm of weight
    upper_bound : float, default None
        Upper limit of norm of weight
    aggregate_num : int, default 4
        Number of weights to be aggregated in a list.
        They are passed to the optimizer for a single optimization step.
        In default, all the weights are aggregated.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 lower_bound=None, upper_bound=None, aggregate_num=4, use_fused_step=True,
                 **kwargs):
        assert aggregate_num <= 45,\
            'When use_fused_step is True, LAMB only supports aggregate_num <= 45,' \
            ' and receives {}'.format(aggregate_num)
        super(LANS, self).__init__(learning_rate=learning_rate,
                                   aggregate_num=aggregate_num,
                                   use_fused_step=use_fused_step,
                                   **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def create_state(self, index, weight):
        stype = weight.stype
        return (zeros(weight.shape, weight.context, dtype=numpy.float32, stype=stype),  # mean
                zeros(weight.shape, weight.context, dtype=numpy.float32, stype=stype))  # var

    def step(self, indices, weights, grads, states):
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

            # preprocess grad
            grad *= self.rescale_grad
            grad /= grad.norm()
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)

            # update mean, var
            mean, var = state
            mean[:] *= self.beta1
            mean[:] += (1. - self.beta1) * grad
            var[:] *= self.beta2
            var[:] += (1. - self.beta2) * square(grad)

            r1 = weight.norm()
            if self.lower_bound is not None:
                r1 = maximum(r1, self.lower_bound)
            if self.upper_bound is not None:
                r1 = minimum(r1, self.upper_bound)

            # apply bias correction
            coef1 = 1. - self.beta1 ** t
            coef2 = 1. - self.beta2 ** t
            mean_hat = mean / coef1
            var_hat = var / coef2
            sqrt(var_hat, out=var_hat)
            var_hat += self.epsilon
            mean_hat /= var_hat
            mean_hat += wd * weight

            g = mean_hat
            r2 = g.norm()

            # calculate lans_trust_ratio for first part
            ratio_m = r1 / r2
            # becomes NaN if ratio == NaN or 0, otherwise 0
            nan_or_zero = 1 - ratio_m / ratio_m
            r_m = where(nan_or_zero, ones_like(ratio_m), ratio_m)

            # update weight using first part of the estimator
            g *= lr * r_m * self.beta1
            weight[:] -= g

            # calculate the second part of the estimator
            mean_hat = grad / var_hat
            mean_hat += wd * weight

            g = mean_hat
            r2 = g.norm()

            # calculate lans_trust_ratio for second part
            ratio_g = r1 / r2
            # becomes NaN if ratio == NaN or 0, otherwise 0
            nan_or_zero = 1 - ratio_g / ratio_g
            r_g = where(nan_or_zero, ones_like(ratio_g), ratio_g)

            # update weight using second part of the estimator
            g *= lr * r_g * (1 - self.beta1)
            weight[:] -= g

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
        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)

        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient
        if self.lower_bound:
            kwargs['lower_bound'] = self.lower_bound
        if self.upper_bound:
            kwargs['upper_bound'] = self.upper_bound

        step_counts = []
        for index in indices:
            step_counts.append(self._index_update_count[index])

        multi_precision = self.multi_precision and weights[0].dtype == numpy.float16

        if not multi_precision:
            mean, var = list(zip(*states))
            multi_lans_update(weights, grads, mean, var,
                              out=weights, step_count=step_counts,
                              lrs=lrs, wds=wds, **kwargs)
        else:
            weights32, mean_var = list(zip(*states))
            mean, var = list(zip(*mean_var))
            multi_mp_lans_update(weights, grads,
                                 mean, var, weights32,
                                 out=weights, step_count=step_counts,
                                 lrs=lrs, wds=wds, **kwargs)

    def update_multi_precision(self, indices, weights, grads, states):
        """Override update_multi_precision.
        """
        if self.use_fused_step:
            self.update(indices, weights, grads, states)
        else:
            super(LANS, self).update_multi_precision(indices, weights, grads, states)
