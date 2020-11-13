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

"""Weight updating functions."""
import os
import warnings
import numpy
from .optimizer import Optimizer, register
from ..ndarray import zeros, NDArray, full
from ..ndarray.contrib import mp_adamw_update, adamw_update, \
    multi_mp_adamw_update, multi_adamw_update

__all__ = ['BERTAdam']

@register
class BERTAdam(Optimizer):
    """The Adam optimizer with weight decay regularization for BERT.

    Updates are applied by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        w = w - learning_rate * (m / (sqrt(v) + epsilon) + wd * w)

    Note that this is different from `mxnet.optimizer.Adam`, where L2 loss is added and
    accumulated in m and v. In BERTAdam, the weight decay term decoupled from gradient
    based update.

    This is also slightly different from the AdamW optimizer described in
    *Fixing Weight Decay Regularization in Adam*, where the schedule multiplier and
    learning rate is decoupled, and the bias-correction terms are removed.
    The BERTAdam optimizer uses the same learning rate to apply gradients
    w.r.t. the loss and weight decay.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`mxnet.optimizer.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional, default is 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional, default is 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional, default is 1e-6
        Small value to avoid division by 0.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 **kwargs):
        super(BERTAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.aggregate_num = max(1, min(50, int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE',
                                                          '4'))))

    def create_state_multi_precision(self, index, weight):
        """multi-precision state creation function."""
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn('Accumulating with float16 in optimizer can lead to '
                          'poor accuracy or slow convergence. '
                          'Consider using multi_precision=True option of the '
                          'BERTAdam optimizer')
        return self.create_state(index, weight)

    def create_state(self, _, weight):
        """state creation function."""
        return (zeros(weight.shape, weight.context, dtype=weight.dtype), #mean
                zeros(weight.shape, weight.context, dtype=weight.dtype)) #variance

    def update(self, index, weight, grad, state):
        """update function"""
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        """multi-precision update function"""
        use_multi_precision = self.multi_precision and weight[0].dtype == numpy.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

    def _update_impl(self, indices, weight, grad, state, multi_precision=False):
        """update function"""
        aggregate = self.aggregate_num > 1
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weight = [weight]
            grad = [grad]
            state = [state]
        for w_i, g_i in zip(weight, grad):
            assert(isinstance(w_i, NDArray))
            assert(isinstance(g_i, NDArray))
            aggregate = (aggregate and
                         w_i.stype == 'default' and
                         g_i.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)

        # pylint: disable=access-member-before-definition
        if not isinstance(self.rescale_grad, NDArray):
            self.rescale_grad = full(shape=(1,), val=self.rescale_grad, ctx=weight[0].context)
        else:
            self.rescale_grad = self.rescale_grad.as_in_context(weight[0].context)

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
                    mean, var = list(zip(*state[sidx:eidx]))
                    multi_adamw_update(weight[sidx:eidx],
                                       grad[sidx:eidx],
                                       mean, var,
                                       out=weight[sidx:eidx],
                                       size=len(weight[sidx:eidx]),
                                       lrs=list(numpy.ones(len(weight[sidx:eidx]))),
                                       wds=wds[sidx:eidx],
                                       etas=lrs[sidx:eidx],
                                       **kwargs)
                else:
                    mean_var = list(zip(*state[sidx:eidx]))[0]
                    tmean_var = list(zip(*mean_var))
                    mean = tmean_var[0]
                    var = tmean_var[1]
                    multi_mp_adamw_update(weight[sidx:eidx],
                                          grad[sidx:eidx],
                                          mean, var,
                                          list(zip(*state[sidx:eidx]))[1],
                                          out=weight[sidx:eidx],
                                          size=len(weight[sidx:eidx]),
                                          lrs=list(numpy.ones(len(weight[sidx:eidx]))),
                                          wds=wds[sidx:eidx],
                                          etas=lrs[sidx:eidx],
                                          **kwargs)
                current_index += self.aggregate_num
        else:
            for w_i, g_i, s_i, lr, wd in zip(weight, grad, state, lrs, wds):
                if not multi_precision:
                    mean, var = s_i
                    adamw_update(w_i, g_i, mean, var, out=w_i,
                                 lr=1, wd=wd, eta=lr, **kwargs)
                else:
                    mean, var = s_i[0]
                    mp_adamw_update(w_i, g_i, mean, var, s_i[1], out=w_i,
                                    lr=1, wd=wd, eta=lr, **kwargs)
