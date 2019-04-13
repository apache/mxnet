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

import itertools

import numpy as np

import mxnet as mx
from mxnet.test_utils import *


# * GroupAdaGrad
class PyGroupAdaGrad(mx.optimizer.Optimizer):
    """The python reference of Group AdaGrad optimizer.

    Parameters
    ----------
    eps: float, optional
        Small value to avoid division by 0.

    """

    def __init__(self, eps=1e-5, **kwargs):
        super(PyGroupAdaGrad, self).__init__(**kwargs)
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        assert len(weight.shape) == 2
        history = mx.nd.zeros(
            (weight.shape[0], 1), weight.context, stype=weight.stype)
        return history

    def update(self, index, weight, grad, state):
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        assert wd == 0

        history = state
        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)
        history[:] += mx.nd.mean(mx.nd.square(grad), axis=1, keepdims=True)
        div = lr * grad / mx.nd.sqrt(history + self.float_stable_eps)
        weight[:] -= div


def test_group_adagrad():
    mx.random.seed(0)
    opt1 = PyGroupAdaGrad
    opt2 = mx.optimizer.contrib.GroupAdaGrad
    shape = (3, 4)
    eps_options = [{}, {'eps': 1e-8}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    for dtype in [np.float32]:
        for options in itertools.product(eps_options, cg_options, rg_options):
            kwarg = dict(wd=0.0)
            for option in options:
                kwarg.update(option)
            compare_optimizer(
                opt1(**kwarg),
                opt2(**kwarg),
                shape,
                dtype,
                compare_states=False)
            compare_optimizer(
                opt1(**kwarg),
                opt2(**kwarg),
                shape,
                dtype,
                w_stype='row_sparse',
                g_stype='row_sparse',
                compare_states=False)
            compare_optimizer(
                opt1(**kwarg),
                opt2(**kwarg),
                shape,
                dtype,
                g_stype='row_sparse',
                compare_states=False)

def test_adamw():
    shape = (3, 4)
    weight = mx.nd.random.uniform(shape=shape)
    weight_ref = weight.copy()
    grad = mx.nd.random.uniform(shape=shape)
    m = mx.nd.random.uniform(shape=shape)
    v = mx.nd.random.uniform(shape=shape)
    rescale_grad = mx.nd.array([10])
    eta, lr, wd, epsilon = 1, 1, 0, 1e-8
    beta1, beta2 = 0.9, 0.999
    kwargs = {'eta': eta, 'lr': lr, 'wd': wd, 'epsilon': epsilon,
              'beta1': beta1, 'beta2': beta2}

    # update is skipped for rescale = nan scalar
    mx.nd.contrib.adamw_update(weight, grad, m, v,
                               np.nan, out=weight, **kwargs)
    # weight remains unchanged
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight.asnumpy())

    # update is skipped for rescale = 0
    mx.nd.contrib.adamw_update(weight, grad, m, v,
                               rescale_grad * 0, out=weight, **kwargs)
    # weight remains unchanged
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight.asnumpy())

    # update is skipped for rescale = nan
    mx.nd.contrib.adamw_update(weight, grad, m, v,
                               rescale_grad * np.nan, out=weight, **kwargs)
    # weight remains unchanged
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight.asnumpy())

    # update is skipped for rescale = inf
    mx.nd.contrib.adamw_update(weight, grad, m, v,
                               rescale_grad * np.inf, out=weight, **kwargs)
    # weight remains unchanged
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight.asnumpy())

    # multi-precision update is skipped for rescale = nan
    weight_fp16 = weight.astype('float16')
    grad_fp16 = grad.astype('float16')
    weight_fp16_ref = weight_fp16.copy()
    mx.nd.contrib.mp_adamw_update(weight_fp16, grad_fp16, m, v, weight,
                                  rescale_grad * np.nan, out=weight_fp16, **kwargs)
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight.asnumpy())
    mx.test_utils.assert_almost_equal(weight_fp16_ref.asnumpy(), weight_fp16.asnumpy())

    # multi-precision update is skipped for rescale = nan scalar
    mx.nd.contrib.mp_adamw_update(weight_fp16, grad_fp16, m, v, weight,
                                  np.nan, out=weight_fp16, **kwargs)
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight.asnumpy())
    mx.test_utils.assert_almost_equal(weight_fp16_ref.asnumpy(), weight_fp16.asnumpy())

    # multi-precision update is skipped for rescale = inf
    mx.nd.contrib.mp_adamw_update(weight_fp16, grad_fp16, m, v, weight,
                                  rescale_grad * np.inf, out=weight_fp16, **kwargs)
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight.asnumpy())
    mx.test_utils.assert_almost_equal(weight_fp16_ref.asnumpy(), weight_fp16.asnumpy())

    # multi-precision update is skipped for rescale = 0
    mx.nd.contrib.mp_adamw_update(weight_fp16, grad_fp16, m, v, weight,
                                  rescale_grad * 0, out=weight_fp16, **kwargs)
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight.asnumpy())
    mx.test_utils.assert_almost_equal(weight_fp16_ref.asnumpy(), weight_fp16.asnumpy())

    # reference normal update
    grad_rescale = rescale_grad * grad
    m_ref = beta1*m + (1-beta1)*grad_rescale
    v_ref = beta2*v + (1-beta2)*(grad_rescale**2)
    weight_ref = weight - eta * (1 * m_ref / (v_ref.sqrt() + epsilon) + weight * wd)
    m_test = m.copy()
    v_test = v.copy()
    weight_test = weight.copy()
    # op normal update
    mx.nd.contrib.adamw_update(weight_test, grad, m_test, v_test,
                               rescale_grad, out=weight_test, **kwargs)
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight_test.asnumpy())
    mx.test_utils.assert_almost_equal(m_ref.asnumpy(), m_test.asnumpy())
    mx.test_utils.assert_almost_equal(v_ref.asnumpy(), v_test.asnumpy())

    # reference normal multi-precision update
    m_fp32 = m.copy()
    v_fp32 = v.copy()
    weight_fp32 = weight.copy()
    grad_rescale = rescale_grad * grad_fp16.astype('float32')
    m_ref = beta1*m_fp32 + (1-beta1)*grad_rescale
    v_ref = beta2*v_fp32 + (1-beta2)*(grad_rescale**2)
    weight_ref = weight - eta * (1 * m_ref / (v_ref.sqrt() + epsilon) + weight * wd)
    weight_fp16_ref = weight_ref.astype('float16')
    # op normal multi-precision update
    mx.nd.contrib.mp_adamw_update(weight_fp16, grad_fp16, m_fp32, v_fp32, weight_fp32,
                                  rescale_grad, out=weight_fp16, **kwargs)
    mx.test_utils.assert_almost_equal(m_ref.asnumpy(), m_fp32.asnumpy())
    mx.test_utils.assert_almost_equal(v_ref.asnumpy(), v_fp32.asnumpy())
    mx.test_utils.assert_almost_equal(weight_ref.asnumpy(), weight_fp32.asnumpy())
    mx.test_utils.assert_almost_equal(weight_fp16_ref.asnumpy(), weight_fp16.asnumpy())


if __name__ == '__main__':
    import nose
    nose.runmodule()
