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
import pytest

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import xfail_when_nonstandard_decimal_separator


@xfail_when_nonstandard_decimal_separator
def test_group_adagrad():
    mx.random.seed(0)
    opt1 = mx.optimizer.contrib.GroupAdaGrad
    opt2 = mx.optimizer.contrib.GroupAdaGrad
    shapes = [(3, 4), [5, 6]]
    eps_options = [{}, {'epsilon': 1e-8}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    agg_options = [{}, {'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float32]:
        for options in itertools.product(eps_options, cg_options, rg_options, agg_options):
            kwarg = dict(wd=0.0)
            for option in options:
                kwarg.update(option)
            compare_optimizer(
                opt1(use_fused_step=False, **kwarg),
                opt2(use_fused_step=True, **kwarg),
                shapes,
                dtype)
            compare_optimizer(
                opt1(use_fused_step=False, **kwarg),
                opt2(use_fused_step=True, **kwarg),
                shapes,
                dtype,
                w_stype='row_sparse',
                g_stype='row_sparse')
            compare_optimizer(
                opt1(use_fused_step=False, **kwarg),
                opt2(use_fused_step=True, **kwarg),
                shapes,
                dtype,
                g_stype='row_sparse')

def _fn_noimpl(*args, **kwargs):
    raise NotImplementedError()

class _AdamLikeTestHelper:
    fn_update = _fn_noimpl
    fn_multi_update = _fn_noimpl
    fn_mp_update = _fn_noimpl
    fn_multi_mp_update = _fn_noimpl
    @staticmethod
    def ref_impl(m, v, weight, grad_rescale, beta1, beta2, lr, eta, wd, epsilon, clip_grad=-1):
        '''Returns (mean_ref, v_ref, weight_ref)'''
        raise NotImplementedError()
    @classmethod
    def run_test(cls, num_elem=1, aggregate=False):
        aggregate = aggregate or num_elem > 1
        rescale_factor = 10
        eta, lr, wd, epsilon = 1, 1, 0.1, 1e-8
        beta1, beta2 = 0.9, 0.999
        clip_gradient = np.random.uniform(rescale_factor, rescale_factor)
        weight, grad, m, v, etas, lrs, wds, weight_ref = [], [], [], [], [], [], [], []
        for i in range(num_elem):
            shape = (np.random.randint(3, high=10), np.random.randint(3, high=10))
            weight.append(mx.nd.random.uniform(shape=shape))
            grad.append(mx.nd.random.uniform(-1.0, 1.0, shape=shape))
            m.append(mx.nd.random.uniform(shape=shape))
            v.append(mx.nd.random.uniform(shape=shape))
            etas.append(eta - 1 / np.random.uniform(9, 10))
            lrs.append(lr - 1 / np.random.uniform(9, 10))
            wds.append(wd - 1 / np.random.uniform(95, 105))
            weight_ref.append(weight[i].copy())

        if aggregate:
            kwargs = {'etas': etas, 'lrs': lrs, 'wds': wds}
        else:
            kwargs = {'eta': etas[0], 'lr': lrs[0], 'wd': wds[0]}

        kwargs.update([('epsilon', epsilon), ('beta1', beta1), ('beta2', beta2), ('clip_gradient', clip_gradient)])

        # Test 1: Update is skipped for rescale = nan scalar
        rescale_grad = mx.nd.array([rescale_factor])
        tested_grad = [rescale_grad * 0, rescale_grad * np.nan, rescale_grad * np.inf]
        tested_rescaled_grad = [np.nan]
        tested_rescaled_grad.extend(tested_grad)

        for rescaled_grad in tested_rescaled_grad:
            if aggregate:
                cls.fn_multi_update(weight, grad, m, v,
                                     rescaled_grad, out=weight, **kwargs)
            else:
                cls.fn_update(weight[0], grad[0], m[0], v[0],
                               rescaled_grad, out=weight[0], **kwargs)
            # weights should remain unchanged
            for j in range(num_elem):
                assert_almost_equal(weight_ref[j], weight[j])

        # Test 2: Same as Test 1 for multi-precision update
        weight_fp16, grad_fp16, weight_fp16_refs = [], [], []
        for i in range(num_elem):
            weight_fp16.append(weight[i].astype('float16'))
            grad_fp16.append(grad[i].astype('float16'))
            weight_fp16_refs.append(weight_fp16[i].copy())

        for rescaled_grad in tested_grad:
            if aggregate:
                cls.fn_multi_mp_update(weight_fp16, grad_fp16, m, v, weight,
                                       rescaled_grad, out=weight_fp16, **kwargs)
            else:
                cls.fn_mp_update(weight_fp16[0], grad_fp16[0], m[0], v[0], weight[0],
                                 rescaled_grad, out=weight_fp16[0], **kwargs)
            # weights should remain unchanged
            for i in range(num_elem):
                assert_almost_equal(weight_ref[i], weight[i])
                assert_almost_equal(weight_fp16_refs[i], weight_fp16[i])

        # Test 3: Reference normal update
        grad_rescale, weight_test, m_refs, v_refs, weight_refs = [], [], [], [], []
        for i in range(num_elem):
            grad_rescale.append(rescale_grad * grad[i])
            m_ref, v_ref, weight_ref = cls.ref_impl(
                m[i], v[i], weight[i], grad_rescale[i],
                beta1, beta2, lrs[i], etas[i], wds[i], epsilon, clip_gradient)
            m_refs.append(m_ref)
            v_refs.append(v_ref)
            weight_refs.append(weight_ref)
            weight_test.append(weight[i].copy())
        # op normal update
        if aggregate:
            cls.fn_multi_update(weight_test, grad, m, v,
                                rescale_grad, out=weight_test, **kwargs)
        else:
            cls.fn_update(weight_test[0], grad[0], m[0], v[0],
                          rescale_grad, out=weight_test[0], **kwargs)
        # Compare results
        atol = 1e-4 if aggregate else 1e-5
        rtol = 1e-4 if aggregate else None
        for i in range(num_elem):
            assert_almost_equal(weight_refs[i], weight_test[i], rtol=rtol, atol=atol)
            assert_almost_equal(m_refs[i], m[i], rtol=rtol, atol=atol)
            assert_almost_equal(v_refs[i], v[i], atol=atol)

        # Test 4: Reference normal multi-precision update
        grad_rescale, m_refs, v_refs, weight_refs, weight_fp16_refs = [], [], [], [], []
        for i in range(num_elem):
            grad_rescale.append(rescale_grad * grad_fp16[i].astype('float32'))
            m_ref, v_ref, weight_ref = cls.ref_impl(
                m[i], v[i], weight[i], grad_rescale[i],
                beta1, beta2, lrs[i], etas[i], wds[i], epsilon, clip_gradient)
            m_refs.append(m_ref)
            v_refs.append(v_ref)
            weight_refs.append(weight_ref)
            weight_fp16_refs.append(weight_ref.astype('float16'))
        # op normal multi-precision update
        if aggregate:
            cls.fn_multi_mp_update(weight_fp16, grad_fp16, m, v, weight,
                                   rescale_grad, out=weight_fp16, **kwargs)
        else:
            cls.fn_mp_update(weight_fp16[0], grad_fp16[0], m[0], v[0], weight[0],
                             rescale_grad, out=weight_fp16[0], **kwargs)
        # Compare results
        for i in range(num_elem):
            assert_almost_equal(m_refs[i], m[i], rtol=rtol, atol=atol)
            assert_almost_equal(v_refs[i], v[i], atol=atol)
            assert_almost_equal(weight_refs[i], weight[i], rtol=rtol, atol=atol)
            assert_almost_equal(weight_fp16_refs[i], weight_fp16[i], rtol=1e-3, atol=atol)

    def __call__(self):
        # Testing aggregated Adam update for one element
        self.run_test(1, aggregate=True)
        # Testing Adam update, if num_elem == 0, OR
        #         aggregated Adam update, if num_elem > 0
        for num_elem in reversed(range(6)):
            self.run_test(num_elem+1)

class _AdamWTestHelper(_AdamLikeTestHelper):
    fn_update = mx.nd.contrib.adamw_update
    fn_multi_update = mx.nd.contrib.multi_adamw_update
    fn_mp_update = mx.nd.contrib.mp_adamw_update
    fn_multi_mp_update = mx.nd.contrib.multi_mp_adamw_update
    @staticmethod
    def ref_impl(m, v, weight, grad_rescale, beta1, beta2, lr, eta, wd, epsilon, clip_grad=-1):
        if clip_grad >= 0:
            grad_rescale = mx.nd.clip(grad_rescale, -clip_grad, clip_grad)

        mean_ref = beta1*m + (1.-beta1)*grad_rescale
        v_ref = beta2*v + (1.-beta2)*(grad_rescale**2)
        weight_ref = weight - eta * (lr * mean_ref / (v_ref.sqrt() + epsilon) + weight * wd)
        return mean_ref, v_ref, weight_ref

class _AdaBeliefTestHelper(_AdamLikeTestHelper):
    fn_update = mx.nd.contrib.adabelief_update
    fn_multi_update = mx.nd.contrib.multi_adabelief_update
    fn_mp_update = mx.nd.contrib.mp_adabelief_update
    fn_multi_mp_update = mx.nd.contrib.multi_mp_adabelief_update
    @staticmethod
    def ref_impl(m, v, weight, grad_rescale, beta1, beta2, lr, eta, wd, epsilon, clip_grad=-1):
        grad_rescale += wd * weight
        if clip_grad >= 0:
            grad_rescale = mx.nd.clip(grad_rescale, -clip_grad, clip_grad)

        mean_ref = beta1*m + (1.-beta1)*grad_rescale
        v_ref = beta2*v + (1.-beta2)*((grad_rescale-mean_ref)**2) + epsilon
        weight_ref = weight - eta * (lr * mean_ref / (v_ref.sqrt() + epsilon))
        return mean_ref, v_ref, weight_ref

@xfail_when_nonstandard_decimal_separator
@pytest.mark.serial
def test_adamw():
    _AdamWTestHelper()()

@xfail_when_nonstandard_decimal_separator
@pytest.mark.serial
def test_adabelief():
    _AdaBeliefTestHelper()()
