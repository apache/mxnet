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

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import with_seed


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


@with_seed()
def test_adamw():
    def get_refs(m, v, weight, grad_rescale, beta1, beta2, lr, eta, wd, epsilon, clip_grad=-1):
        if clip_grad >= 0:
            grad_rescale = mx.nd.clip(grad_rescale, -clip_grad, clip_grad)

        mean_ref = beta1*m + (1-beta1)*grad_rescale
        v_ref = beta2*v + (1-beta2)*(grad_rescale**2)
        weight_ref = weight - eta * (lr * mean_ref / (v_ref.sqrt() + epsilon) + weight * wd)
        return mean_ref, v_ref, weight_ref

    def run_adamw_test(nElem=1, aggregate=False):
        aggregate = aggregate or nElem > 1
        rescale_factor = 10
        eta, lr, wd, epsilon = 1, 1, 0.1, 1e-8
        beta1, beta2 = 0.9, 0.999
        clip_gradient = np.random.uniform(rescale_factor, rescale_factor)
        weight, grad, m, v, etas, lrs, wds, weight_ref = [], [], [], [], [], [], [], []
        for i in range(nElem):
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
                mx.nd.contrib.multi_adamw_update(weight, grad, m, v,
                                                 rescaled_grad, out=weight, **kwargs)
            else:
                mx.nd.contrib.adamw_update(weight[0], grad[0], m[0], v[0],
                                           rescaled_grad, out=weight[0], **kwargs)

            # weights should remain unchanged
            for j in range(nElem):
                assert_almost_equal(weight_ref[j], weight[j])


        # Test 2: Same as Test 1 for multi-precision update
        weight_fp16, grad_fp16, weight_fp16_refs = [], [], []
        for i in range(nElem):
            weight_fp16.append(weight[i].astype('float16'))
            grad_fp16.append(grad[i].astype('float16'))
            weight_fp16_refs.append(weight_fp16[i].copy())

        for rescaled_grad in tested_grad:
            if aggregate:
                mx.nd.contrib.multi_mp_adamw_update(weight_fp16, grad_fp16, m, v, weight,
                                                    rescaled_grad, out=weight_fp16, **kwargs)
            else:
                mx.nd.contrib.mp_adamw_update(weight_fp16[0], grad_fp16[0], m[0], v[0], weight[0],
                                              rescaled_grad, out=weight_fp16[0], **kwargs)

            # weights should remain unchanged
            for i in range(nElem):
                assert_almost_equal(weight_ref[i], weight[i])
                assert_almost_equal(weight_fp16_refs[i], weight_fp16[i])


        # Test 3: Reference normal update
        grad_rescale, weight_test, m_refs, v_refs, weight_refs = [], [], [], [], []
        for i in range(nElem):
            grad_rescale.append(rescale_grad * grad[i])
            m_ref, v_ref, weight_ref = get_refs(m[i], v[i], weight[i], grad_rescale[i], beta1, beta2, lrs[i], etas[i], wds[i], epsilon, clip_gradient)
            m_refs.append(m_ref)
            v_refs.append(v_ref)
            weight_refs.append(weight_ref)
            weight_test.append(weight[i].copy())

        # op normal update
        if aggregate:
            mx.nd.contrib.multi_adamw_update(weight_test, grad, m, v,
                                             rescale_grad, out=weight_test, **kwargs)
        else:
            mx.nd.contrib.adamw_update(weight_test[0], grad[0], m[0], v[0],
                                       rescale_grad, out=weight_test[0], **kwargs)

        # Compare results
        atol = 1e-4 if aggregate else 1e-5
        rtol = 1e-4 if aggregate else None
        for i in range(nElem):
            assert_almost_equal(weight_refs[i], weight_test[i], rtol=rtol, atol=atol)
            assert_almost_equal(m_refs[i], m[i], rtol=rtol, atol=atol)
            assert_almost_equal(v_refs[i], v[i], atol=atol)


        # Test 4: Reference normal multi-precision update
        grad_rescale, m_refs, v_refs, weight_refs, weight_fp16_refs = [], [], [], [], []
        for i in range(nElem):
            grad_rescale.append(rescale_grad * grad_fp16[i].astype('float32'))
            m_ref, v_ref, weight_ref = get_refs(m[i], v[i], weight[i], grad_rescale[i], beta1, beta2, lrs[i], etas[i], wds[i], epsilon, clip_gradient)
            m_refs.append(m_ref)
            v_refs.append(v_ref)
            weight_refs.append(weight_ref)
            weight_fp16_refs.append(weight_ref.astype('float16'))

        # op normal multi-precision update
        if aggregate:
            mx.nd.contrib.multi_mp_adamw_update(weight_fp16, grad_fp16, m, v, weight,
                                                rescale_grad, out=weight_fp16, **kwargs)
        else:
            mx.nd.contrib.mp_adamw_update(weight_fp16[0], grad_fp16[0], m[0], v[0], weight[0],
                                          rescale_grad, out=weight_fp16[0], **kwargs)

        # Compare results
        for i in range(nElem):
            assert_almost_equal(m_refs[i], m[i], rtol=rtol, atol=atol)
            assert_almost_equal(v_refs[i], v[i], atol=atol)
            assert_almost_equal(weight_refs[i], weight[i], rtol=rtol, atol=atol)
            assert_almost_equal(weight_fp16_refs[i], weight_fp16[i], rtol=1e-3, atol=atol)

    # Testing aggregated Adam update for one element
    run_adamw_test(1, aggregate=True)

    # Testing Adam update, if nElem = 0, OR
    #         aggregated Adam update, if nElem > 0
    for nElem in range(6):
        run_adamw_test(nElem+1)

if __name__ == '__main__':
    import nose
    nose.runmodule()
