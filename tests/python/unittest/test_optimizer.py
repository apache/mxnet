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

import numpy as np
import mxnet as mx
import mxnet.lr_scheduler as lr_scheduler
from mxnet import gluon
import unittest
from nose.tools import raises
import math
from mxnet.test_utils import *

def test_learning_rate():
    o1 = mx.optimizer.Optimizer(learning_rate=0.01)
    o1.set_learning_rate(0.2)
    assert o1.learning_rate == 0.2

    lr_s = lr_scheduler.FactorScheduler(step=1)
    o2 = mx.optimizer.Optimizer(lr_scheduler=lr_s, learning_rate=0.3)
    assert o2.learning_rate == 0.3
    o2.lr_scheduler.base_lr = 0.4
    assert o2.learning_rate == 0.4


@raises(UserWarning)
def test_learning_rate_expect_user_warning():
    lr_s = lr_scheduler.FactorScheduler(step=1)
    o = mx.optimizer.Optimizer(lr_scheduler=lr_s, learning_rate=0.3)
    o.set_learning_rate(0.5)


def test_lr_wd_mult():
    data = mx.sym.Variable('data')
    bias = mx.sym.Variable('fc1_bias', lr_mult=1.0)
    fc1 = mx.sym.FullyConnected(data=data, bias=bias, name='fc1', num_hidden=10, lr_mult=0)
    fc2 = mx.sym.FullyConnected(data=fc1, name='fc2', num_hidden=10, wd_mult=0.5)

    mod = mx.mod.Module(symbol=fc2, label_names=None, context=default_context())
    mod.bind(data_shapes=[('data', (5,10))])
    mod.init_params(initializer=mx.init.Uniform(1.0))
    mod.init_optimizer(optimizer_params={'learning_rate': 1.0})
    args1, _ = mod.get_params()
    args1 = {k: v.asnumpy() for k, v in args1.items()}
    mod.forward(mx.io.DataBatch(data=[mx.random.uniform(low=-1.0, high=1.0, shape=(5,10))], label=None), is_train=True)
    mod.backward(mod.get_outputs())
    mod.update()
    args2, _ = mod.get_params()
    args2 = {k: v.asnumpy() for k, v in args2.items()}

    assert mod._optimizer.lr_mult == {'fc1_bias': 1.0, 'fc1_weight': 0.0}
    assert mod._optimizer.wd_mult == {'fc2_bias': 0.5, 'fc2_weight': 0.5, 'fc1_bias': 0.0}
    assert mx.test_utils.almost_equal(args1['fc1_weight'], args2['fc1_weight'], 1e-10)
    assert not mx.test_utils.almost_equal(args1['fc1_bias'], args2['fc1_bias'], 1e-1)
    assert not mx.test_utils.almost_equal(args1['fc2_weight'], args2['fc2_weight'], 1e-1)

def compare_ndarray_tuple(t1, t2, rtol=None, atol=None):
    if t1 is not None and t2 is not None:
        if isinstance(t1, tuple):
            for s1, s2 in zip(t1, t2):
                compare_ndarray_tuple(s1, s2, rtol, atol)
        else:
            assert_almost_equal(t1.asnumpy(), t2.asnumpy(), rtol=rtol, atol=atol)


def compare_optimizer(opt1, opt2, shape, dtype, w_stype='default', g_stype='default'):
    if w_stype == 'default':
        w2 = mx.random.uniform(shape=shape, ctx=default_context(), dtype=dtype)
        w1 = w2.copyto(default_context())
    elif w_stype == 'row_sparse' or w_stype == 'csr':
        w2 = rand_ndarray(shape, w_stype, density=1, dtype=dtype)
        w1 = w2.copyto(default_context()).tostype('default')
    else:
        raise Exception("type not supported yet")
    if g_stype == 'default':
        g2 = mx.random.uniform(shape=shape, ctx=default_context(), dtype=dtype)
        g1 = g2.copyto(default_context())
    elif g_stype == 'row_sparse' or g_stype == 'csr':
        g2 = rand_ndarray(shape, g_stype, dtype=dtype)
        g1 = g2.copyto(default_context()).tostype('default')
    else:
        raise Exception("type not supported yet")

    state1 = opt1.create_state_multi_precision(0, w1)
    state2 = opt2.create_state_multi_precision(0, w2)
    compare_ndarray_tuple(state1, state2)

    opt1.update_multi_precision(0, w1, g1, state1)
    opt2.update_multi_precision(0, w2, g2, state2)
    compare_ndarray_tuple(state1, state2, rtol=1e-4, atol=1e-5)
    assert_almost_equal(w1.asnumpy(), w2.asnumpy(), rtol=1e-4, atol=1e-5)

# SGD

class PySGD(mx.optimizer.Optimizer):
    """python reference implemenation of sgd"""
    def __init__(self, learning_rate=0.01, momentum=0.0, multi_precision=False, **kwargs):
        super(PySGD, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum
        self.multi_precision = multi_precision

    def create_state(self, index, weight):
        """Create additional optimizer state: momentum

        Parameters
        ----------
        weight : NDArray
        The weight data

        """
        momentum = None
        weight_master_copy = None
        do_multi_precision = self.multi_precision and weight.dtype == np.float16
        if do_multi_precision:
            if self.momentum != 0.0:
                momentum = mx.nd.zeros(weight.shape, weight.context, dtype=np.float32)
            weight_master_copy = array(weight, ctx=weight.context, dtype=np.float32)
            return (momentum, weight_master_copy)
        else:
            if self.momentum != 0.0:
                momentum = mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype)
            return momentum

    def create_state_multi_precision(self, index, weight):
        return self.create_state(index, weight)

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
        An unique integer key used to index the parameters

        weight : NDArray
        weight ndarray

        grad : NDArray
        grad ndarray

        state : NDArray or other objects returned by init_state
        The auxiliary state used in optimization.
        """
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        use_multi_precision = isinstance(state, list) or isinstance(state, tuple)

        if not use_multi_precision:
            if self.momentum == 0.0:
                if self.clip_gradient is not None:
                    weight[:] = ((1 - lr*wd)*weight -
                        lr*mx.nd.clip(grad*self.rescale_grad, -self.clip_gradient, self.clip_gradient))
                else:
                    weight[:] = (1 - lr*wd)*weight - lr*self.rescale_grad*grad
            else:
                mom = state
                if self.clip_gradient is not None:
                    mom[:] = (self.momentum*mom - lr*wd*weight -
                        lr*mx.nd.clip(grad*self.rescale_grad, -self.clip_gradient, self.clip_gradient))
                    weight += mom
                else:
                    mom[:] = self.momentum*mom - lr*wd*weight - lr*self.rescale_grad*grad
                    weight += mom
        else:
            grad32 = array(grad, ctx=grad.context, dtype=np.float32)
            mom = state[0]
            weight32 = state[1]
            if self.momentum == 0.0:
                if self.clip_gradient is not None:
                    weight32[:] = ((1 - lr*wd)*weight32 -
                        lr*mx.nd.clip(grad32*self.rescale_grad, -self.clip_gradient, self.clip_gradient))
                else:
                    weight32[:] = (1 - lr*wd)*weight32 - lr*self.rescale_grad*grad32
            else:
                if self.clip_gradient is not None:
                    mom[:] = (self.momentum*mom - lr*wd*weight32 -
                        lr*mx.nd.clip(grad32*self.rescale_grad, -self.clip_gradient, self.clip_gradient))
                    weight32 += mom
                else:
                    mom[:] = self.momentum*mom - lr*wd*weight32 - lr*self.rescale_grad*grad32
                    weight32 += mom
            tmp = weight32.astype(weight.dtype)
            tmp.copyto(weight)

    def update_multi_precision(self, index, weight, grad, state):
        self.update(index, weight, grad, state)

@unittest.skip("Test fails intermittently. Temporarily disabled until fixed. Tracked at https://github.com/apache/incubator-mxnet/issues/9000")
def test_sgd():
    mx.random.seed(0)
    opt1 = PySGD
    opt2 = mx.optimizer.SGD
    shape = (3, 4, 5)
    mom_options = [{}, {'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{}, {'multi_precision': False}, {'multi_precision': True}]
    for dtype in [np.float16, np.float32, np.float64]:
        for mom_option in mom_options:
            for cg_option in cg_options:
                for rg_option in rg_options:
                    for wd_option in wd_options:
                        for mp_option in mp_options:
                            kwarg = {}
                            kwarg.update(mom_option)
                            kwarg.update(cg_option)
                            kwarg.update(rg_option)
                            kwarg.update(wd_option)
                            kwarg.update(mp_option)
                            if (dtype == np.float16 and
                                    ('multi_precision' not in kwarg or
                                        not kwarg['multi_precision'])):
                                continue
                            compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype)
                            # test operator fallback on cpu
                            if (default_context() == mx.cpu()):
                                compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype,
                                                  g_stype='row_sparse')
                                if dtype != np.float16:
                                    compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape[:2],
                                                      dtype, w_stype='csr', g_stype='csr')
    # test optimizer with a big shape
    big_shape = (54686454, 1)
    kwarg = {'momentum': 0.9, 'wd': 0.05}
    compare_optimizer(opt1(**kwarg), opt2(**kwarg), big_shape, np.float32)

class PySparseSGD(mx.optimizer.Optimizer):
    """python reference implemenation of sgd"""
    def __init__(self, learning_rate=0.01, momentum=0.0, **kwargs):
        super(PySparseSGD, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        """Create additional optimizer state: momentum

        Parameters
        ----------
        weight : NDArray
        The weight data

        """
        if self.momentum == 0.0:
            return None
        else:
            return mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype)

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
        An unique integer key used to index the parameters

        weight : NDArray
        weight ndarray

        grad : NDArray
        grad ndarray

        state : NDArray or other objects returned by init_state
        The auxiliary state used in optimization.
        """
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        num_rows = weight.shape[0]
        if self.momentum == 0.0:
            # Update on a per row basis, skip all-zero rows
            for row in range(num_rows):
                grad_row = grad[row].asnumpy()
                all_zeros = mx.test_utils.almost_equal(grad_row, np.zeros_like(grad_row))
                if all_zeros:
                   continue
                if self.clip_gradient is not None:
                    weight[row] = ((1 - lr*wd)*weight[row] -
                        lr*mx.nd.clip(grad[row]*self.rescale_grad,
                                     -self.clip_gradient, self.clip_gradient))
                else:
                    weight[row] = (1 - lr*wd)*weight[row] - lr*self.rescale_grad*grad[row]
        else:
            mom = state
            for row in range(num_rows):
              grad_row = grad[row].asnumpy()
              all_zeros = mx.test_utils.almost_equal(grad_row, np.zeros_like(grad_row))
              if all_zeros:
                  continue
              if self.clip_gradient is not None:
                  mom[row] = (self.momentum*mom[row] - lr*wd*weight[row] -
                      lr*mx.nd.clip(grad[row]*self.rescale_grad, -self.clip_gradient, self.clip_gradient))
                  weight[row] += mom[row]
              else:
                  mom[row] = self.momentum*mom[row] - lr*wd*weight[row] - lr*self.rescale_grad*grad[row]
                  weight[row] += mom[row]

def test_sparse_sgd():
    mx.random.seed(0)
    opt1 = PySparseSGD
    opt2 = mx.optimizer.SGD
    shape = (3, 4, 5)
    mom_options = [{}, {'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{}, {'multi_precision': False}, {'multi_precision': True}]
    for dtype in [np.float32]:
        for mom_option in mom_options:
            for cg_option in cg_options:
                for rg_option in rg_options:
                    for wd_option in wd_options:
                        for mp_option in mp_options:
                            kwarg = {}
                            kwarg.update(mom_option)
                            kwarg.update(cg_option)
                            kwarg.update(rg_option)
                            kwarg.update(wd_option)
                            kwarg.update(mp_option)
                            compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype,
                                              w_stype='row_sparse', g_stype='row_sparse')


def test_std_sparse_sgd():
    mx.random.seed(0)
    opt1 = PySGD
    opt2 = mx.optimizer.SGD
    shape = (3, 4, 5)
    mom_options = [{'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    for dtype in [np.float32]:
        for mom_option in mom_options:
            for cg_option in cg_options:
                for rg_option in rg_options:
                    for wd_option in wd_options:
                        kwarg = {}
                        kwarg.update(mom_option)
                        kwarg.update(cg_option)
                        kwarg.update(rg_option)
                        kwarg.update(wd_option)
                        compare_optimizer(opt1(**kwarg), opt2(lazy_update=False, **kwarg), shape, dtype,
                                          w_stype='row_sparse', g_stype='row_sparse')


class PyNAG(PySGD):
    def __init__(self, **kwargs):
        super(PyNAG, self).__init__(**kwargs)

    def create_state(self, index, weight):
        """Create additional optimizer state: momentum

        Parameters
        ----------
        weight : NDArray
        The weight data

        """
        momentum = None
        weight_master_copy = None
        do_multi_precision = self.multi_precision and weight.dtype == np.float16
        if do_multi_precision:
            if self.momentum != 0.0:
                momentum = mx.nd.zeros(weight.shape, weight.context, dtype=np.float32)
            weight_master_copy = array(weight, ctx=weight.context, dtype=np.float32)
            return (weight_master_copy, momentum)
        else:
            if self.momentum != 0.0:
                momentum = mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype)
            return momentum

    def create_state_multi_precision(self, index, weight):
        return self.create_state(index, weight)

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
        An unique integer key used to index the parameters

        weight : NDArray
        weight ndarray

        grad : NDArray
        grad ndarray

        state : NDArray or other objects returned by init_state
        The auxiliary state used in optimization.
        """
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        use_multi_precision = isinstance(state, list) or isinstance(state, tuple)
        if not use_multi_precision:
            grad = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)
            if self.momentum == 0.0:
                weight[:] += -lr * (grad + wd * weight)
            else:
              mom = state
              mom[:] *= self.momentum
              grad += wd * weight
              mom[:] += grad
              grad[:] += self.momentum * mom
              weight[:] += -lr * grad 
        else:
            grad32 = array(grad, ctx=grad.context, dtype=np.float32)
            grad32 = grad32 * self.rescale_grad
            if self.clip_gradient is not None:
                grad32 = mx.nd.clip(grad32, -self.clip_gradient, self.clip_gradient)
            mom = state[1]
            weight32 = state[0]
            if self.momentum == 0.0:
                weight32[:] += -lr * (grad32 + wd * weight32)
            else:
                mom[:] *= self.momentum
                grad32 += wd * weight32
                mom[:] += grad32
                grad32[:] += self.momentum * mom
                weight32[:] += -lr * grad32
            tmp = weight32.astype(weight.dtype)
            tmp.copyto(weight)


def test_nag():
    mx.random.seed(0)
    opt1 = PyNAG
    opt2 = mx.optimizer.NAG
    shape = (3, 4, 5)
    mom_options = [{}, {'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{}, {'multi_precision': False}, {'multi_precision': True}]
    for dtype in [np.float16, np.float32, np.float64]:
        for mom_option in mom_options:
            for cg_option in cg_options:
                for rg_option in rg_options:
                    for wd_option in wd_options:
                        for mp_option in mp_options:
                            kwarg = {}
                            kwarg.update(mom_option)
                            kwarg.update(cg_option)
                            kwarg.update(rg_option)
                            kwarg.update(wd_option)
                            kwarg.update(mp_option)
                            if (dtype == np.float16 and
                                    ('multi_precision' not in kwarg or
                                        not kwarg['multi_precision'])):
                                continue
                            compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype)



# FTML

class PyFTML(mx.optimizer.Optimizer):
    """python reference implemenation of FTML"""
    def __init__(self, beta1=0.6, beta2=0.999, epsilon=1e-8, **kwargs):
        super(PyFTML, self).__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype), # d_0
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype), # v_0
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype)) # z_0

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, mx.nd. NDArray))
        assert(isinstance(grad, mx.nd.NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        t = self._index_update_count[index]

        grad = grad * self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)
        # get previous states
        prev_d, prev_v, prev_z = state
        # compute states
        v_t = self.beta2 * prev_v + (1 - self.beta2) * mx.nd.square(grad)
        d_t = (1 - pow(self.beta1, t)) / lr * (mx.nd.sqrt(v_t / (1 - pow(self.beta2, t))) + self.epsilon)
        sigma_t = d_t - self.beta1 * prev_d
        z_t = self.beta1 * prev_z + (1 - self.beta1) * grad - sigma_t * weight
        # update weight
        weight[:] = - z_t / d_t
        # update states
        prev_d[:] = d_t
        prev_v[:] = v_t
        prev_z[:] = z_t


def test_ftml():
    mx.random.seed(0)
    opt1 = PyFTML
    opt2 = mx.optimizer.FTML
    shape = (3, 4, 5)
    beta1_options = [{}, {'beta1': 0.5}, {'beta1': 0.7}]
    beta2_options = [{}, {'beta2': 0.8}, {'beta2': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    for dtype in [np.float32]:
        for beta1_option in beta1_options:
            for beta2_option in beta2_options:
                for cg_option in cg_options:
                    for rg_option in rg_options:
                        for wd_option in wd_options:
                            kwarg = {}
                            kwarg.update(beta1_option)
                            kwarg.update(beta2_option)
                            kwarg.update(cg_option)
                            kwarg.update(rg_option)
                            kwarg.update(wd_option)
                            compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype)


# ADAM

class PyAdam(mx.optimizer.Optimizer):
    """python reference implemenation of adam"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 decay_factor=(1 - 1e-8), sparse_update=False, **kwargs):
        super(PyAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.sparse_update = sparse_update

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
        The weight data

        """
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
        An unique integer key used to index the parameters

        weight : NDArray
        weight ndarray

        grad : NDArray
        grad ndarray

        state : NDArray or other objects returned by init_state
        The auxiliary state used in optimization.
        """
        lr = self._get_lr(index)
        self._update_count(index)

        t = self._index_update_count[index]
        mean, variance = state

        wd = self._get_wd(index)
        num_rows = weight.shape[0]
        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1
        for row in range(num_rows):
            # check row slices of all zeros
            all_zeros = mx.test_utils.almost_equal(grad[row].asnumpy(), np.zeros_like(grad[row].asnumpy()))
            # skip zeros during sparse update
            if all_zeros and self.sparse_update:
                continue
            grad[row] = grad[row] * self.rescale_grad + wd * weight[row]
            # clip gradients
            if self.clip_gradient is not None:
                mx.nd.clip(grad[row], -self.clip_gradient, self.clip_gradient, out=grad[row])
            # update mean
            mean[row] *= self.beta1
            mean[row] += grad[row] * (1. - self.beta1)
            # update variance
            variance[row] *= self.beta2
            variance[row] += (1 - self.beta2) * mx.nd.square(grad[row], out=grad[row])
            # update weight
            weight[row] -= lr*mean[row]/(mx.nd.sqrt(variance[row]) + self.epsilon)


def test_adam():
    mx.random.seed(0)
    opt1 = PyAdam
    opt2 = mx.optimizer.Adam
    shape = (3, 4, 5)
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{}, {'multi_precision': False}, {'multi_precision': True}]
    for dtype in [np.float16, np.float32, np.float64]:
        for cg_option in cg_options:
            for rg_option in rg_options:
                for wd_option in wd_options:
                    for mp_option in mp_options:
                        kwarg = {}
                        kwarg.update(cg_option)
                        kwarg.update(rg_option)
                        kwarg.update(wd_option)
                        kwarg.update(mp_option)
                        if (dtype == np.float16 and
                                ('multi_precision' not in kwarg or
                                    not kwarg['multi_precision'])):
                            continue
                        compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype)
                        compare_optimizer(opt1(sparse_update=True, **kwarg), opt2(**kwarg), shape,
                                          dtype, w_stype='row_sparse', g_stype='row_sparse')
                        compare_optimizer(opt1(**kwarg), opt2(lazy_update=False, **kwarg), shape,
                                          dtype, w_stype='row_sparse', g_stype='row_sparse')

# Signum
class PySignum(mx.optimizer.Optimizer):
    """The python reference of Signum optimizer.

    The optimizer updates the weight by:

        rescaled_grad = rescale_grad * clip(grad, clip_gradient) + wd * weight
        state = momentum * state + (1-momentum)*rescaled_grad
        weight = (1 - lr * wd_lh) * weight - lr * sign(state)

    See the original paper at: https://jeremybernste.in/projects/amazon/signum.pdf

    For details of the update algorithm see
    :class:`~mxnet.ndarray.signsgd_update` and :class:`~mxnet.ndarray.signum_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
    wd_lh : float, optitional
       The amount of decoupled weight decay regularization.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9, wd_lh = 0.0, **kwargs):
        super(PySignum, self).__init__(learning_rate = learning_rate, **kwargs)
        self.momentum = momentum
        self.wd_lh = wd_lh

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype, stype=weight.stype)
        return momentum

    def update(self, index, weight, grad, state):
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        if state is not None:
            mom = state
            if self.clip_gradient is not None:
              mom[:] = (self.momentum*mom - (1-self.momentum)*(wd*weight +
                  mx.nd.clip(grad*self.rescale_grad, -self.clip_gradient, self.clip_gradient)))
            else:
              mom[:] = self.momentum*mom - (1-self.momentum)*wd*weight - (1-self.momentum)*self.rescale_grad*grad
            weight[:] = (1 - lr*self.wd_lh)*weight + lr*mx.nd.sign(mom)
        else:
            weight[:] = (1 - lr*(wd+self.wd_lh))*weight - lr*mx.nd.sign(grad)

def test_signum():
    mx.random.seed(0)
    opt1 = PySignum
    opt2 = mx.optimizer.Signum
    shape = (3, 4, 5)
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    wd_lh_options = [{}, {'wd_lh': 0.015}, {'wd_lh': 0.0}]
    mom_options = [{}, {'momentum': 0.9}]
    lr_options = [{'learning_rate': 0.05},{'learning_rate': 0.01}]
    for dtype in [np.float32, np.float64]:
        for cg_option in cg_options:
            for rg_option in rg_options:
                for wd_option in wd_options:
                    for mp_option in wd_lh_options:
                        for lr_option in lr_options:
                            for mom_option in mom_options:
                                kwarg = {}
                                kwarg.update(cg_option)
                                kwarg.update(rg_option)
                                kwarg.update(wd_option)
                                kwarg.update(mp_option)
                                kwarg.update(lr_option)
                                kwarg.update(mom_option)
                                compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype)


# RMSProp
class PyRMSProp(mx.optimizer.Optimizer):
    """RMSProp optimizer of Tieleman & Hinton, 2012,

    For centered=False, the code follows the version in
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf by
    Tieleman & Hinton, 2012

    For centered=True, the code follows the version in
    http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.001.
    gamma1: float, optional
        decay factor of moving average for gradient, gradient^2.
        Default value is set to 0.9.
    gamma2: float, optional
        "momentum" factor.
        Default value if set to 0.9.
        Only used if centered=True
    epsilon : float, optional
        Default value is set to 1e-8.
    centered : boolean, optional
        Use Graves or Tielemans & Hintons version of RMSProp
    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient.
    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    clip_weights : float, optional
        clip weights in range [-clip_weights, clip_weights]

    """
    def __init__(self, learning_rate=0.001, gamma1=0.9, gamma2=0.9,
                 epsilon=1e-8, centered=False, clip_weights=None, **kwargs):
        super(PyRMSProp, self).__init__(learning_rate=learning_rate, **kwargs)
        self.centered = centered
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.epsilon = epsilon
        self.clip_weights = clip_weights

    def create_state(self, index, weight):
        """Create additional optimizer state.

        For centered=False: n
        For centered=True: n, g, delta

        Parameters
        ----------
        weight : NDArray
            The weight data
        """
        if self.centered:
            return (mx.nd.zeros(weight.shape, weight.context),  # n
                    mx.nd.zeros(weight.shape, weight.context),  # g
                    mx.nd.zeros(weight.shape, weight.context))  # delta
        else:
            return (mx.nd.zeros(weight.shape, weight.context), )  # n

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        grad = grad * self.rescale_grad + wd * weight

        if not self.centered:
            (n, ) = state
            if self.clip_gradient is not None:
                grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)
            n[:] = (1 - self.gamma1) * (grad * grad) + self.gamma1 * n
            weight[:] -= lr * grad/(mx.nd.sqrt(n + self.epsilon))

        else:
            n, g, delta = state
            if self.clip_gradient is not None:
                grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)
            n[:] = (1 - self.gamma1) * (grad * grad) + self.gamma1 * n
            g[:] = (1 - self.gamma1) * grad + self.gamma1 * g
            delta[:] = (self.gamma2) * delta - lr * grad/(mx.nd.sqrt(n - g*g + self.epsilon))
            weight[:] += delta

        if self.clip_weights:
             mx.ndarray.clip(weight, -self.clip_weights, self.clip_weights, out=weight)

@unittest.skip("Test fails intermittently. Temporarily disabled until fixed. Tracked at https://github.com/apache/incubator-mxnet/issues/8230")
def test_rms():
    mx.random.seed(0)
    opt1 = PyRMSProp
    opt2 = mx.optimizer.RMSProp
    shape = (3, 4, 5)
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    cw_options = [{}, {'clip_weights': 0.01}]
    center_options = [{}, {'centered': False}, {'centered': True}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{}, {'multi_precision': False}, {'multi_precision': True}]
    for dtype in [np.float16, np.float32]:
        for cw_option in cw_options:
            for cg_option in cg_options:
                for center_option in center_options:
                    for rg_option in rg_options:
                        for wd_option in wd_options:
                            for mp_option in mp_options:
                                kwarg = {}
                                kwarg.update(cw_option)
                                kwarg.update(cg_option)
                                kwarg.update(center_option)
                                kwarg.update(rg_option)
                                kwarg.update(wd_option)
                                kwarg.update(mp_option)
                                if (dtype == np.float16 and
                                        ('multi_precision' not in kwarg or
                                            not kwarg['multi_precision'])):
                                    continue
                                compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype)
                                if (default_context() == mx.cpu()):
                                    compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype, g_stype='row_sparse')

class PyFtrl(mx.optimizer.Optimizer):
    """The Ftrl optimizer.

    Referenced from *Ad Click Prediction: a View from the Trenches*, available at
    http://dl.acm.org/citation.cfm?id=2488200.

    Parameters
    ----------
    lamda1 : float, optional
        L1 regularization coefficient.
    learning_rate : float, optional
        The initial learning rate.
    beta : float, optional
        Per-coordinate learning rate correlation parameter.
    eta :
        .. math::
           \\eta_{t,i} = \\frac{learningrate}{\\beta+\\sqrt{\\sum_{s=1}^tg_{s,i}^t}}
    """

    def __init__(self, lamda1=0.01, learning_rate=0.1, beta=1, sparse_update=False, **kwargs):
        super(PyFtrl, self).__init__(**kwargs)
        self.lamda1 = lamda1
        self.beta = beta
        self.lr = learning_rate
        self.sparse_update = sparse_update

    def create_state(self, index, weight):
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # dn
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # n

    def update(self, index, weight, grad, state):
        self._update_count(index)
        wd = self._get_wd(index)
        lr = self._get_lr(index)
        num_rows = weight.shape[0]

        dn, n = state
        for row in range(num_rows):
            all_zeros = mx.test_utils.almost_equal(grad[row].asnumpy(), np.zeros_like(grad[row].asnumpy()))
            if all_zeros and self.sparse_update:
                continue
            grad[row] = grad[row] * self.rescale_grad
            if self.clip_gradient is not None:
                mx.nd.clip(grad[row], -self.clip_gradient, self.clip_gradient, out=grad[row])

            #update dn, n
            dn[row] += grad[row] - (mx.nd.sqrt(n[row] + grad[row] * grad[row]) - mx.nd.sqrt(n[row])) * weight[row] / lr
            n[row] += grad[row] * grad[row]

            # update weight
            weight[row] = (mx.nd.sign(dn[row]) * self.lamda1 - dn[row]) / \
                          ((self.beta + mx.nd.sqrt(n[row])) / lr + wd) * (mx.nd.abs(dn[row]) > self.lamda1)

def test_ftrl():
    mx.random.seed(0)
    opt1 = PyFtrl
    opt2 = mx.optimizer.Ftrl
    shape = (3, 4, 5)
    kwargs = [{},
              {'clip_gradient': 0.5},
              {'clip_gradient': 0.4, 'rescale_grad': 0.14},
              {'rescale_grad': 0.8},
              {'clip_gradient': 0.5, 'wd': 0.07},
              {'clip_gradient': 0.4, 'rescale_grad': 0.14, 'wd': 0.03},
              {'rescale_grad': 0.8, 'wd': 0.05},
              {'rescale_grad': 0.8, 'wd': 0.05, 'lamda1': 0.01},
              {'clip_gradient': 0.5, 'wd': 0.07, 'lamda1': 1.0}]
    for kwarg in kwargs:
        compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, np.float32)
        compare_optimizer(opt1(sparse_update=True, **kwarg), opt2(**kwarg), shape,
                          np.float32, w_stype='row_sparse', g_stype='row_sparse')

def test_nadam():

    def get_net(num_hidden, flatten=True):
        data = mx.symbol.Variable('data')
        fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128, flatten=flatten)
        act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
        fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64, flatten=flatten)
        act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
        fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=num_hidden, flatten=flatten)
        return fc3
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=5, label_name='label', shuffle=True)
    output = get_net(1)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.L1Loss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=60, optimizer_params={'learning_rate': 0.0005, 'wd': 0.0005},
            initializer=mx.init.Xavier(magnitude=2), eval_metric=mx.metric.Loss(),
            optimizer='nadam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.1


if __name__ == '__main__':
    import nose
    nose.runmodule()
