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

import os
import sys
import tempfile
import math
import numpy as np
import mxnet as mx

from mxnet.test_utils import rand_ndarray, assert_almost_equal, rand_coord_2d, default_device, check_symbolic_forward, create_2d_tensor
from mxnet.util import TemporaryDirectory
from mxnet import gluon, nd
from common import with_seed
import pytest


# dimension constants
MEDIUM_X = 10000
VLARGE_X = 4300000000
LARGE_X = 100000000
SMALL_X = 100
SMALL_Y = 50
LARGE_SIZE = LARGE_X * SMALL_Y
LARGE_TENSOR_SHAPE = 2**32
RNN_LARGE_TENSOR = 2**28


@pytest.mark.timeout(0)
def test_nn():
    def check_gluon_embedding():
        m = gluon.nn.Embedding(SMALL_Y, MEDIUM_X)
        m.initialize()
        a = nd.zeros((MEDIUM_X, SMALL_Y))
        b = m(a)
        assert b.shape == (MEDIUM_X, SMALL_Y, MEDIUM_X)
        assert b.asnumpy().size == LARGE_SIZE

    def check_fully_connected():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.ones(shape=(SMALL_Y, SMALL_Y))
        c = nd.ones(shape=(b.shape[0],))

        # w/o bias
        res = nd.FullyConnected(a, b, num_hidden=b.shape[0], no_bias=True)
        assert np.sum(res[-1].asnumpy() == a.shape[1]) == b.shape[0]

        # w/ bias
        res = nd.FullyConnected(a, b, c, num_hidden=b.shape[0], no_bias=False)
        assert np.sum(res[-1].asnumpy() == a.shape[1] + 1) == b.shape[0]

    def check_dense():
        data = mx.nd.ones(shape=(50*1000*1000, 100))
        linear = gluon.nn.Dense(100)
        linear.initialize()
        res = linear(data)
        assert res.shape == (50000000, 100)

    def check_softmax():
        input_data = mx.nd.ones((SMALL_Y, LARGE_X))
        for axis in [0, 1]:
            true_output = np.full((SMALL_Y, LARGE_X), (1 / input_data.shape[axis]))
            output = nd.softmax(input_data, axis=axis)
            assert_almost_equal(output.asnumpy(), true_output, rtol=1e-5, atol=1e-5)

    def check_softmax_cross_entropy():
        # dtype of input data, mxnet cross entropy set explicitly to float64
        # numpy implicitly takes care of double precision
        batch_size = SMALL_Y
        num_labels = LARGE_X
        input_data = mx.nd.ones((batch_size, num_labels), dtype="float64")
        input_label = mx.nd.zeros((batch_size,), dtype="float64")
        true_softmax = np.full((batch_size, num_labels), (1 / num_labels))
        # use 1/batch_size when softmax axis=0
        # here 1/num_labels since softmax_cross_entropy uses default axis
        # by default axis=1
        np_one_hot_label = np.zeros((batch_size, num_labels))
        np_one_hot_label[:, 0] = 1
        true_softmax_cross_entropy = np.sum(-np.log(true_softmax) *
                                            np_one_hot_label)
        mx_softmax_cross_entropy = mx.nd.softmax_cross_entropy(input_data,
                                                               input_label,
                                                               dtype="float64")
        assert_almost_equal(mx_softmax_cross_entropy.asnumpy(),
                            true_softmax_cross_entropy, rtol=1e-3, atol=1e-5)

    def check_softmax_activation():
        data = nd.random_normal(shape=(2**29, 2, 2, 2))
        out = nd.random_normal(shape=(2**29, 2, 2, 2))

        res = nd.SoftmaxActivation(data=data, out=out)

        assert res.shape[0] == 536870912
        assert res.shape[1] == 2
        assert res.shape[2] == 2
        assert res.shape[3] == 2

    def np_softmax(x, axis=-1, temperature=1.0):
        x = x - np.max(x, axis=axis, keepdims=True)
        x = np.exp(x/temperature)
        x /= np.sum(x, axis=axis, keepdims=True)
        return x

    @pytest.mark.skip(reason="log_softmax flaky, tracked at "
                      "https://github.com/apache/mxnet/issues/17397")
    def check_log_softmax():
        ndim = 2
        shape = (SMALL_Y, LARGE_X)
        axis = np.random.randint(0, ndim)
        data = np.random.uniform(-2, 2, size=shape)
        sym = mx.sym.log_softmax(axis=axis-ndim)
        check_symbolic_forward(sym, [data], [np.log(np_softmax(data, axis=axis)+1e-20)])

    # TODO: correctness of prelu (currently flaky)
    def check_leaky_relu():
        a = -1*mx.nd.ones((LARGE_X, SMALL_Y))

        def check_leaky():
            res = mx.nd.LeakyReLU(a, act_type="leaky", slope=0.3)
            assert_almost_equal(res[-1][-1].asnumpy(), 0.3*a[-1][-1].asnumpy(), atol=1e-3, rtol=1e-3)

        def check_elu():
            res = mx.nd.LeakyReLU(a, act_type="elu", slope=0.3)
            assert_almost_equal(res[-1][-1].asnumpy(), 0.3*(np.exp(a[-1][-1].asnumpy())-1), atol=1e-3, rtol=1e-3)

        def check_selu():
            lam = 1.0507009873554804934193349852946
            alpha = 1.6732632423543772848170429916717
            res = mx.nd.LeakyReLU(a, act_type="selu")
            assert_almost_equal(res[-1][-1].asnumpy(), (lam * alpha * (np.exp(a[-1][-1].asnumpy())-1)), atol=1e-3, rtol=1e-3)

        def check_rrelu():
            lower = 0.125
            upper = 0.333999991
            res = mx.nd.LeakyReLU(a, act_type="rrelu")
            assert_almost_equal(res[0][-1][-1].asnumpy(), (lower + upper) / 2 * a[-1][-1].asnumpy(), atol=1e-3, rtol=1e-3)

        check_leaky()
        check_elu()
        check_selu()
        check_rrelu()

    def check_pooling():
        a = mx.nd.ones((MEDIUM_X, 200, SMALL_Y, SMALL_Y))

        def check_avg_pooling():
            res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='avg')
            assert_almost_equal(res[-1][-1][-1][-1].asnumpy(), 1.0000001, atol=1e-3, rtol=1e-3)
            assert res.shape[-1] == SMALL_Y - 5 + 1

        def check_max_pooling():
            res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='max')
            assert_almost_equal(res[-1][-1][-1][-1].asnumpy(), 1., atol=1e-3, rtol=1e-3)
            assert res.shape[-1] == SMALL_Y - 5 + 1

        def check_sum_pooling():
            res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='sum')
            assert_almost_equal(res[-1][-1][-1][-1].asnumpy(), 25, atol=1e-3, rtol=1e-3)
            assert res.shape[-1] == SMALL_Y - 5 + 1

        def check_lp_pooling():
            res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='lp', p_value=2)
            assert_almost_equal(res[-1][-1][-1][-1].asnumpy(), 5., atol=1e-3, rtol=1e-3)
            assert res.shape[-1] == SMALL_Y - 5 + 1

            res = mx.nd.Pooling(a, kernel=(5, 5), pool_type='lp', p_value=1)
            assert_almost_equal(res[-1][-1][-1][-1].asnumpy(), 25., atol=1e-3, rtol=1e-3)
            assert res.shape[-1] == SMALL_Y - 5 + 1

        check_avg_pooling()
        check_max_pooling()
        check_sum_pooling()
        check_lp_pooling()

    def check_layer_norm():
        dtype = np.float32
        forward_check_eps = 1E-3
        axis = 1
        eps = 1E-5
        in_shape = (LARGE_X, SMALL_Y)
        ctx = mx.cpu()

        def npy_layer_norm(data, gamma, beta, axis=1, eps=1E-5):
            if axis < 0:
                axis += data.ndim
            broadcast_shape = [1 for _ in range(data.ndim)]
            broadcast_shape[axis] = data.shape[axis]
            mean = data.mean(axis=axis, keepdims=True).astype(dtype)
            var = data.var(axis=axis, keepdims=True).astype(dtype)
            std = np.sqrt(var + dtype(eps)).astype(dtype)
            out = np.reshape(gamma, broadcast_shape) * (data - mean) / std + \
                  np.reshape(beta, broadcast_shape)
            return out
        data = np.random.normal(0, 1, in_shape).astype(dtype)
        gamma = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
        beta = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
        data_s = mx.symbol.Variable('data')
        gamma_s = mx.symbol.Variable('gamma')
        beta_s = mx.symbol.Variable('beta')
        out_s = mx.symbol.LayerNorm(data=data_s, gamma=gamma_s, beta=beta_s,
                                    axis=axis, eps=eps)
        exe = out_s._simple_bind(ctx, data=in_shape)
        exe.arg_dict['data'][:] = data
        exe.arg_dict['gamma'][:] = gamma
        exe.arg_dict['beta'][:] = beta
        out_nd = exe.forward()[0]
        out = npy_layer_norm(data, gamma, beta, axis, eps)
        assert_almost_equal(out, out_nd.asnumpy(), forward_check_eps,
                            forward_check_eps)

    # TODO: correctness of dropout
    # currently only test for dropout to work
    # since testing for correctness involves flakiness issue #14288
    def check_dropout():
        shape = (LARGE_X, SMALL_Y)
        x = mx.sym.var('data')
        y = mx.sym.Dropout(x, p=1, cudnn_off=True)
        exe = y._simple_bind(ctx=default_device(), data=shape)
        exe.arg_arrays[0][:] = 1
        out = exe.forward(is_train=True)
        nd.waitall()
        assert out[0].shape == shape

    def check_activation():
        x = mx.nd.ones((LARGE_X, SMALL_Y))
        check_x = -2
        x[-1, -1] = check_x
        # Hyperbolic tangent (tanh)
        # y = (exp(x)-exp(-x))/(exp(x)+exp(-x))
        y = mx.nd.Activation(x, act_type="tanh")
        tanh_x = ((np.exp(check_x)-np.exp(-check_x))/(np.exp(check_x)+np.exp(-check_x)))
        assert y[-1][-1] == np.float32(tanh_x)
        # Recitified Linear Unit (relu)
        # y = max(x,0)
        y = mx.nd.Activation(x, act_type="relu")
        assert y[-1][-1] == 0
        # Sigmoid
        # y = x/(1+abs(x))
        y = mx.nd.Activation(x, act_type="sigmoid")
        sigmoid_x = (1/(1+math.exp(-check_x)))
        assert_almost_equal(y[-1][-1].asnumpy(), np.float32(sigmoid_x), atol=1e-3, rtol=1e-3)
        # Soft Sign
        # y = 1/(1+exp(-x))
        y = mx.nd.Activation(x, act_type="softsign")
        softsign_x = (check_x/(1+abs(check_x)))
        assert y[-1][-1] == np.float32(softsign_x)


    # TODO: correctness of batchnorm
    # in future, we could test if mean, var of output
    # matches target output's mean, var
    def check_batchnorm():
        def get_np_mean_var(data, running_mean, running_var, eps, use_global_status=True):
            if not use_global_status:
                # train mode, calculate the real mean and var
                mean = np.mean(data, axis=(0, 2, 3))
                mean_broad = np.expand_dims(mean, axis=0)
                mean_broad = np.expand_dims(mean_broad, axis=2)
                mean_broad = np.expand_dims(mean_broad, axis=3)
                mean_broad = np.broadcast_to(mean_broad, data.shape)
                var = np.square(data - mean_broad)
                var = np.mean(var, axis=(0, 2, 3))
            else:
                # inference mode, use running_mean and running_var instead
                mean = np.full((data.shape[1],), running_mean)
                var = np.full((data.shape[1],), running_var)
            # calculate the inverse of standard variance
            invstdvar = 1. / np.sqrt(var + eps)
            return mean, invstdvar
        # Here use 4D input to cover dnnl BN and non-dnnl BN
        shape = (1, 2, LARGE_X, SMALL_Y)
        axis = 1  # default
        eps = 1e-3
        nch = shape[axis]
        data = mx.nd.ones(shape=shape)
        bn_gamma = mx.nd.random.uniform(shape=(nch,))
        bn_beta = mx.nd.random.uniform(shape=(nch,))
        bn_running_mean = mx.nd.zeros(nch)
        bn_running_var = mx.nd.ones(nch)
        output = mx.nd.BatchNorm(data, bn_gamma, bn_beta,
                                 bn_running_mean, bn_running_var, output_mean_var=True)
        assert output[0].shape == shape
        mean, invstdvar = output[1], output[2]
        np_mean, np_invstdvar = get_np_mean_var(data.asnumpy(), bn_running_mean.asnumpy(), bn_running_var.asnumpy(),
                                                eps, use_global_status=True)
        assert_almost_equal(mean.asnumpy(), np_mean)
        assert_almost_equal(invstdvar.asnumpy(), np_invstdvar)

    def check_relu():
        def frelu(x):
            return np.maximum(x, 0.0)

        def frelu_grad(x):
            return 1.0 * (x > 0.0)
        shape = (SMALL_Y, LARGE_X)
        x = mx.symbol.Variable("x")
        y = mx.sym.relu(x)
        xa = np.random.uniform(low=-1.0, high=1.0, size=shape)
        eps = 1e-4
        xa[abs(xa) < eps] = 1.0
        ya = frelu(xa)
        ga = frelu_grad(xa)
        check_symbolic_forward(y, [xa], [ya])

    def check_sigmoid():
        def fsigmoid(a):
            return np.divide(1.0, (1.0 + np.exp(-a)))
        shape = (SMALL_Y, LARGE_X)
        x = mx.symbol.Variable("x")
        y = mx.sym.sigmoid(x)
        xa = np.random.uniform(low=-1.0, high=1.0, size=shape)
        ya = fsigmoid(xa)
        check_symbolic_forward(y, [xa], [ya])

    def check_l2_normalization():
        x = nd.ones((2, LARGE_X*2))
        x[0] = 3
        x[1] = 4
        # Channel Mode
        z = x.reshape(1, 2, LARGE_X*2)
        y = nd.L2Normalization(z, mode='channel')
        assert y[0][0][0] == 0.6
        assert y[0][0][-1] == 0.6
        assert y[0][1][0] == 0.8
        assert y[0][1][-1] == 0.8
        # Instance Mode
        z = x.T
        y = nd.L2Normalization(z, mode='instance')
        assert y[0][0] == 0.6
        assert y[0][1] == 0.8
        assert y[-1][0] == 0.6
        assert y[-1][1] == 0.8
        # Spatial Mode
        z = z.reshape(1, 200000000, 2)
        y = nd.L2Normalization(z, mode='spatial')
        assert y[0][0][0] == 0.6
        assert y[0][0][1] == 0.8
        assert y[0][-1][0] == 0.6
        assert y[0][-1][1] == 0.8

    def check_instance_norm():
        dtype = np.float32
        forward_check_eps = 1E-3
        axis = -1
        eps = 1E-5
        in_shape = (LARGE_X, 1, SMALL_Y)
        ctx = mx.cpu()

        # Implementation of instance normalization using numpy
        def npy_instance_norm(data, gamma, beta, axis, eps=1E-5):
            if axis < 0:
                axis += data.ndim
            broadcast_shape = [1 for _ in range(data.ndim)]
            broadcast_shape[axis] = data.shape[axis]
            mean = data.mean(axis=axis, keepdims=True).astype(dtype)
            var = data.var(axis=axis, keepdims=True).astype(dtype)
            std = np.sqrt(var + dtype(eps)).astype(dtype)
            out = gamma * (data - mean) / std + \
                  beta
            return out
        data = np.random.normal(0, 1, in_shape).astype(dtype)
        gamma = np.random.normal(0, 1, (1,)).astype(dtype)
        beta = np.random.normal(0, 1, (1,)).astype(dtype)
        data_s = mx.symbol.Variable('data')
        gamma_s = mx.symbol.Variable('gamma')
        beta_s = mx.symbol.Variable('beta')
        out_s = mx.symbol.InstanceNorm(data=data_s, gamma=gamma_s, beta=beta_s,
                                       eps=eps)
        exe = out_s._simple_bind(ctx, data=in_shape)
        exe.arg_dict['data'][:] = data
        exe.arg_dict['gamma'][:] = gamma
        exe.arg_dict['beta'][:] = beta
        out_nd = exe.forward()[0]
        # Calls implementation of instance norm in numpy and compares the output
        out = npy_instance_norm(data, gamma, beta, axis, eps)
        assert_almost_equal(out, out_nd.asnumpy(), forward_check_eps,
                            forward_check_eps)

    def check_col2im():
        data = nd.random_normal(shape=(1, 2**30, 4))
        output_size = (2, 2, 1)
        kernel = (1, 1, 1)

        res = nd.col2im(data=data, output_size=output_size, kernel=kernel)

        assert res.shape[0] == 1
        assert res.shape[1] == 1073741824
        assert res.shape[2] == 2
        assert res.shape[3] == 2
        assert res.shape[4] == 1

    def check_embedding():
        data = nd.random_normal(shape=(LARGE_TENSOR_SHAPE, 1))
        weight = nd.random_normal(shape=(LARGE_TENSOR_SHAPE, 1))
        input_dim = LARGE_TENSOR_SHAPE
        output_dim = 1

        out = nd.Embedding(data=data, weight=weight, input_dim=input_dim, output_dim=output_dim)

        assert out.shape[0] == LARGE_TENSOR_SHAPE
        assert out.shape[1] == 1

    def check_spatial_transformer():
        data = nd.random_normal(shape=(2, 2**29, 1, 6))
        loc = nd.random_normal(shape=(2, 6))
        transform_type = 'affine'
        sampler_type = 'bilinear'
        target_shape = (2, 6)

        res = nd.SpatialTransformer(data=data, loc=loc, transform_type=transform_type,
                                    sampler_type=sampler_type, target_shape=target_shape)

        assert res.shape[0] == 2
        assert res.shape[1] == 536870912
        assert res.shape[2] == 2
        assert res.shape[3] == 6

    def check_ravel():
        data = nd.random_normal(shape=(2, LARGE_TENSOR_SHAPE))
        shape = (2, 10)

        out = nd.ravel_multi_index(data=data, shape=shape)

        assert out.shape[0] == LARGE_TENSOR_SHAPE

    def check_cumsum():
        a = nd.ones((LARGE_X, SMALL_Y))
        axis = 1

        res = nd.cumsum(a=a, axis=axis)

        assert res.shape[0] == LARGE_X
        assert res.shape[1] == SMALL_Y
        assert res[0][SMALL_Y - 1] == 50.

    def check_multi_lars():
        lrs = nd.random_normal(shape=(LARGE_TENSOR_SHAPE + 1, 1))
        weights_sum_sq = nd.random_normal(shape=(LARGE_TENSOR_SHAPE + 1, 1))
        grads_sum_sq = nd.random_normal(shape=(LARGE_TENSOR_SHAPE + 1, 1))
        wds = nd.random_normal(shape=(LARGE_TENSOR_SHAPE + 1, 1))
        eta = .1
        eps = .9

        out = nd.multi_lars(lrs=lrs, weights_sum_sq=weights_sum_sq, grads_sum_sq=grads_sum_sq,
                            wds=wds, eta=eta, eps=eps)

        assert out.shape[0] == LARGE_TENSOR_SHAPE + 1
        assert out.shape[1] == 1

        # Trigger lazy evaluation of the output NDArray and ensure that it has been filled
        assert type(out[0, 0].asscalar()).__name__ == 'float32'

    def check_rnn():
        data = nd.random_normal(shape=(RNN_LARGE_TENSOR, 4, 4))
        parameters_relu_tanh = nd.random_normal(shape=(7,))
        parameters_lstm = nd.random_normal(shape=(28,))
        parameters_gru = nd.random_normal(shape=(21,))
        state = nd.random_normal(shape=(1, 4, 1))
        state_cell = nd.random_normal(shape=(1, 4, 1))
        mode_relu = 'rnn_relu'
        mode_tanh = 'rnn_tanh'
        mode_lstm = 'lstm'
        mode_gru = 'gru'
        state_size = 1
        num_layers = 1

        out_relu = nd.RNN(data=data, parameters=parameters_relu_tanh, state=state, mode=mode_relu,
                          state_size=state_size, num_layers=num_layers)

        out_tanh = nd.RNN(data=data, parameters=parameters_relu_tanh, state=state, mode=mode_tanh,
                          state_size=state_size, num_layers=num_layers)

        out_lstm = nd.RNN(data=data, parameters=parameters_lstm, state=state, mode=mode_lstm,
                          state_cell=state_cell, state_size=state_size, num_layers=num_layers)

        out_gru = nd.RNN(data=data, parameters=parameters_gru, state=state, mode=mode_gru,
                         state_size=state_size, num_layers=num_layers)

        for out in [out_relu, out_tanh, out_lstm, out_gru]:
            assert out.shape[0] == RNN_LARGE_TENSOR
            assert out.shape[1] == 4
            assert out.shape[2] == 1

            assert type(out[0, 0, 0].asscalar()).__name__ == 'float32'

    check_gluon_embedding()
    check_fully_connected()
    check_dense()
    check_softmax()
    check_softmax_cross_entropy()
    check_softmax_activation()
    check_log_softmax()
    check_leaky_relu()
    check_pooling()
    check_layer_norm()
    check_dropout()
    check_activation()
    check_batchnorm()
    check_relu()
    check_sigmoid()
    check_l2_normalization()
    check_instance_norm()
    check_col2im()
    check_embedding()
    check_spatial_transformer()
    check_ravel()
    check_cumsum()
    check_multi_lars()
    check_rnn()


@pytest.mark.timeout(0)
def test_tensor():
    def check_ndarray_zeros():
        a = nd.zeros(shape=(LARGE_X, SMALL_Y))
        assert a[-1][0] == 0
        assert a.shape == (LARGE_X, SMALL_Y)
        assert a.size == LARGE_SIZE

    def check_ndarray_ones():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        assert a[-1][0] == 1
        assert nd.sum(a).asnumpy() == LARGE_SIZE

    @with_seed()
    def check_ndarray_random_uniform():
        a = nd.random.uniform(shape=(LARGE_X, SMALL_Y))
        assert a[-1][0] != 0

    @pytest.mark.skip(reason="Randint flaky, tracked at "
                      "https://github.com/apache/mxnet/issues/16172")
    @with_seed()
    def check_ndarray_random_randint():
        a = nd.random.randint(100, 10000, shape=(LARGE_X, SMALL_Y))
        assert a.shape == (LARGE_X, SMALL_Y)
        # check if randint can generate value greater than 2**32 (large)
        low_large_value = 2**32
        high_large_value = 2**34
        a = nd.random.randint(low_large_value, high_large_value, dtype=np.int64)
        low = mx.nd.array([low_large_value], dtype='int64')
        high = mx.nd.array([high_large_value], dtype='int64')
        assert a >= low and a < high
        assert a[-1][0].dtype == np.int64

    @with_seed()
    def check_ndarray_random_exponential():
        scale_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        a = nd.random.exponential(scale=scale_array, shape=(SMALL_X, SMALL_Y))
        assert a[-1][0][0][0] >= 0
        assert a.shape == (MEDIUM_X, SMALL_X, SMALL_X, SMALL_Y)

    @with_seed()
    def check_ndarray_random_gamma():
        alpha_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        beta_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        a = nd.random.gamma(alpha=alpha_array, beta=beta_array,
                            shape=(SMALL_X, SMALL_Y))
        assert a[-1][0][0][0] >= 0
        assert a.shape == (MEDIUM_X, SMALL_X, SMALL_X, SMALL_Y)

    @with_seed()
    def check_ndarray_random_multinomial():
        # test 1 shape dimension
        probs = nd.random.uniform(shape=(LARGE_X, SMALL_Y))
        a = nd.random.multinomial(probs)
        assert a[-1] >= 0
        assert a.shape == (LARGE_X,)
        # test for NDArray multi-dimension shape
        a = nd.random.multinomial(probs, shape=(2, SMALL_Y))
        assert a[-1][0][0] >= 0
        assert a.shape == (LARGE_X, 2, SMALL_Y)
        # test log_likelihood output shape
        a = nd.random.multinomial(probs, shape=(2, SMALL_Y), get_prob=True)
        assert a[0][0][0][0] >= 0
        assert a[0].shape == (LARGE_X, 2, SMALL_Y) and a[0].shape == a[1].shape

    @with_seed()
    def check_ndarray_random_generalized_negative_binomial():
        alpha_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        mu_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        a = nd.random.generalized_negative_binomial(mu=mu_array, alpha=alpha_array,
                                                    shape=(SMALL_X, SMALL_Y))
        assert a[-1][0][0][0] >= 0
        assert a.shape == (MEDIUM_X, SMALL_X, SMALL_X, SMALL_Y)

    @with_seed()
    def check_ndarray_random_negative_binomial():
        k_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        p_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        a = nd.random.negative_binomial(k=k_array, p=p_array,
                                        shape=(SMALL_X, SMALL_Y))
        assert a[-1][0][0][0] >= 0
        assert a.shape == (MEDIUM_X, SMALL_X, SMALL_X, SMALL_Y)

    @with_seed()
    def check_ndarray_random_normal():
        scale_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        loc_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        a = nd.random.normal(loc=loc_array, scale=scale_array,
                             shape=(SMALL_X, SMALL_Y))
        assert a.shape == (MEDIUM_X, SMALL_X, SMALL_X, SMALL_Y)

    @with_seed()
    def check_ndarray_random_poisson():
        lambda_array = nd.random.uniform(shape=(MEDIUM_X, SMALL_X))
        a = nd.random.poisson(lam=lambda_array, shape=(SMALL_X, SMALL_Y))
        assert a[-1][0][0][0] >= 0
        assert a.shape == (MEDIUM_X, SMALL_X, SMALL_X, SMALL_Y)

    @with_seed()
    def check_ndarray_random_randn():
        a = nd.random.randn(LARGE_X, SMALL_Y)
        assert a.shape == (LARGE_X, SMALL_Y)
        # TODO: Once PR #15772 for randn ndarray dtype for loc,scale param merged
        # Add check for (x,y,m,n) where x,y shape of loc,scale and m,n input shape

    @with_seed()
    def check_ndarray_random_shuffle():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        a[-1] = 3  # assign 3 to entire last row
        a = nd.random.shuffle(a)
        # slice first column from shuffled array
        # pass LARGE_X values to numpy instead of LARGE_X*SMALL_Y
        # could have assigned to last column (so as to pass SMALL_Y)
        # but shuffle operation is performed along first axis
        unique_a = np.unique(a[:, 0].asnumpy())
        assert len(unique_a) == 2  # only 2 unique values
        assert unique_a[0] == 1  # first unique value is 1
        assert unique_a[1] == 3  # second unique value is 3
        assert a.shape == (LARGE_X, SMALL_Y)

    def check_ndarray_empty():
        a = nd.empty((LARGE_X, SMALL_Y))
        assert a.shape == (LARGE_X, SMALL_Y)

    def check_zeros_like():
        a = nd.array(np.ones((SMALL_Y, LARGE_X)))
        b = nd.zeros_like(a)
        assert b[-1][-1] == 0
        assert b.shape == a.shape

    def check_ones_like():
        a = nd.array(np.zeros((SMALL_Y, LARGE_X)))
        b = nd.ones_like(a)
        assert b[-1][-1] == 1
        assert b.shape == a.shape

    def check_broadcast():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.arange(0, LARGE_X).reshape(LARGE_X, 1)
        res = nd.broadcast_to(b, shape=(b.shape[0], SMALL_Y))
        assert np.sum(res[-1].asnumpy() == LARGE_X) == res.shape[1]
        res = mx.nd.broadcast_like(b, a)
        assert np.sum(res[-1].asnumpy() == LARGE_X) == a.shape[1]

    def check_clip():
        a = nd.arange(0, LARGE_X * SMALL_Y).reshape(LARGE_X, SMALL_Y)
        res = nd.clip(a, a_min=100, a_max=1000)
        assert np.sum(res[-1].asnumpy() == 1000) == a.shape[1]

    def check_split():
        a = nd.arange(0, LARGE_X * SMALL_Y).reshape(LARGE_X, SMALL_Y)
        outs = nd.split(a, num_outputs=SMALL_Y, axis=1)
        result = sum(1 for i, v in enumerate(outs) if i == v[0].asnumpy())
        assert result == a.shape[1]

    def check_tile():
        a = nd.arange(0, LARGE_X).reshape(LARGE_X, 1)
        b = nd.tile(a, reps=(1, SMALL_Y))
        assert np.sum(b[-1].asnumpy() == LARGE_X) == b.shape[1]

    def check_take():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        idx = nd.arange(LARGE_X - 1000, LARGE_X)
        res = nd.take(a, idx)
        assert np.sum(res[-1].asnumpy() == 1) == res.shape[1]

    def check_slice():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        res = nd.slice(a, begin=(LARGE_X-1000, 1), end=(LARGE_X, SMALL_Y))
        assert np.sum(res[-1].asnumpy() == 1) == res.shape[1]

    def check_slice_assign():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        a[LARGE_X-1:LARGE_X] = 1000
        assert np.sum(a[-1].asnumpy() == 1000) == a.shape[1]

    def check_slice_like():
        a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        b = nd.array(np.ones((SMALL_Y//2, LARGE_X//2)))
        c = nd.slice_like(a, b)
        d = nd.slice_like(a, b, axes=(0))
        e = nd.slice_like(a, b, axes=(-1))
        assert c.shape == b.shape
        assert d.shape[0] == b.shape[0]
        assert e.shape[-1] == b.shape[-1]
        assert c[0][-1] == 0
        assert d[-1][0] == (SMALL_Y//2-1)
        assert e[-1][-1] == (SMALL_Y-1)

    def check_slice_axis():
        a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        c = nd.slice_axis(a, axis=0, begin=0, end=SMALL_Y//2)
        d = nd.slice_axis(a, axis=1, begin=0, end=LARGE_X//2)
        assert c.shape[0] == a.shape[0]//2
        assert d.shape[1] == a.shape[1]//2
        assert c[-1][0] == (SMALL_Y//2-1)
        assert d[-1][-1] == (SMALL_Y-1)

    def check_expand_dims():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        res = nd.expand_dims(a, axis=1)
        res.wait_to_read()
        assert a[0][0][0] == 1
        assert res.shape == (a.shape[0], 1, a.shape[1])

    def check_squeeze():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        data = nd.expand_dims(a, axis=1)
        res = nd.squeeze(data)
        assert res.shape == a.shape

    def check_broadcast_div():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.ones(shape=(LARGE_X, 1)) * 2
        res = a / b
        assert np.sum(res[-1].asnumpy() == 0.5) == a.shape[1]

    def check_where():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.arange(0, LARGE_X * SMALL_Y).reshape(LARGE_X, SMALL_Y)
        res = nd.where(b > 100, a, b)
        assert np.sum(res[-1].asnumpy() == 1) == b.shape[1]
        csr_cond = nd.sparse.cast_storage(b < 10, 'csr')
        res = nd.sparse.where(csr_cond, a, b)
        assert np.sum(res[0].asnumpy() == 1) == 10

    def check_pick():
        a = mx.nd.ones(shape=(256 * 35, 1024 * 1024))
        b = mx.nd.ones(shape=(256 * 35, ))
        res = mx.nd.pick(a, b)
        assert res.shape == b.shape

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, "
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_depthtospace():
        def numpy_depth_to_space(x, blocksize):
            b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h,
                             w])
            tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
            y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize,
                           w * blocksize])
            return y

        shape_inp = (LARGE_X, 8, 4, 2)
        data = rand_ndarray(shape_inp, 'default')
        data_np = data.asnumpy()
        expected = numpy_depth_to_space(data_np, 2)
        output = mx.nd.depth_to_space(data, 2)
        assert_almost_equal(output.asnumpy(), expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, "
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_spacetodepth():
        def numpy_space_to_depth(x, blocksize):
            b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            tmp = np.reshape(x, [b, c, h // blocksize, blocksize, w // blocksize,
                             blocksize])
            tmp = np.transpose(tmp, [0, 3, 5, 1, 2, 4])
            y = np.reshape(tmp, [b, c * (blocksize**2), h // blocksize,
                           w // blocksize])
            return y

        shape_inp = (LARGE_X, 2, 8, 4)
        data = rand_ndarray(shape_inp, 'default')
        data_np = data.asnumpy()
        expected = numpy_space_to_depth(data_np, 2)
        output = mx.nd.space_to_depth(data, 2)
        assert_almost_equal(output.asnumpy(), expected, atol=1e-3, rtol=1e-3)

    @with_seed()
    def check_diag():
        a_np = np.random.random((LARGE_X, SMALL_Y)).astype(np.float32)
        a = mx.nd.array(a_np)

        # k == 0
        r = mx.nd.diag(a)
        assert_almost_equal(r.asnumpy(), np.diag(a_np))

        # k == 1
        k = 1
        r = mx.nd.diag(a, k=k)
        assert_almost_equal(r.asnumpy(), np.diag(a_np, k=k))

        # k == -1
        k = -1
        r = mx.nd.diag(a, k=k)
        assert_almost_equal(r.asnumpy(), np.diag(a_np, k=k))

        # random k
        k = np.random.randint(-min(LARGE_X, SMALL_Y) + 1, min(LARGE_X, SMALL_Y))
        r = mx.nd.diag(a, k=k)
        assert_almost_equal(r.asnumpy(), np.diag(a_np, k=k))

    @with_seed()
    def check_ravel_multi_index():
        x1, y1 = rand_coord_2d((LARGE_X - 100), LARGE_X, 10, SMALL_Y)
        x2, y2 = rand_coord_2d((LARGE_X - 200), LARGE_X, 9, SMALL_Y)
        x3, y3 = rand_coord_2d((LARGE_X - 300), LARGE_X, 8, SMALL_Y)
        indices_2d = [[x1, x2, x3], [y1, y2, y3]]
        idx = mx.nd.ravel_multi_index(mx.nd.array(indices_2d, dtype=np.int64),
                                      shape=(LARGE_X, SMALL_Y))
        idx_numpy = np.ravel_multi_index(indices_2d, (LARGE_X, SMALL_Y))
        assert np.sum(1 for i in range(idx.size) if idx[i] == idx_numpy[i]) == 3

    @with_seed()
    def check_unravel_index():
        x1, y1 = rand_coord_2d((LARGE_X - 100), LARGE_X, 10, SMALL_Y)
        x2, y2 = rand_coord_2d((LARGE_X - 200), LARGE_X, 9, SMALL_Y)
        x3, y3 = rand_coord_2d((LARGE_X - 300), LARGE_X, 8, SMALL_Y)
        original_2d_indices = [[x1, x2, x3], [y1, y2, y3]]
        idx_numpy = np.ravel_multi_index(original_2d_indices, (LARGE_X, SMALL_Y))
        indices_2d = mx.nd.unravel_index(mx.nd.array(idx_numpy, dtype=np.int64),
                                         shape=(LARGE_X, SMALL_Y))
        assert (indices_2d.asnumpy() == np.array(original_2d_indices)).all()

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, " +
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_transpose():
        check_dtypes = [np.float32, np.int64]
        for dtype in check_dtypes:
            b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y, dtype=dtype)
            t = b.T
            assert t.shape == (SMALL_Y, LARGE_X)
            ref_out = np.transpose(b.asnumpy())
            assert_almost_equal(t.asnumpy(), ref_out, rtol=1e-10)

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, " +
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_swapaxes():
        b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
        t = nd.swapaxes(b, dim1=0, dim2=1)
        assert np.sum(t[:, -1].asnumpy() == (LARGE_X - 1)) == b.shape[1]
        assert t.shape == (SMALL_Y, LARGE_X)

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, " +
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_flip():
        b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
        t = nd.flip(b, axis=0)
        assert np.sum(t[-1, :].asnumpy() == 0) == b.shape[1]
        assert t.shape == (LARGE_X, SMALL_Y)

    def check_sequence_mask():
        # Sequence Mask input [max_sequence_length, batch_size, other_feature_dims]
        # test with input batch_size = 2
        a = nd.arange(0, LARGE_X * SMALL_Y * 2).reshape(LARGE_X, 2, SMALL_Y)
        # test as identity operator
        b = nd.SequenceMask(a)
        assert b[-1][0][1] == a[-1][0][1]
        assert b.shape == a.shape
        # test with default mask
        b = nd.SequenceMask(a, sequence_length=nd.array([1, 1]),
                            use_sequence_length=True)
        assert b[0][1][-1] == a[0][1][-1]  # first sequence of each batch kept
        assert b[-1][-1][-1] != a[-1][-1][-1]  # rest sequences masked
        assert b[-1][-1][-1] == 0

        # test with mask value
        b = nd.SequenceMask(a, sequence_length=nd.array([1, 1]),
                            use_sequence_length=True, value=-1)
        assert b[-1][-1][-1] == -1

    def check_sequence_reverse():
        a = nd.arange(0, LARGE_X * SMALL_Y * 2).reshape(LARGE_X, 2, SMALL_Y)
        # test as reverse operator
        b = nd.SequenceReverse(a)
        assert b[-1][0][0] == a[0][0][0]
        assert b.shape == a.shape
        # test with sequence length
        # 2 rows of batch 1 and 3 rows of batch 2 reversed
        b = nd.SequenceReverse(a, sequence_length=nd.array([2, 3]),
                               use_sequence_length=True)
        assert b[1][0][0] == a[0][0][0]  # check if reversed
        assert b[-1][0][0] == a[-1][0][0]  # check if intact
        assert b.shape == a.shape

    def check_sequence_last():
        a = nd.arange(0, LARGE_X * SMALL_Y * 2).reshape(LARGE_X, 2, SMALL_Y)
        # test if returns last sequence
        b = nd.SequenceLast(a)
        assert_almost_equal(b.asnumpy(), a[-1].asnumpy())  # only checks for (2, SMALL_Y) tensor
        assert b.shape == (2, SMALL_Y)
        # test with sequence length
        # parameter sequence_length - NDArray with shape (batch_size)
        # (2,3) indicates 2nd sequence from batch 1 and 3rd sequence from batch 2
        b = nd.SequenceLast(a, sequence_length=mx.nd.array([2, 3]),
                            use_sequence_length=True)
        # check if it takes 2nd sequence from the first batch
        assert b[0][-1] == a[1][0][-1]

    def check_index_copy():
        x = mx.nd.zeros((LARGE_X, SMALL_Y))
        t = mx.nd.arange(1, SMALL_Y + 1).reshape((1, SMALL_Y))
        index = mx.nd.array([LARGE_X - 1], dtype="int64")

        x = mx.nd.contrib.index_copy(x, index, t)
        assert x[-1][-1] == t[0][-1]

    def check_one_hot():
        # default dtype of ndarray is float32 which cannot index elements over 2^32
        a = nd.array([1, (VLARGE_X - 1)], dtype=np.int64)
        b = nd.one_hot(a, VLARGE_X)
        b[0][1] == 1
        b[1][-1] == 1

    def check_full():
        a = nd.full((SMALL_Y, LARGE_X), 3)
        assert a.shape == (SMALL_Y, LARGE_X)
        assert a[SMALL_Y//2][LARGE_X//2] == 3
        assert a[-1][-1] == 3

    def check_shape():
        b = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        mx.nd.waitall()
        assert b.shape == (SMALL_Y, LARGE_X)

    def check_size():
        b = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        mx.nd.waitall()
        assert b.size == LARGE_SIZE

    def check_copy():
        a = nd.ones((SMALL_Y, LARGE_X))
        b = a.copy()
        nd.waitall()
        assert b.shape == a.shape
        assert b.size == LARGE_SIZE

    def check_copy_to():
        a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        b = nd.array(np.zeros((SMALL_Y, LARGE_X)))
        c = a.copyto(b)
        assert c is b
        assert b[-1][-1] == SMALL_Y-1

    def check_reshape_like():
        a = nd.array(np.zeros((SMALL_Y, LARGE_X)))
        b = nd.array(np.zeros((SMALL_Y//2, LARGE_X*2)))
        c = nd.reshape_like(a, b)
        assert c.shape == (SMALL_Y//2, LARGE_X*2)

    def check_flatten():
        check_dtypes = [np.float32, np.int64]
        for dtype in check_dtypes:
            a = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y, dtype=dtype).reshape((LARGE_X//2, 2, SMALL_Y))
            b = nd.flatten(a)
            # Here we removed the value asserts due to different precision of `int64` and `float32`.
            # For `float32`, it will lose some precision when `LARGE_X` is too large, that is `LARGE_X-1`
            # and `LARGE_X-2` can not represent the accurate value in the current situation.
            assert b.shape == (LARGE_X//2, SMALL_Y*2)
            assert_almost_equal(b[-1,-1].asnumpy(), a[-1,-1,-1].asnumpy(), rtol=1e-8)

    def check_concat():
        a = nd.array(np.ones((SMALL_Y, LARGE_X)))
        b = nd.array(np.zeros((SMALL_Y, LARGE_X)))
        for axis in [0, 1]:
            c = nd.concat(a, b, dim=axis)
            c.wait_to_read()
            assert c.shape[axis] == b.shape[axis] * 2
            assert c.shape[1-axis] == b.shape[1-axis]

    def check_stack():
        a = nd.array(np.ones((SMALL_Y, LARGE_X)))
        b = nd.array(np.zeros((SMALL_Y, LARGE_X)))
        c = nd.stack(a, b, axis=1)
        assert c.shape == (b.shape[0], 2, LARGE_X)

    def check_broadcast_axes():
        a = create_2d_tensor(rows=1, columns=LARGE_X)
        b = nd.broadcast_axis(a, axis=[0], size=2)
        assert b.shape == (a.shape[0]*2, a.shape[1])

    def check_astype():
        x = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        y = x.astype('int32')
        assert y.dtype == np.int32
        assert y[-1][-1] == SMALL_Y-1

    def check_cast():
        x = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        y = nd.cast(x, np.int32)
        assert y.dtype == np.int32
        assert y[-1][-1] == SMALL_Y-1

    def check_repeat():
        x = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X//2)
        y = nd.repeat(x, repeats=2, axis = 1)
        assert y.shape == (SMALL_Y, LARGE_X)
        assert y[0][1] == 0
        assert y[-1][-1] == SMALL_Y-1
        x = create_2d_tensor(rows=SMALL_Y//2, columns=LARGE_X)
        y = nd.repeat(x, repeats=2, axis = 0)
        assert y.shape == (SMALL_Y, LARGE_X)
        assert y[0][1] == 0
        assert y[-1][0] == SMALL_Y//2-1

    def check_ndarray_convert():
        a = nd.zeros(shape=(LARGE_X, SMALL_Y))
        b = a.astype(np.int32)
        assert b.dtype == np.int32
        b = a.tostype('row_sparse')
        assert isinstance(b, mx.nd.sparse.RowSparseNDArray)

    def check_load_save():
        x = create_2d_tensor(SMALL_Y, LARGE_X)
        with TemporaryDirectory() as tmp:
            tmpfile = os.path.join(tmp, 'large_tensor')
            nd.save(tmpfile, [x])
            y = nd.load(tmpfile)
            y = y[0]
            assert x[0][0] == y[0][0]
            assert x[-1][-1]== y[-1][-1]

    def check_pad():
        x = create_2d_tensor(rows=SMALL_Y-2, columns=LARGE_X//2-2, dtype=np.float32).reshape(1 , 1, SMALL_Y-2, LARGE_X//2-2)
        y = nd.pad(x, mode="edge", pad_width=(0, 0, 0, 0, 1, 1, 1, 1))
        assert y[0][0][1][0] == 0
        assert y[0][0][1][-1] == 0
        assert y[0][0][-1][0] == SMALL_Y-3
        assert y[0][0][-1][-1] == SMALL_Y-3
        assert y.shape == (1, 1, SMALL_Y, LARGE_X//2)

    def check_gather():
        arr = mx.nd.ones((LARGE_X, SMALL_Y))
        idx = mx.nd.random.randint(0, LARGE_X, SMALL_X)
        # Calls gather_nd internally
        tmp = arr[idx]
        assert np.sum(tmp[0].asnumpy() == 1) == SMALL_Y
        # Calls gather_nd internally
        arr[idx] += 1
        assert np.sum(arr[idx[0]].asnumpy() == 2) == SMALL_Y

    def check_binary_broadcast():
        def check_correctness(mxnet_op, numpy_op, atol=1e-3):
            a = mx.nd.ones((LARGE_X, SMALL_Y)).as_np_ndarray()
            b = 2*mx.nd.ones((LARGE_X, SMALL_Y)).as_np_ndarray()
            res = mxnet_op(a, b)
            np_res = numpy_op(1, 2)
            assert np.abs(res[-1][-1] - np_res) < atol
        check_correctness(mx.np.arctan2, np.arctan2)
        check_correctness(mx.np.hypot, np.hypot)

    check_ndarray_zeros()
    check_ndarray_ones()
    check_ndarray_random_uniform()
    check_ndarray_random_randint()
    check_ndarray_random_exponential()
    check_ndarray_random_gamma()
    check_ndarray_random_multinomial()
    check_ndarray_random_generalized_negative_binomial()
    check_ndarray_random_negative_binomial()
    check_ndarray_random_normal()
    check_ndarray_random_poisson()
    check_ndarray_random_randn()
    check_ndarray_random_shuffle()
    check_ndarray_empty()
    check_zeros_like()
    check_ones_like()
    check_broadcast()
    check_clip()
    check_split()
    check_tile()
    check_take()
    check_slice()
    check_slice_assign()
    check_slice_like()
    check_slice_axis()
    check_expand_dims()
    check_squeeze()
    check_broadcast_div()
    check_where()
    check_pick()
    check_depthtospace()
    check_spacetodepth()
    check_diag()
    check_ravel_multi_index()
    check_unravel_index()
    check_transpose()
    check_swapaxes()
    check_flip()
    check_sequence_mask()
    check_sequence_reverse()
    check_sequence_last()
    check_index_copy()
    check_one_hot()
    check_full()
    check_shape()
    check_size()
    check_copy()
    check_copy_to()
    check_reshape_like()
    check_flatten()
    check_concat()
    check_stack()
    check_broadcast_axes()
    check_astype()
    check_cast()
    check_repeat()
    check_ndarray_convert()
    check_load_save()
    check_pad()
    check_gather()
    check_binary_broadcast()


@pytest.mark.timeout(0)
def test_basic():
    def check_elementwise():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.ones(shape=(LARGE_X, SMALL_Y))
        res = a + b
        assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]
        res = a + 1
        assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]
        res = nd.sqrt(a + 3)
        assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]

    def check_reduce():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        assert nd.sum(a).asnumpy() == a.shape[0] * a.shape[1]

    def check_dot():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.ones(shape=(SMALL_Y, SMALL_Y))
        res = nd.dot(a, b)
        assert np.sum(res[-1].asnumpy() == SMALL_Y) == b.shape[1]

    def check_argmin():
        a = nd.arange(0, LARGE_X * SMALL_Y).reshape(LARGE_X, SMALL_Y)
        idx = mx.nd.argmin(a, axis=0)
        assert idx.shape[0] == SMALL_Y

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, " +
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_argsort():
        b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
        s = nd.argsort(b, axis=0, is_ascend=False, dtype=np.int64)
        mx.nd.waitall()
        assert (s[0].asnumpy() == (LARGE_X - 1)).all()

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, " +
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_sort():
        b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
        s = nd.sort(b, axis=0, is_ascend=False)
        assert np.sum(s[-1][SMALL_Y//2:SMALL_Y].asnumpy() == 0).all()
        s = nd.sort(b, is_ascend=False)
        assert np.sum(s[0].asnumpy() == 0).all()

    @pytest.mark.skip(reason="Memory doesn't free up after stacked execution with other ops, " +
                      "tracked at https://github.com/apache/mxnet/issues/17411")
    def check_topk():
        b = create_2d_tensor(rows=LARGE_X, columns=SMALL_Y)
        k = nd.topk(b, k=10, axis=0, dtype=np.int64)
        assert np.sum(k.asnumpy() == (LARGE_X - 1)) == SMALL_Y
        ind, val = mx.nd.topk(b, k=3, axis=0, dtype=np.int64, ret_typ="both",
                              is_ascend=False)
        assert np.all(ind == val)
        b = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        l = nd.topk(b, k=1, axis=-1, dtype=np.int64, ret_typ="value")
        assert l.sum() == np.sum(np.arange(0, SMALL_Y))

    def check_exponent_logarithm_operators():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        # exponent
        result = nd.exp(a)
        assert result[0][-1] == 7.389056
        assert result.shape == a.shape
        # exponent minus 1
        result = nd.expm1(a)
        assert result[0][-1] == 6.389056
        assert result.shape == a.shape
        # log2
        result = nd.log2(a)
        assert result[0][-1] == 1
        assert result.shape == a.shape
        # log10
        result = nd.log10(a)
        assert result[0][-1] == 0.30103
        assert result.shape == a.shape
        # log1p
        result = nd.log1p(a)
        assert result[0][-1] == 1.0986123
        assert result.shape == a.shape
        # log
        result = nd.log(a)
        assert result[0][-1] == 0.6931472
        assert result.shape == a.shape

    def check_power_operators():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        # sqrt
        result = nd.sqrt(a)
        assert result[0][-1] == 1.4142135
        assert result.shape == a.shape
        # rsqrt
        result = nd.rsqrt(a)
        assert result[0][-1] == 0.70710677
        assert result.shape == a.shape
        # cbrt
        result = nd.cbrt(a)
        assert result[0][-1] == 1.2599211
        assert result.shape == a.shape
        # rcbrt
        result = nd.rcbrt(a)
        assert result[0][-1] == 0.7937005
        assert result.shape == a.shape
        # square
        result = nd.square(a)
        assert result[0][-1] == 4
        assert result.shape == a.shape
        # reciprocal
        result = nd.reciprocal(a)
        assert result[0][-1] == 0.5
        assert result.shape == a.shape

    def check_elemwise_add():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.ones(shape=(LARGE_X, SMALL_Y))
        res = nd.elemwise_add(a, b)
        assert np.sum(res[-1].asnumpy() == 2) == a.shape[1]

    def check_add():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__add__(a)
        assert c[0][-1] == 2
        assert c.shape == a.shape

    def check_sub():
        a = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__sub__(a)
        assert c[0][-1] == -2
        assert c.shape == a.shape

    def check_rsub():
        a = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__rsub__(a)
        assert c[0][-1] == 2
        assert c.shape == a.shape

    def check_neg():
        a = nd.ones(shape=(LARGE_X, SMALL_Y))
        c = a.__neg__()
        assert c[0][-1] == -1
        assert c.shape == a.shape

    def check_mul():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__mul__(a)
        assert c[0][-1] == 6
        assert c.shape == a.shape

    def check_div():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__div__(a)
        mx_divide = nd.divide(b, a)
        assert c[0][-1] == 3/2
        assert mx_divide[0][-1] == c[0][-1]
        assert c.shape == a.shape

    def check_rdiv():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__rdiv__(a)
        assert c[0][-1] == 2/3
        assert c.shape == a.shape

    def check_mod():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__mod__(a)
        assert c[0][-1] == 1
        assert c.shape == a.shape

    def check_rmod():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__rmod__(a)
        assert c[0][-1] == 2
        assert c.shape == a.shape

    def check_imod():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__imod__(a)
        assert c[0][-1] == 1
        assert c.shape == a.shape

    def check_pow():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__pow__(a)
        assert c[0][-1] == 9
        assert c.shape == a.shape

    def check_rpow():
        a = 2*nd.ones(shape=(LARGE_X, SMALL_Y))
        b = 3*nd.ones(shape=(LARGE_X, SMALL_Y))
        c = b.__rpow__(a)
        assert c[0][-1] == 8
        assert c.shape == a.shape

    def check_sum():
        a = nd.array(np.ones((SMALL_Y, LARGE_X)))
        b = nd.sum(a, axis=1)
        assert b.shape[0] == SMALL_Y

    def check_prod():
        a = nd.array(np.ones((SMALL_Y, LARGE_X)))
        b = nd.prod(a, axis=1)
        assert b.shape[0] == SMALL_Y

    def check_mean():
        a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        b = nd.mean(a, axis=0)
        assert b[0] == (SMALL_Y/2-1)

    def check_min():
        a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        b = nd.min(a, axis=0)
        assert b[0] == 0
        assert b[-1] == 0

    def check_max():
        a = create_2d_tensor(rows=SMALL_Y, columns=LARGE_X)
        b = nd.max(a, axis=0)
        assert b[0] == (SMALL_Y-1)
        assert b[-1] == (SMALL_Y-1)

    def check_norm():
        a = np.array(np.full((1, LARGE_X), 3))
        b = np.array(np.full((1, LARGE_X), 4))
        c = nd.array(np.concatenate((a, b), axis=0))
        d = nd.norm(c, ord=2, axis=0)
        e = nd.norm(c, ord=1, axis=0)
        assert d.shape[0] == LARGE_X
        assert e.shape[0] == LARGE_X
        assert d[-1] == 5
        assert e[-1] == 7

    def check_argmax():
        a = np.ones((SMALL_Y, LARGE_X))
        b = np.zeros((SMALL_Y, LARGE_X))
        c = nd.array(np.concatenate((a, b), axis=0))
        d = nd.argmax(c, axis=0)
        assert d.shape[0] == LARGE_X
        assert d[-1] == d[0] == 0

    def check_iadd():
        a = nd.array(np.ones((SMALL_Y, LARGE_X)))
        b = nd.array(np.ones((SMALL_Y, LARGE_X)))
        c = b + a
        assert c.shape == a.shape
        assert c[0][-1] == 2

    def check_isub():
        a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        b = nd.array(np.ones((SMALL_Y, LARGE_X)))
        c = a - b
        assert c.shape == a.shape
        assert c[0][-1] == 2

    def check_imul():
        a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        b = nd.array(np.ones((SMALL_Y, LARGE_X)))
        c = b * a
        assert c.shape == a.shape
        assert c[0][-1] == 3

    def check_idiv():
        a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 4)))
        b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
        c = a / b
        assert c.shape == a.shape
        assert c[0][-1] == 2

    def check_eq():
        a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        c = (a == b)
        assert np.sum(c[0].asnumpy() == 1).all()

    def check_neq():
        a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
        b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        c = (a != b)
        assert np.sum(c[0].asnumpy() == 1).all()

    def check_lt():
        a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
        b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        d = (a <= b)
        assert np.sum(d[0].asnumpy() == 1).all()

    def check_lte():
        a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
        b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        c = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
        d = (a <= b)
        e = (a <= c)
        assert np.sum(d[0].asnumpy() == 1).all()
        assert np.sum(e[0].asnumpy() == 1).all()

    def check_gt():
        a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
        d = (a >= b)
        assert np.sum(d[0].asnumpy() == 1).all()

    def check_gte():
        a = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        b = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 2)))
        c = nd.array(np.array(np.full((SMALL_Y, LARGE_X), 3)))
        d = (a >= b)
        e = (a >= c)
        assert np.sum(d[0].asnumpy() == 1).all()
        assert np.sum(e[0].asnumpy() == 1).all()

    def check_sign():
        a = mx.nd.random.normal(-1,1, shape=(LARGE_X, SMALL_Y))
        mx_res = mx.nd.sign(a)
        assert_almost_equal(mx_res[-1][-1].asnumpy(), np.sign(a[-1][-1].asnumpy()))

    def check_logical():
        def check_logical_and(a, b):
            mx_res = mx.nd.logical_and(a, b)
            assert_almost_equal(mx_res[-1][-1].asnumpy(), np.logical_and(a[-1][-1].asnumpy(), b[-1][-1].asnumpy()))

        def check_logical_or(a, b):
            mx_res = mx.nd.logical_or(a, b)
            assert_almost_equal(mx_res[-1][-1].asnumpy(), np.logical_or(a[-1][-1].asnumpy(), b[-1][-1].asnumpy()))

        def check_logical_not(a, b):
            mx_res = mx.nd.logical_not(a, b)
            assert_almost_equal(mx_res[-1][-1].asnumpy(), np.logical_not(a[-1][-1].asnumpy(), b[-1][-1].asnumpy()))

        def check_logical_xor(a, b):
            mx_res = mx.nd.logical_xor(a, b)
            assert_almost_equal(mx_res[-1][-1].asnumpy(), np.logical_xor(a[-1][-1].asnumpy(), b[-1][-1].asnumpy()))

        a = mx.nd.ones((LARGE_X, SMALL_Y))
        b = mx.nd.zeros((LARGE_X, SMALL_Y))
        check_logical_and(a, b)
        check_logical_or(a, b)
        check_logical_not(a, b)
        check_logical_xor(a, b)

    def create_input_for_rounding_ops():
        # Creates an vector with values (-LARGE_X/2 .... -2, -1, 0, 1, 2, .... , LARGE_X/2-1)
        # then divides each element by 2 i.e (-LARGE_X/4 .... -1, -0.5, 0, 0.5, 1, .... , LARGE_X/4-1)
        # and finally broadcasts to
        inp = nd.arange(-LARGE_X//2, LARGE_X//2, dtype=np.float64).reshape(1, LARGE_X)
        inp = inp/2
        inp = nd.broadcast_to(inp, (SMALL_Y, LARGE_X))
        return inp

    def assert_correctness_of_rounding_ops(output, mid, expected_vals):
        # checks verifies 5 values at the middle positions of the input vector
        # i.e mid-2, mid-1, mid, mid+1, mid+2
        output_idx_to_inspect = [mid-2, mid-1, mid, mid+1, mid+2]
        for i in range(len(output_idx_to_inspect)):
            assert output[1][output_idx_to_inspect[i]] == expected_vals[i]

    # TODO(access2rohit): merge similar tests in large vector and array into one file.
    def check_rounding_ops():
        x = create_input_for_rounding_ops()
        def check_ceil():
            y = nd.ceil(x)
            # expected ouput for middle 5 values after applying ceil()
            expected_output = [-1, 0, 0, 1, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)
        def check_fix():
            y = nd.fix(x)
            # expected ouput for middle 5 values after applying fix()
            expected_output = [-1, 0, 0, 0, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)
        def check_floor():
            y = nd.floor(x)
            # expected ouput for middle 5 values after applying floor()
            expected_output = [-1, -1, 0, 0, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)
        def check_rint():
            y = nd.rint(x)
            # expected ouput for middle 5 values after applying rint()
            expected_output = [-1, -1, 0, 0, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)
        def check_round():
            y = nd.round(x)
            # expected ouput for middle 5 values after applying round()
            expected_output = [-1, -1, 0, 1, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)
        def check_trunc():
            y = nd.trunc(x)
            # expected ouput for middle 5 values after applying trunc()
            expected_output = [-1, 0, 0, 0, 1]
            assert_correctness_of_rounding_ops(y, LARGE_X//2, expected_output)
        check_ceil()
        check_fix()
        check_floor()
        check_rint()
        check_round()
        check_trunc()

    def create_input_for_trigonometric_ops(vals):
        # Creates large vector input of size(LARGE_X*10, SMALL_Y/10) from vals using broadcast_to operator
        inp = nd.array(vals).reshape(1, 5)
        inp = nd.broadcast_to(inp, (LARGE_X*10, SMALL_Y//10))
        return inp

    def assert_correctness_of_trigonometric_ops(output, expected_vals, atol=1e-3):
        # checks verifies 5 values at positions(0, 1, -3, -2, -1) of the input vector
        output_idx_to_inspect = [0, 1, -3, -2, -1]
        for i in range(len(output_idx_to_inspect)):
            assert np.abs(output[1][output_idx_to_inspect[i]].asnumpy()-expected_vals[i]) <= atol

    def check_trigonometric_ops():
        def check_arcsin():
            x = create_input_for_trigonometric_ops([-1, -.707, 0, .707, 1])
            y = nd.arcsin(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arcsin()
            expected_output = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arccos():
            x = create_input_for_trigonometric_ops([-1, -.707, 0, .707, 1])
            y = nd.arccos(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arccos()
            expected_output = [np.pi, 3*np.pi/4, np.pi/2, np.pi/4, 0]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arctan():
            x = create_input_for_trigonometric_ops([-np.Inf, -1, 0, 1, np.Inf])
            y = nd.arctan(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arctan()
            expected_output = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_sin():
            x = create_input_for_trigonometric_ops([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
            y = nd.sin(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying sin()
            expected_output = [-1, -.707, 0, .707, 1]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_cos():
            x = create_input_for_trigonometric_ops([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            y = nd.cos(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying cos()
            expected_output = [1, .707, 0, -.707, -1]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_tan():
            x = create_input_for_trigonometric_ops([-np.pi/6, -np.pi/4, 0, np.pi/4, np.pi/6])
            y = nd.tan(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying tan()
            expected_output = [-.577, -1, 0, 1, .577]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arcsinh():
            x = create_input_for_trigonometric_ops([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
            y = nd.arcsinh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arcsinh()
            expected_output = [np.arcsinh(-np.pi/2), np.arcsinh(-np.pi/4), 0, np.arcsinh(np.pi/4), np.arcsinh(np.pi/2)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arccosh():
            x = create_input_for_trigonometric_ops([1, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4])
            y = nd.arccosh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arccosh()
            expected_output = [0, np.arccosh(np.pi/2), np.arccosh(3*np.pi/4), np.arccosh(np.pi), np.arccosh(5*np.pi/4)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_arctanh():
            x = create_input_for_trigonometric_ops([-1/4, -1/2, 0, 1/4, 1/2])
            y = nd.arctanh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying arctanh()
            expected_output = [np.arctanh(-1/4), np.arctanh(-1/2), 0, np.arctanh(1/4), np.arctanh(1/2)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_sinh():
            x = create_input_for_trigonometric_ops([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
            y = nd.sinh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying sinh()
            expected_output = [np.sinh(-np.pi/2), np.sinh(-np.pi/4), 0, np.sinh(np.pi/4), np.sinh(np.pi/2)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_cosh():
            x = create_input_for_trigonometric_ops([0, 1, np.pi/2, 3*np.pi/4, np.pi])
            y = nd.cosh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying cosh()
            expected_output = [1, np.cosh(1), np.cosh(np.pi/2), np.cosh(3*np.pi/4), np.cosh(np.pi)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_tanh():
            x = create_input_for_trigonometric_ops([-1/4, -1/2, 0, 1/4, 1/2])
            y = nd.tanh(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying tanh()
            expected_output = [np.tanh(-1/4), np.tanh(-1/2), 0, np.tanh(1/4), np.tanh(1/2)]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_radians():
            x = create_input_for_trigonometric_ops([0, 90, 180, 270, 360])
            y = nd.radians(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying radians()
            expected_output = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        def check_degrees():
            x = create_input_for_trigonometric_ops([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            y = nd.degrees(x)
            # expected ouput for indices=(0, 1, -3, -2, -1) after applying degrees()
            expected_output = [0, 90, 180, 270, 360]
            assert_correctness_of_trigonometric_ops(y, expected_output)

        check_arcsin()
        check_arccos()
        check_arctan()
        check_sin()
        check_cos()
        check_tan()
        check_arcsinh()
        check_arccosh()
        check_arctanh()
        check_sinh()
        check_cosh()
        check_tanh()
        check_radians()
        check_degrees()

    def check_add_n():
        x = [nd.ones(LARGE_X) for j in range(SMALL_Y)]
        y = nd.add_n(*x)
        assert y[0] == SMALL_Y
        assert y[-1] == SMALL_Y

    def check_modulo():
        x = mx.nd.ones((SMALL_Y, LARGE_X))*6
        y = mx.nd.ones(LARGE_X)*4
        z = (x%y)
        assert z[0][0] == 2
        assert z[-1][-1] == 2
        x = mx.nd.ones((SMALL_Y, LARGE_X))*5
        z = nd.modulo(x,y)
        assert z[0][0] == 1
        assert z[-1][-1] == 1

    def check_maximum():
        x = mx.nd.ones((SMALL_Y, LARGE_X))*3
        y = mx.nd.ones(LARGE_X)*4
        z = nd.maximum(x, y)
        assert z[0][0] == 4
        assert z[-1][-1] == 4
        z = nd.maximum(x, 5)
        assert z[0][0] == 5
        assert z[-1][-1] == 5

    def check_minimum():
        x = mx.nd.ones((SMALL_Y, LARGE_X))*3
        y = mx.nd.ones(LARGE_X)*2
        z = nd.minimum(x, y)
        assert z[0][0] == 2
        assert z[-1][-1] == 2
        z = nd.minimum(x, 5)
        assert z[0][0] == 3
        assert z[-1][-1] == 3

    check_elementwise()
    check_reduce()
    check_dot()
    check_argmin()
    check_argsort()
    check_sort()
    check_topk()
    check_exponent_logarithm_operators()
    check_power_operators()
    check_elemwise_add()
    check_add()
    check_sub()
    check_rsub()
    check_neg()
    check_mul()
    check_div()
    check_rdiv()
    check_mod()
    check_rmod()
    check_imod()
    check_pow()
    check_rpow()
    check_sum()
    check_prod()
    check_mean()
    check_min()
    check_max()
    check_norm()
    check_argmax()
    check_iadd()
    check_isub()
    check_imul()
    check_idiv()
    check_eq()
    check_neq()
    check_lt()
    check_lte()
    check_gt()
    check_gte()
    check_sign()
    check_logical()
    check_rounding_ops()
    check_trigonometric_ops()
    check_add_n()
    check_modulo()
    check_maximum()
    check_minimum()


@pytest.mark.timeout(0)
def test_sparse_dot():
    shape = (2, VLARGE_X)
    sp_mat1 = nd.sparse.csr_matrix(([2], [6], [0, 1, 1]), shape=shape)
    mat2 = nd.ones((VLARGE_X, 2))
    out = nd.dot(sp_mat1, mat2)
    assert out.asnumpy()[0][0] == 2
    assert out.shape == (2, 2)

