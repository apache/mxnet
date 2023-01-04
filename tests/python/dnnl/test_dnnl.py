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

"""
DNNL related test cases
"""
import sys
import os
import numpy as np
import mxnet as mx
import pytest
from mxnet.test_utils import rand_ndarray, assert_almost_equal
from mxnet import gluon, context, use_np
from mxnet.gluon import nn
from mxnet.test_utils import *
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../unittest/'))
import itertools

@use_np
@pytest.mark.seed(1234)
def test_dnnl_ndarray_slice():
    ctx = mx.cpu()
    net = gluon.nn.HybridSequential()
    net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation=None))
    net.initialize(ctx=ctx)
    x = mx.np.array(np.ones([32, 3, 224, 224]), ctx=ctx)
    y = net(x)

    # trigger computation on ndarray slice
    assert_almost_equal(y[0].asnumpy()[0, 0, 0], np.array(0.056331709))


# In python3.8, functions are only pickable if they are defined in
# the top-level of a module
# https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
class Dummy(gluon.data.Dataset):
    def __len__(self):
        return 2
    def __getitem__(self, key):
        return key, np.ones((3, 224, 224)), np.ones((10, ))

@use_np
@pytest.mark.seed(1234)
def test_dnnl_engine_threading():
    net = gluon.nn.HybridSequential()
    net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation=None))
    net.initialize(ctx=mx.cpu())

    loader = gluon.data.DataLoader(Dummy(), batch_size=2, num_workers=1)

    X = (32, 3, 32, 32)
    # trigger dnnl execution thread
    y = net(mx.np.array(np.ones(X))).asnumpy()

    # Use Gluon dataloader to trigger different thread.
    # below line triggers different execution thread
    for _ in loader:
        y = net(mx.np.array(np.ones(X))).asnumpy()
        # output should be 056331709 (non-dnnl mode output)
        assert_almost_equal(y[0, 0, 0, 0], np.array(0.056331709))
        break

def test_dnnl_reshape():
    def test_reshape_after_conv(dst_shape):
        shape = (1,1,4,4)
        data = mx.symbol.Variable('data')
        conv = mx.symbol.Convolution(data=data, num_filter=16, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
        res = mx.symbol.reshape(data=conv, shape=dst_shape)
        exe = res._simple_bind(mx.cpu(), data=shape, grad_req='null')

        val1 = np.random.uniform(-1, 1, shape)
        val2 = np.random.uniform(-1, 1, (16, 1, 1, 1))
        val3 = np.random.uniform(-1 ,1, (1))

        exe.arg_arrays[0][:] = val1
        exe.arg_arrays[1][:] = val2
        exe.arg_arrays[2][:] = val3
        outputs = exe.forward(is_train=False)[0].asnumpy()

        conv_exe = conv._simple_bind(mx.cpu(), data=shape, grad_req='null')
        conv_exe.arg_arrays[0][:] = val1
        conv_exe.arg_arrays[1][:] = val2
        conv_exe.arg_arrays[2][:] = val3
        data_npy = conv_exe.forward(is_train=False)[0].asnumpy()
        assert_almost_equal(outputs, data_npy.reshape(dst_shape))


    # Test dnnl reshape (Using shape)
    test_cases = [(256), (16, 16), (4, 4, 16), (4, 4, 4, 4)]
    for test_case in test_cases:
        test_reshape_after_conv(test_case)


@use_np
def test_reshape_before_conv():
    class Net(gluon.HybridBlock):
        """
        test Net
        """
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(10, (3, 3))
            self.conv1 = nn.Conv2D(5, (3, 3))

        def forward(self, x, *args, **kwargs):
            x_reshape = x.reshape((2, 4, 20, 5))
            y = self.conv0(x_reshape)
            y_reshape = y.reshape((2, 10, 9, 6))
            out = self.conv1(y_reshape)
            return out

    x = mx.np.random.uniform(size=(2, 4, 10, 10))
    x.attach_grad()
    net = Net()
    net.initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()
    dx1 = x.grad
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x)
    out2.backward()
    assert_almost_equal(dx1, x.grad, rtol=1e-5, atol=1e-6)
    assert_almost_equal(out1, out2, rtol=1e-5, atol=1e-6)


@use_np
def test_slice_before_conv():
    class Net(gluon.HybridBlock):
        """
        test Net
        """
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(4, (3, 3))
            self.conv1 = nn.Conv2D(4, (3, 3))

        def forward(self, x, *args, **kwargs):
            x_slice = mx.npx.slice(x, begin=(0, 0, 0, 0), end=(2, 4, 10, 10))
            y = self.conv0(x_slice)
            y_slice = mx.npx.slice(y, begin=(1, 0, 2, 2), end=(2, 1, 7, 7))
            out = self.conv1(y_slice)
            return out

    x = mx.np.random.uniform(size=(2, 10, 10, 10))
    x.attach_grad()
    net = Net()
    net.initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()
    dx1 = x.grad
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x)
    out2.backward()
    assert_almost_equal(dx1, x.grad, rtol=1e-5, atol=1e-6)
    assert_almost_equal(out1, out2, rtol=1e-5, atol=1e-6)


@use_np
def test_slice_reshape_before_conv():
    class Net(gluon.HybridBlock):
        """
        test Net
        """
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv0 = nn.Conv2D(4, (3, 3))
            self.conv1 = nn.Conv2D(4, (3, 3))

        def forward(self, x, *args, **kwargs):
            x_slice = mx.npx.slice(x, begin=(0, 0, 0, 0), end=(2, 4, 8, 9))
            y = self.conv0(x_slice)
            y_reshape = y.reshape((2, 4, 14, 3))
            out = self.conv1(y_reshape)
            return out

    x = mx.np.random.uniform(size=(2, 10, 10, 10))
    x.attach_grad()
    net = Net()
    net.initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()
    dx1 = x.grad
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x)
    out2.backward()
    assert_almost_equal(dx1, x.grad, rtol=1e-5, atol=1e-6)
    assert_almost_equal(out1, out2, rtol=1e-5, atol=1e-6)


def test_flatten_slice_after_conv():
    data = mx.symbol.Variable('data')
    weight = mx.symbol.Variable('weight')
    bias = mx.symbol.Variable('bias')
    conv1= mx.symbol.Convolution(data = data, weight=weight, bias=bias, name='conv1', num_filter=64, kernel=(3,3), stride=(1,1))
    flatten1 = mx.symbol.flatten(data = conv1)
    slice1 = mx.symbol.slice(data = flatten1, begin=0, end=1)

    shape = (2, 16, 16, 16)
    val = np.random.rand(2, 16, 16, 16).astype(np.float32)
    exe = slice1._simple_bind(context.current_context(), data=shape)
    exe.arg_arrays[0][:] = val
    exe.arg_arrays[1][:] = np.random.normal(size=exe.arg_arrays[1].shape)
    exe.arg_arrays[2][:] = np.random.normal(size=exe.arg_arrays[2].shape)
    p = exe.forward(is_train=False)
    p[0].wait_to_read()
    print(p[0])


def test_dnnl_sum_with_dnnl_layout():

    x_shape = (32, 3, 224, 224)
    x_npy = np.ones(x_shape, dtype='float32')
    w_shape = (32, 3, 3, 3)
    w_npy = np.ones(w_shape, dtype='float32')

    x = mx.sym.Variable("x")
    w = mx.sym.Variable("w")
    z = mx.symbol.Convolution(data=x, weight=w, num_filter=32, kernel=(3, 3))
    num_inputs = [2, 3, 4, 5]
    for i in num_inputs:
        inputs = []
        for _ in range(i):
            inputs.append(z)
        y = mx.sym.add_n(*inputs) # (only DNNL data input)
        exe = y._simple_bind(ctx=mx.cpu(), x=x_shape, w=w_shape)
        out = exe.forward(is_train=False, x=x_npy, w=np.ones(w_shape))[0]
        #conv with kernel (3,3) on ones should give result=27
        single_cov = 27.0
        assert_almost_equal(out[0].asnumpy()[0, 0, 0], single_cov*i)

def test_dnnl_sum_inplace_with_cpu_layout():
    x_shape = (32, 3, 224, 224)
    x_npy = np.ones(x_shape, dtype='float32')
    y_shape = (32, 32, 222, 222)
    y_npy = np.ones(y_shape, dtype='float32')
    x = mx.sym.Variable("x")
    y = mx.sym.Variable("y")
    z = mx.symbol.Convolution(data=x, num_filter=32, kernel=(3, 3))
    z = mx.sym.add_n(z, y) # (DNNL data, cpu data)
    exe = z._simple_bind(ctx=mx.cpu(), x=x_shape, y=y_shape)
    out = exe.forward(is_train=False, x=x_npy, y=y_npy)[0]
    assert_almost_equal(out[0].asnumpy()[0, 0, 0], 1.0)


def test_batchnorm():
    def check_batchnorm_training(stype):
        for shape in [(2, 3), (2, 4), (2, 3, 2, 2), (2, 4, 2, 2)]:
            data_tmp = np.random.normal(-0.1, 0.1, size=shape)
            s = shape[1],
            gamma = np.ones(s)
            beta = np.ones(s)
            gamma[1] = 3
            beta[0] = 3

            rolling_mean = np.random.uniform(size=s)
            rolling_std = np.random.uniform(size=s)

            data = mx.symbol.Variable('data', stype=stype)
            in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(gamma).tostype(stype),
                           mx.nd.array(beta).tostype(stype)]
            mean_std = [mx.nd.array(rolling_mean).tostype(stype), mx.nd.array(rolling_std).tostype(stype)]

            test = mx.symbol.BatchNorm(data, fix_gamma=False)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-2)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_batchnorm_training(stype)


def test_softmax():
    def check_softmax_training(stype):
        for shape in [(2, 3), (2, 3, 2, 2)]:
            data_tmp = np.random.normal(-0.1, 0.1, size=shape)

            data = mx.symbol.Variable('data', stype=stype)
            in_location = [mx.nd.array(data_tmp).tostype(stype)]

            test = mx.symbol.softmax(data, axis=-1)
            check_numeric_gradient(test, in_location, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_softmax_training(stype)


def test_pooling():
    def check_pooling_training(stype):
        for shape in [(3, 3, 10), (3, 3, 20, 20), (3, 3, 10, 20, 20)]:
            data_tmp = np.random.normal(-0.1, 0.1, size=shape)
            data = mx.symbol.Variable('data', stype=stype)
            in_location = [mx.nd.array(data_tmp).tostype(stype)]

            if np.array(shape).shape[0] == 3:
                test = mx.symbol.Pooling(data=data, kernel=(3), stride=(2), pool_type='avg')
            elif np.array(shape).shape[0] == 4:
                test = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='avg')
            elif np.array(shape).shape[0] == 5:
                test = mx.symbol.Pooling(data=data, kernel=(3, 3, 3), stride=(2, 2, 2), pool_type='avg')
            else:
                return 0
            check_numeric_gradient(test, in_location, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_pooling_training(stype)

@pytest.mark.parametrize('num_filter', [4, 8, 16])
@pytest.mark.parametrize('output_size', [4, 5, 8, 16])
@pytest.mark.parametrize('stype', ['row_sparse', 'default'])
@pytest.mark.parametrize('shape', [(3, 3, 8, 8), (3, 3, 20, 20), (3, 3, 32, 32)])
def test_adaptive_pooling(num_filter, output_size, stype, shape):
    data_tmp = mx.nd.random.uniform(shape=shape)
    data = mx.sym.var('data', stype=stype)
    in_channels = shape[1]

    data = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1,1), num_filter=num_filter)
    data = mx.sym.contrib.AdaptiveAvgPooling2D(data=data, output_size=output_size)

    weight_tmp = np.random.normal(-0.1, 0.1, size=(num_filter, in_channels, 3, 3))
    bias_tmp = np.random.normal(0.1, 0.1, size=(num_filter,))
    
    in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(weight_tmp).tostype(stype),
                    mx.nd.array(bias_tmp).tostype(stype)]
                    
    check_numeric_gradient(data, in_location, numeric_eps=1e-2, rtol=0.16, atol=1e-4)
    

def test_activation():
    def check_activation_training(stype):
        for shape in [(2, 3, 3), (2, 3, 2, 2)]:
            eps = 1e-5
            data_tmp = np.random.normal(-0.1, 1, size=shape)
            # Avoid finite difference method inaccuracies due to discontinuous gradient at the origin.
            # Here we replace small problematic inputs with 1.0.  Repro issue with seed 851486559.
            data_tmp[abs(data_tmp) < eps] = 1.0

            data = mx.symbol.Variable('data', stype=stype)
            in_location = [mx.nd.array(data_tmp).tostype(stype)]

            test = mx.symbol.Activation(data, act_type="relu")
            check_numeric_gradient(test, in_location, numeric_eps=eps, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_activation_training(stype)


def test_convolution():
    def check_convolution_training(stype):
        for shape in [(3, 3, 10), (3, 3, 10, 10), (3, 3, 10, 10, 10)]:
            data_tmp = np.random.normal(-0.1, 1, size=shape)
            data = mx.symbol.Variable('data', stype=stype)

            if np.array(shape).shape[0] == 3:
                test = mx.symbol.Convolution(data=data, kernel=(3,), stride=(2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(4, 3, 3))
            elif np.array(shape).shape[0] == 4:
                test = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2, 2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(4, 3, 3, 3))
            elif np.array(shape).shape[0] == 5:
                test = mx.symbol.Convolution(data=data, kernel=(3, 3, 3), stride=(2, 2, 2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(4, 3, 3, 3, 3))
            else:
                return 0
            bias_tmp = np.random.normal(0.1, 0.1, size=(4,))
            in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(weight_tmp).tostype(stype),
                           mx.nd.array(bias_tmp).tostype(stype)]
            check_numeric_gradient(test, in_location, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_convolution_training(stype)


def test_Deconvolution():
    def check_Deconvolution_training(stype):
        for shape in [(3, 3, 10), (3, 3, 10, 10), (3, 3, 3, 10, 10)]:
            data_tmp = np.random.normal(-0.1, 1, size=shape)
            data = mx.symbol.Variable('data', stype=stype)

            if np.array(shape).shape[0] == 3:
                test = mx.symbol.Deconvolution(data=data, kernel=(3,), stride=(2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(3, 4, 3))
            elif np.array(shape).shape[0] == 4:
                test = mx.symbol.Deconvolution(data=data, kernel=(3, 3), stride=(2, 2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(3, 4, 3, 3))
            elif np.array(shape).shape[0] == 5:
                test = mx.symbol.Deconvolution(data=data, kernel=(3,3,3), stride=(2,2,2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(3, 4, 3, 3, 3))
            else:
                return 0
            bias_tmp = np.random.normal(0.1, 0.1, size=(4,))
            in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(weight_tmp).tostype(stype),
                           mx.nd.array(bias_tmp).tostype(stype)]
            check_numeric_gradient(test, in_location, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_Deconvolution_training(stype)


def test_LRN():
    def check_LRN_training(stype):
        for shape in [(3, 4, 5, 5)]:
            data_tmp = np.random.normal(-0.1, 0.1, size=shape)
            data = mx.symbol.Variable('data', stype=stype)
            in_location = [mx.nd.array(data_tmp).tostype(stype)]

            test = mx.symbol.LRN(data, nsize=3)
            check_numeric_gradient(test, in_location, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_LRN_training(stype)


def test_fullyconnected():
    def check_fullyconnected_training(stype):
        data_shape = rand_shape_nd(2)
        weight_shape = rand_shape_nd(2)
        weight_shape = (weight_shape[0], data_shape[1])
        for density in [1.0, 0.5, 0.0]:
            x = rand_ndarray(shape=data_shape, stype=stype, density=density)
            w = rand_ndarray(shape=weight_shape, stype=stype, density=density)
            x_sym = mx.sym.Variable("data")
            w_sym = mx.sym.Variable("weight")
            sym = mx.sym.FullyConnected(data=x_sym, weight=w_sym, num_hidden=weight_shape[0], no_bias=True)
            in_location = [x, w]
            check_numeric_gradient(sym, in_location, numeric_eps=1e-3, rtol=1e-3, atol=5e-3)
    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_fullyconnected_training(stype)

def test_softmax_with_large_inputs():
    def softmax_forward(input_data, true_output):
        data = mx.sym.Variable('data')
        out1 = data.softmax(axis=1)
        exec1 = out1._bind(mx.cpu(), args={'data': input_data})
        exec1.forward()[0].wait_to_read()
        ndarr = exec1.outputs[0][0][0][0]
        nparr = ndarr.asnumpy()
        assert_almost_equal(nparr, true_output, rtol=1e-5, atol=1e-5)

    softmax_forward(mx.nd.array([[[[-1e30,-1e30]]]]), np.array([1.0,1.0]))
    softmax_forward(mx.nd.array([[[[1e30,1e30]]]]), np.array([1.0,1.0]))
    softmax_forward(mx.nd.array([[[[-3.4e38,-3.4e38]]]]), np.array([1.0,1.0]))
    softmax_forward(mx.nd.array([[[[3.4e38,3.4e38]]]]), np.array([1.0,1.0]))

def test_non_dnnl_fcomputeex():
    # test special case where DNNL formatted NDArray feeds into non-dnnl fcomputeex operator
    # conv is example where DNNL NDArray is created from regular NDArrays
    # CustomOps is example of non-dnnl fcomputeex operator

    @mx.operator.register("custom")
    class CustomProp(mx.operator.CustomOpProp):
        def __int__(self):
            super(CustomProp, self).__init__(need_top_grad=False)

        def list_arguments(self):
            return ['data']

        def list_outputs(self):
            return ['output']

        def infer_shape(self, in_shape):
            data_shape = in_shape[0]
            output_shape = in_shape[0]
            return [data_shape], [output_shape], []

        def infer_type(self, in_type):
            dtype = in_type[0]
            return [dtype], [dtype], []

        def create_operator(self, ctx, shapes, dtypes):
            return Custom()


    class Custom(mx.operator.CustomOp):
        def forward(self, is_train, req, in_data, out_data, aux):
            print(in_data[0])
            self.assign(out_data[0], req[0], in_data[0])

        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            self.assign(in_grad[0], req[0], out_grad)

    data = mx.symbol.Variable('data')
    conv = mx.sym.Convolution(data=data, kernel=(5, 5), pad=(1, 1), stride=(1,1), num_filter=8, name="conv", no_bias=True)
    custom = mx.symbol.Custom(name='custom', data=conv, op_type='custom')
    exec1 = custom._bind(mx.cpu(), args={'data': mx.nd.ones([10,3,96,96]), 'conv_weight': mx.nd.ones([8,3,5,5])})
    exec1.forward()[0].wait_to_read()

def test_conv_transpose():
    axes = [(0,2,1,3), (0,2,3,1), (1,2,3,0), (3,2,1,0)]
    a = np.random.rand(10, 16, 50, 50)
    b = np.random.rand(32, 16, 3, 3)
    x = mx.nd.array(a)
    w = mx.nd.array(b)
    y = mx.nd.Convolution(data=x, weight=w, kernel=(3, 3), num_group=1, num_filter=32, no_bias=True)
    for axis in axes:
        t = mx.nd.transpose(y, axis)
        t.wait_to_read()
        s = y.asnumpy()
        n = np.transpose(s, axis)
        np.allclose(t.asnumpy(), n)


# This test case is contributed by @awsbillz in https://github.com/apache/mxnet/issues/14766
@use_np
def test_reshape_transpose_6d():
    class Reshape2D(gluon.HybridBlock):
        def __init__(self, factor):
            super(Reshape2D, self).__init__()
            self._factors = (int(factor),) * 2

        def forward(self, x):
            f1, f2 = self._factors
            N = 1
            C = 2
            H = W = 596
                                                          # (N, f1*f2*C, H, W)
            x = mx.np.reshape(x, (N, C, f1 * f2, H, W))  # (N, C, f1*f2, H, W)
            x = mx.np.reshape(x, (N, C, f1, f2, H, W))    # (N, C, f1, f2, H, W)
            x = mx.np.transpose(x, (0, 1, 4, 2, 5, 3))        # (N, C, H, f1, W, f2)
            x = mx.np.reshape(x, (N, C, H*f1, W*f2))              # (N, C, H*f1, W*f2)
            return x


    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            self.conv1 = nn.Conv2D(8, kernel_size=5)
            self.reshape2D = Reshape2D(2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.reshape2D(x)
            return x

    net = Net()
    net.initialize(mx.init.Xavier(), ctx=mx.cpu())
    net.hybridize()
    data = mx.np.random.normal(size=(1, 3, 600, 600))
    output = net(data)
    a = output.asnumpy()

def test_concat():
    def ref_concat(a, b, axis):
      return np.concatenate((a, b), axis=axis)

    a_sym = mx.sym.Variable("a")
    b_sym = mx.sym.Variable("b")
    dshape = rand_shape_nd(4)
    a_shape = tuple(dshape)
    b_shape = tuple(dshape)

    for axis in range(0, 4):
      z = mx.sym.concat(a_sym, b_sym, dim=axis)
      a = np.random.uniform(-1, 1, a_shape)
      b = np.random.uniform(-1, 1, b_shape)
      exe = z._simple_bind(ctx=mx.cpu(), a=a_shape, b=b_shape)
      out = exe.forward(is_train=False, a=a, b=b)
      ref_out = ref_concat(a, b, axis=axis)
      out = out[0].asnumpy()
      assert_almost_equal(out, ref_out)

    def check_concat_training(stype):
        data_shape = rand_shape_nd(4)
        for density in [1.0, 0.5, 0.0]:
            a_sym = mx.sym.Variable('a')
            b_sym = mx.sym.Variable('b')
            sym = mx.sym.concat(a_sym, b_sym, dim=1)
            a = rand_ndarray(shape=data_shape, stype=stype, density=density)
            b = rand_ndarray(shape=data_shape, stype=stype, density=density)
            in_location = [a, b]
            check_numeric_gradient(sym, in_location, numeric_eps=1e-3, rtol=1e-3, atol=5e-3)
    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_concat_training(stype)

def test_concat_blocked():
    ctx = mx.cpu()
    axis = 1
    filters = 32  # must be a multiple of 16
    kernel = (3, 3)
    for in_dim_size in range(1, 17):  # check cases with and without padding
        in_shape = (1, in_dim_size, 64, 64)
        in_data = mx.nd.random.uniform(-1, 1, in_shape, ctx=ctx)
        conv_weights = mx.nd.random.uniform(-1, 1, (filters, in_shape[1], kernel[0], kernel[1]), ctx=ctx)

        def calc_output_of_layer(layer):
            ex = layer._simple_bind(ctx, x=in_shape)
            in_data.copyto(ex.arg_arrays[0])
            conv_weights.copyto(ex.arg_arrays[1])
            return ex.forward()[0].asnumpy()

        x = mx.sym.Variable('x')
        w = mx.sym.Variable('w')
        # convolution, so a blocked format is selected
        conv = mx.sym.Convolution(data=x, weight=w, num_filter=filters, kernel=kernel, pad=(1, 1), no_bias=True)
        conc = mx.sym.concat(conv, x, dim=axis)

        # first calculate the output of the convolution to determine ref_out
        conv_out = calc_output_of_layer(conv)
        ref_out = np.concatenate((conv_out, in_data.asnumpy()), axis=axis)

        out = calc_output_of_layer(conc)
        assert_almost_equal(out, ref_out)

def test_elemwise_add():
    def ref_add(a, b):
      return np.add(a, b)

    a_sym = mx.sym.Variable("a")
    b_sym = mx.sym.Variable("b")
    dshape = rand_shape_nd(4)
    a_shape = tuple(dshape)
    b_shape = tuple(dshape)
    z = mx.sym.elemwise_add(a_sym, b_sym)
    a = np.random.uniform(-1, 1, a_shape)
    b = np.random.uniform(-1, 1, b_shape)
    exe = z._simple_bind(ctx=mx.cpu(), a=a_shape, b=b_shape)
    out = exe.forward(is_train=False, a=a, b=b)
    ref_out = ref_add(a, b)
    out = out[0].asnumpy()
    assert_almost_equal(out, ref_out, rtol=1e-6, atol=1e-6)

    def check_elemwise_add_training(stype):
        data_shape = rand_shape_nd(4)
        for density in [1.0, 0.5, 0.0]:
            a_sym = mx.sym.Variable('a')
            b_sym = mx.sym.Variable('b')
            sym = mx.sym.elemwise_add(a_sym, b_sym)
            a = rand_ndarray(shape=data_shape, stype=stype, density=density)
            b = rand_ndarray(shape=data_shape, stype=stype, density=density)
            in_location = [a, b]
            check_numeric_gradient(sym, in_location, numeric_eps=1e-3, rtol=1e-3, atol=5e-3)
    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_elemwise_add_training(stype)

def test_rnn():
    SEQ_LENGTH = [2**10, 2**5]
    STATE_SIZE = [1, 2]
    BATCH_SIZE = [4]
    INPUT_SIZE = [4]
    def batch_check(seq_length, state_size, batch_size, input_size):
        modes_params = [('rnn_relu', mx.np.random.normal(0, 1, ((input_size + state_size + 2)*state_size),)),
                        ('rnn_tanh', mx.np.random.normal(0, 1, ((input_size + state_size + 2)*state_size),)),
                        ('gru', mx.np.random.normal(0, 1, ((input_size + state_size + 2)*state_size*3),))
                        ]
        for m, p in modes_params:
            data = mx.np.random.normal(0, 1, (seq_length, batch_size, input_size))
            state = mx.np.random.normal(0, 1, (1, batch_size, state_size))
            data.attach_grad()
            state.attach_grad()

            with mx.autograd.record():
                y = mx.npx.rnn(data=data, parameters=p, mode=m, \
                               state=state, state_size=state_size, num_layers=1)
            assert y.shape == (seq_length, batch_size, state_size)
            assert type(y[0]).__name__ == 'ndarray'
            y.backward()
            assert state.shape == (1, batch_size, state_size)
            assert type(state[0]).__name__ == 'ndarray'

    for sl, ss, bs, in_s in itertools.product(SEQ_LENGTH, STATE_SIZE, BATCH_SIZE, INPUT_SIZE): 
        batch_check(sl, ss, bs, in_s)
