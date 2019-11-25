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
MKL-DNN related test cases
"""
import sys
import os
import numpy as np
import mxnet as mx
import unittest
from mxnet.test_utils import rand_ndarray, assert_almost_equal
from mxnet.module import Module
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.test_utils import *
import test_mkldnn_install as install
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../unittest/'))
from common import with_seed


def test_mkldnn_model():
    model = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data",
                         "test_mkldnn_test_mkldnn_model_model1.json")
    shape = (32, 3, 300, 300)
    ctx = mx.cpu()

    sym = mx.sym.load(model)
    args = sym.list_arguments()
    shapes = sym.infer_shape(data=shape)

    def get_tensors(args, shapes, ctx):
        return {x: mx.nd.ones(y, ctx) for x, y in zip(args, shapes)}

    inputs = get_tensors(args, shapes[0], ctx)
    grads = get_tensors(args, shapes[0], ctx)

    try:
        exe = sym.bind(ctx, inputs, args_grad=grads)
        for _ in range(2):
            exe.forward(is_train=True)
            for y in exe.outputs:
                y.wait_to_read()
            exe.backward()
            for y in exe.grad_arrays:
                y.wait_to_read()
    except:  # pylint: disable=bare-except
        assert 0, "test_mkldnn_model exception in bind and execution"

def test_mkldnn_ndarray_slice():
    ctx = mx.cpu()
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation=None))
    net.collect_params().initialize(ctx=ctx)
    x = mx.nd.array(np.ones([32, 3, 224, 224]), ctx)
    y = net(x)

    # trigger computation on ndarray slice
    assert_almost_equal(y[0].asnumpy()[0, 0, 0], 0.3376348)

def test_mkldnn_engine_threading():
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation=None))
    net.collect_params().initialize(ctx=mx.cpu())
    class Dummy(gluon.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, key):
            return key, np.ones((3, 224, 224)), np.ones((10, ))

    loader = gluon.data.DataLoader(Dummy(), batch_size=2, num_workers=1)

    X = (32, 3, 32, 32)
    # trigger mkldnn execution thread
    y = net(mx.nd.array(np.ones(X))).asnumpy()

    # Use Gluon dataloader to trigger different thread.
    # below line triggers different execution thread
    for _ in loader:
        y = net(mx.nd.array(np.ones(X))).asnumpy()
        # output should be 016711406 (non-mkldnn mode output)
        assert_almost_equal(y[0, 0, 0, 0], 0.016711406)
        break

@with_seed()
def test_mkldnn_reshape():
    def test_reshape_after_conv(dst_shape):
        shape = (1,1,4,4)
        data = mx.symbol.Variable('data')
        conv = mx.symbol.Convolution(data=data, num_filter=16, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
        res = mx.symbol.reshape(data=conv, shape=dst_shape)
        exe = res.simple_bind(mx.cpu(), data=shape, grad_req='null')

        val1 = np.random.uniform(-1, 1, shape)
        val2 = np.random.uniform(-1, 1, (16, 1, 1, 1))
        val3 = np.random.uniform(-1 ,1, (1))

        exe.arg_arrays[0][:] = val1
        exe.arg_arrays[1][:] = val2
        exe.arg_arrays[2][:] = val3
        outputs = exe.forward(is_train=False)[0].asnumpy()

        conv_exe = conv.simple_bind(mx.cpu(), data=shape, grad_req='null')
        conv_exe.arg_arrays[0][:] = val1
        conv_exe.arg_arrays[1][:] = val2
        conv_exe.arg_arrays[2][:] = val3
        data_npy = conv_exe.forward(is_train=False)[0].asnumpy()
        assert_almost_equal(outputs, data_npy.reshape(dst_shape))


    # Test mkldnn reshape (Using shape)
    test_cases = [(256), (16, 16), (4, 4, 16), (4, 4, 4, 4)]
    for test_case in test_cases:
        test_reshape_after_conv(test_case)


@with_seed()
def test_reshape_before_conv():
    class Net(gluon.HybridBlock):
        """
        test Net
        """
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(10, (3, 3))
                self.conv1 = nn.Conv2D(5, (3, 3))

        def hybrid_forward(self, F, x, *args, **kwargs):
            x_reshape = x.reshape((0, 0, 20, 5))
            y = self.conv0(x_reshape)
            y_reshape = y.reshape((0, 0, 9, 6))
            out = self.conv1(y_reshape)
            return out
    x = mx.nd.random.uniform(shape=(2, 4, 10, 10))
    x.attach_grad()
    net = Net()
    net.collect_params().initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()
    dx1 = x.grad
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x)
    out2.backward()
    mx.test_utils.assert_almost_equal(dx1.asnumpy(), x.grad.asnumpy(), rtol=1e-5, atol=1e-6)
    mx.test_utils.assert_almost_equal(out1.asnumpy(), out2.asnumpy(), rtol=1e-5, atol=1e-6)


@with_seed()
def test_slice_before_conv():
    class Net(gluon.HybridBlock):
        """
        test Net
        """
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(4, (3, 3))
                self.conv1 = nn.Conv2D(4, (3, 3))

        def hybrid_forward(self, F, x, *args, **kwargs):
            x_slice = x.slice(begin=(0, 0, 0, 0), end=(2, 4, 10, 10))
            y = self.conv0(x_slice)
            y_slice = y.slice(begin=(1, 0, 2, 2), end=(2, 1, 7, 7))
            out = self.conv1(y_slice)
            return out
    x = mx.nd.random.uniform(shape=(2, 10, 10, 10))
    x.attach_grad()
    net = Net()
    net.collect_params().initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()
    dx1 = x.grad
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x)
    out2.backward()
    mx.test_utils.assert_almost_equal(dx1.asnumpy(), x.grad.asnumpy(), rtol=1e-5, atol=1e-6)
    mx.test_utils.assert_almost_equal(out1.asnumpy(), out2.asnumpy(), rtol=1e-5, atol=1e-6)


@with_seed()
def test_slice_reshape_before_conv():
    class Net(gluon.HybridBlock):
        """
        test Net
        """
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(4, (3, 3))
                self.conv1 = nn.Conv2D(4, (3, 3))

        def hybrid_forward(self, F, x, *args, **kwargs):
            x_slice = x.slice(begin=(0, 0, 0, 0), end=(2, 4, 8, 9))
            y = self.conv0(x_slice)
            y_reshape = y.reshape((0, 0, 14, 3))
            out = self.conv1(y_reshape)
            return out
    x = mx.nd.random.uniform(shape=(2, 10, 10, 10))
    x.attach_grad()
    net = Net()
    net.collect_params().initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()
    dx1 = x.grad
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x)
    out2.backward()
    mx.test_utils.assert_almost_equal(dx1.asnumpy(), x.grad.asnumpy(), rtol=1e-5, atol=1e-6)
    mx.test_utils.assert_almost_equal(out1.asnumpy(), out2.asnumpy(), rtol=1e-5, atol=1e-6)


@with_seed()
def test_flatten_slice_after_conv():
    data = mx.symbol.Variable('data')
    weight = mx.symbol.Variable('weight')
    bias = mx.symbol.Variable('bias')
    conv1= mx.symbol.Convolution(data = data, weight=weight, bias=bias, name='conv1', num_filter=64, kernel=(3,3), stride=(1,1))
    flatten1 = mx.symbol.flatten(data = conv1)
    slice1 = mx.symbol.slice(data = flatten1, begin=0, end=1)

    shape = (2, 16, 16, 16)
    val = np.random.rand(2, 16, 16, 16).astype(np.float32)
    exe = slice1.simple_bind(Context.default_ctx, data=shape)
    exe.arg_arrays[0][:] = val
    exe.arg_arrays[1][:] = np.random.normal(size=exe.arg_arrays[1].shape)
    exe.arg_arrays[2][:] = np.random.normal(size=exe.arg_arrays[2].shape)
    p = exe.forward(is_train=False)
    p[0].wait_to_read()
    print(p[0])


def test_mkldnn_sum_inplace_with_cpu_layout():

    x_shape = (32, 3, 224, 224)
    x_npy = np.ones(x_shape)
    y_shape = (32, 32, 222, 222)
    y_npy = np.ones(y_shape)
    x = mx.sym.Variable("x")
    y = mx.sym.Variable("y")
    z = mx.symbol.Convolution(data=x, num_filter=32, kernel=(3, 3))
    z = mx.sym.add_n(z, y)
    exe = z.simple_bind(ctx=mx.cpu(), x=x_shape, y=y_shape)
    out = exe.forward(is_train=False, x=x_npy, y=y_npy)[0]
    assert_almost_equal(out[0].asnumpy()[0, 0, 0], 1.0)


@with_seed()
def test_batchnorm():
    def check_batchnorm_training(stype):
        for shape in [(2, 3), (2, 3, 2, 2)]:
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


@with_seed()
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


@with_seed()
def test_pooling():
    def check_pooling_training(stype):
        for shape in [(3, 3, 10), (3, 3, 20, 20)]:
            data_tmp = np.random.normal(-0.1, 0.1, size=shape)
            data = mx.symbol.Variable('data', stype=stype)
            in_location = [mx.nd.array(data_tmp).tostype(stype)]

            if np.array(shape).shape[0] == 3:
                test = mx.symbol.Pooling(data=data, kernel=(3,), stride=(2), pool_type='avg')
            elif np.array(shape).shape[0] == 4:
                test = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='avg')
            else:
                return 0
            check_numeric_gradient(test, in_location, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_pooling_training(stype)


@with_seed()
def test_activation():
    def check_activation_training(stype):
        for shape in [(2, 3, 3), (2, 3, 2, 2)]:
            data_tmp = np.random.normal(-0.1, 1, size=shape)

            data = mx.symbol.Variable('data', stype=stype)
            in_location = [mx.nd.array(data_tmp).tostype(stype)]

            test = mx.symbol.Activation(data, act_type="relu")
            check_numeric_gradient(test, in_location, numeric_eps=1e-5, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_activation_training(stype)


@with_seed()
def test_convolution():
    def check_convolution_training(stype):
        for shape in [(3, 3, 10), (3, 3, 10, 10)]:
            data_tmp = np.random.normal(-0.1, 1, size=shape)
            data = mx.symbol.Variable('data', stype=stype)

            if np.array(shape).shape[0] == 3:
                test = mx.symbol.Convolution(data=data, kernel=(3,), stride=(2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(4, 3, 3))
            elif np.array(shape).shape[0] == 4:
                test = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2, 2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(4, 3, 3, 3))
            else:
                return 0
            bias_tmp = np.random.normal(0.1, 0.1, size=(4,))
            in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(weight_tmp).tostype(stype),
                           mx.nd.array(bias_tmp).tostype(stype)]
            check_numeric_gradient(test, in_location, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_convolution_training(stype)


@with_seed()
@unittest.skip("Flaky test https://github.com/apache/incubator-mxnet/issues/12579")
def test_Deconvolution():
    def check_Deconvolution_training(stype):
        for shape in [(3, 3, 10), (3, 3, 10, 10)]:
            data_tmp = np.random.randint(256, size=shape)
            data = mx.symbol.Variable('data', stype=stype)

            if np.array(shape).shape[0] == 3:
                test = mx.symbol.Deconvolution(data=data, kernel=(3,), stride=(2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(3, 4, 3))
            elif np.array(shape).shape[0] == 4:
                test = mx.symbol.Deconvolution(data=data, kernel=(3, 3), stride=(2, 2), num_filter=4)
                weight_tmp = np.random.normal(-0.1, 0.1, size=(3, 4, 3, 3))
            else:
                return 0
            bias_tmp = np.random.normal(0.1, 0.1, size=(4,))
            in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(weight_tmp).tostype(stype),
                           mx.nd.array(bias_tmp).tostype(stype)]
            check_numeric_gradient(test, in_location, numeric_eps=1e-2, rtol=0.16, atol=1e-4)

    stypes = ['row_sparse', 'default']
    for stype in stypes:
        check_Deconvolution_training(stype)


@with_seed()
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


@with_seed()
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
        exec1 = out1.bind(mx.cpu(), args={'data': input_data})
        exec1.forward()[0].wait_to_read()
        ndarr = exec1.outputs[0][0][0][0]
        nparr = ndarr.asnumpy()
        assert_almost_equal(nparr, true_output, rtol=1e-5, atol=1e-5)

    softmax_forward(mx.nd.array([[[[-1e30,-1e30]]]]), np.array([1.0,1.0]))
    softmax_forward(mx.nd.array([[[[1e30,1e30]]]]), np.array([1.0,1.0]))
    softmax_forward(mx.nd.array([[[[-3.4e38,-3.4e38]]]]), np.array([1.0,1.0]))
    softmax_forward(mx.nd.array([[[[3.4e38,3.4e38]]]]), np.array([1.0,1.0]))

@with_seed()
def test_non_mkldnn_fcomputeex():
    # test special case where MKLDNN formatted NDArray feeds into non-mkldnn fcomputeex operator
    # conv is example where MKLDNN NDArray is created from regular NDArrays
    # CustomOps is example of non-mkldnn fcomputeex operator

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
    exec1 = custom.bind(mx.cpu(), args={'data': mx.nd.ones([10,3,96,96]), 'conv_weight': mx.nd.ones([8,3,5,5])})
    exec1.forward()[0].wait_to_read()

@with_seed()
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


# This test case is contributed by @awsbillz in https://github.com/apache/incubator-mxnet/issues/14766
@with_seed()
def test_reshape_transpose_6d():
    class Reshape2D(gluon.HybridBlock):
        def __init__(self, factor):
            super(Reshape2D, self).__init__()
            self._factors = (int(factor),) * 2

        def hybrid_forward(self, F, x):
            f1, f2 = self._factors
                                                          # (N, f1*f2*C, H, W)
            x = F.reshape(x, (0, -4, -1, f1 * f2, 0, 0))  # (N, C, f1*f2, H, W)
            x = F.reshape(x, (0, 0, -4, f1, f2, 0, 0))    # (N, C, f1, f2, H, W)
            x = F.transpose(x, (0, 1, 4, 2, 5, 3))        # (N, C, H, f1, W, f2)
            x = F.reshape(x, (0, 0, -3, -3))              # (N, C, H*f1, W*f2)
            return x


    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv1 = nn.Conv2D(8, kernel_size=5)
                self.reshape2D = Reshape2D(2)

        def hybrid_forward(self, F, x):
            x = self.conv1(x)
            x = self.reshape2D(x)
            return x

    net = Net()
    net.initialize(mx.init.Xavier(), ctx=mx.cpu())
    net.hybridize()
    data = mx.nd.random_normal(shape=(1, 3, 600, 600))
    output = net(data)
    a = output.asnumpy()

@with_seed()
def test_weight_async_reorder():
    data = mx.sym.Variable("data")
    w1 = mx.sym.Variable("1_weight")
    w2 = mx.sym.Variable("2_weight")
    conv1 = mx.sym.Convolution(data=data, weight=w1 + w1, num_filter=32, no_bias=True, kernel=(3, 3))
    conv2 = mx.sym.Convolution(data=conv1, weight=w2 + w2, num_filter=32, no_bias=True, kernel=(1, 1))
    mod = Module(symbol=conv2, label_names=None, context=mx.current_context())
    mod.bind(for_training=False, data_shapes=[('data', (10, 16, 50, 50))])
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    data = [mx.random.uniform(-1.0, 1.0, shape=(10, 16, 50, 50), ctx=mx.current_context())]
    batch=mx.io.DataBatch(data, [])
    for i in range(2):
        mod.forward(batch, is_train=False)
        for output in mod.get_outputs():
            output.wait_to_read()

if __name__ == '__main__':
    install.test_mkldnn_install()
