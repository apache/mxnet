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

import mxnet as mx
from mxnet.gluon import HybridBlock, nn
import numpy as np
import onnxruntime as rt
from mxnet.test_utils import assert_almost_equal
import pytest
import tempfile


def def_model(namespace, op_name, dummy_input=False, **params):
    class Model(HybridBlock):
        def __init__(self, **kwargs):
            super(Model, self).__init__(**kwargs)

        def forward(self, *inputs):
            names = op_name.split('.')
            func = getattr(namespace, names[-1])
            if dummy_input:
                return func(**params), inputs[0]
            else:
                return func(*inputs, **params)
    return Model

def def_model_from_func(func, dummy_input=False, **params):
    class Model(HybridBlock):
        def __init__(self, **kwargs):
            super(Model, self).__init__(**kwargs)

        def forward(self, *inputs):
            if dummy_input:
                return func(**params), inputs[0]
            else:
                return func(*inputs, **params)
    return Model

def op_export_test(model_name, Model, inputs, tmp_path, dummy_input=False, onnx_map=None, mx_map=None, rtol=None, atol=None):
    def export_to_onnx(model, model_name, inputs):
        model_path = '{}/{}'.format(tmp_path, model_name)
        model.export(model_path, epoch=0)
        sym_file = '{}-symbol.json'.format(model_path)
        params_file = '{}-0000.params'.format(model_path)
        onnx_file = '{}/{}.onnx'.format(tmp_path, model_name)
        mx.onnx.export_model(sym_file, params_file, [inp.shape for inp in inputs],
                             [inp.dtype for inp in inputs], onnx_file)
        return onnx_file

    def onnx_rt(onnx_file, inputs):
        sess = rt.InferenceSession(onnx_file)
        dtype_0 = inputs[0].asnumpy().dtype
        input_dict = dict((sess.get_inputs()[i].name, inputs[i].asnumpy()) for i in range(len(inputs)))
        pred = sess.run(None, input_dict)
        return pred

    # create a new model 
    model = Model()
    model.initialize(ctx=mx.cpu(0))
    model.hybridize()
    pred_mx = model(*inputs)

    # this is for ops such as mx.np.concatenate
    if isinstance(inputs[0], tuple):
        inputs = list(inputs[0])
    onnx_file = export_to_onnx(model, model_name, inputs)
    pred_onx = onnx_rt(onnx_file, inputs)
    if dummy_input:
        pred_mx = pred_mx[0]
    if isinstance(pred_mx, list):
        for i in range(len(pred_mx)):
            pred_onx_i = onnx_map(pred_onx[i]) if onnx_map else pred_onx[i]
            pred_mx_i = mx_map(pred_mx[i]) if mx_map else pred_mx[i]
            assert_almost_equal(pred_onx_i, pred_mx_i, equal_nan=True, rtol=rtol, atol=atol)
    else:
        pred_onx = onnx_map(pred_onx[0]) if onnx_map else pred_onx[0]
        pred_mx = mx_map(pred_mx) if mx_map else pred_mx
        assert_almost_equal(pred_onx, pred_mx, equal_nan=True, rtol=rtol, atol=atol)


def test_onnx_export_np_abs(tmp_path):
    M = def_model(mx.np, 'abs')
    x = mx.np.array([[-2, -1], [0, 99]], dtype='float32')
    op_export_test('abs', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'float16', 'int32', 'int64'])
@pytest.mark.parametrize('params', [[(0, 1), (2,3), (1, 1)],
                                    [(None, 1), (2, None), None],
                                    [(0, 0, 0), (None, 4, 5), (None, 1, 2)]])
def test_onnx_export_npx_slice(tmp_path, dtype, params):
    M = def_model(mx.npx, 'slice', begin=params[0], end=params[1], step=params[2])
    x = mx.np.arange(start=0, stop=60, dtype=dtype).reshape((3, 4, 5))
    op_export_test('slice', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(1), (1,2), (2,3,4), (5,6,7)])
def test_onnx_export_np_zeros(tmp_path, dtype, shape):
    M = def_model(mx.np, 'zeros', shape=shape, dtype=dtype, dummy_input=True)
    x = mx.np.array([1])
    op_export_test('zeros', M, [x], tmp_path, dummy_input=True)


@pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(1), (1,2), (2,3,4), (5,6,7)])
def test_onnx_export_np_ones(tmp_path, dtype, shape):
    M = def_model(mx.np, 'ones', shape=shape, dtype=dtype, dummy_input=True)
    x = mx.np.array([0])
    op_export_test('ones', M, [x], tmp_path, dummy_input=True)


@pytest.mark.parametrize('dtype', [None, 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1), (1,2), (2,3,4), (5,6,7)])
def test_onnx_export_np_zeros_like(tmp_path, dtype, shape):
    M = def_model(mx.np, 'zeros_like', dtype=dtype)
    x = mx.np.random.uniform(0, 1, shape, dtype='float32')
    op_export_test('zeros_like', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', [None, 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1), (1,2), (2,3,4), (5,6,7)])
def test_onnx_export_np_ones_like(tmp_path, dtype, shape):
    M = def_model(mx.np, 'ones_like', dtype=dtype)
    x = mx.np.random.uniform(0, 1, shape, dtype='float32')
    op_export_test('ones_like', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', [None, 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1), (1,2), (2,3,4), (5,6,7)])
@pytest.mark.parametrize('fill_value', [0, 1, -1, 12.34])
def test_onnx_export_np_full_like(tmp_path, dtype, shape, fill_value):
    M = def_model(mx.np, 'full_like', dtype=dtype, fill_value=fill_value)
    x = mx.np.random.uniform(0, 1, shape, dtype='float32')
    op_export_test('full_like', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("axis", [None,0,1])
@pytest.mark.parametrize("start", [0, 0.5, 1])
@pytest.mark.parametrize("step", [0.01, 0.1, 0.5, 1])
@pytest.mark.parametrize("test_data", [ mx.np.random.uniform(0, 1, (10,20)), [[0,1,2,3,4,5],[4,5,6,7,8,9],[8,9,10,11,12,13]]])
def test_onnx_export_npx_arange_like(tmp_path, dtype, axis, start, step, test_data):
    M = def_model(mx.npx, 'arange_like', axis=axis, start=start, step=step)
    x = mx.np.array(test_data, dtype=dtype)
    op_export_test('arange_like', M, [x], tmp_path)


@pytest.mark.parametrize("params", [[0, 2, 1], [0, 50, 0.25], [-100, 100, 0.5], [5, None, 1], [-5, None, -1]])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_onnx_export_np_arange(tmp_path, dtype, params):
    start, stop, step = params[0], params[1], params[2]
    if "int" in dtype:
        start = int(start)
        stop = int(stop) if stop != None else None
        step = int(step)
        if step == 0:
            step = 1
    M = def_model(mx.np, 'arange', dummy_input=True, start=start, stop=stop, step=step, dtype=dtype)
    x = mx.np.array([1], dtype='float32')
    op_export_test('arange', M, [x], tmp_path, dummy_input=True)


@pytest.mark.parametrize('dtype', ['float32'])
def test_onnx_export_npx_layer_norm(tmp_path, dtype):
    x = mx.np.random.uniform(1, 2, (3, 4, 5), dtype=dtype)
    axes = list(range(np.shape(np.shape(x))[0]))
    axes.append(-1)
    for axis in axes:
        M = def_model(mx.npx, 'layer_norm', axis=axis)
        gamma = mx.np.random.uniform(0, 1, [np.shape(x)[axis]], dtype=dtype)
        beta = mx.np.random.uniform(0, 1, [np.shape(x)[axis]], dtype=dtype)
        op_export_test('layer_norm', M, [x, gamma, beta], tmp_path)


@pytest.mark.skip(reason='broadcast_axis is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32'])
def test_onnx_export_broadcast_axis(tmp_path, dtype):
    M1 = def_model('broadcast_axis', axis=(0, 2), size=(3, 4))
    M2 = def_model('broadcast_axis', axis=(0, 2), size=(1, 5))
    x1 = mx.nd.array([[[1], [2]]], dtype=dtype)
    op_export_test('broadcast_axis_1', M1, [x1], tmp_path)
    op_export_test('broadcast_axis_2', M2, [x1], tmp_path)
    M3 = def_model('broadcast_axis', axis=(1, 4), size=(3, 5))
    x2 = mx.nd.ones((1, 1, 3, 1, 1, 1), dtype=dtype)
    op_export_test('broadcast_axis_3', M3, [x2], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
def test_onnx_export_npx_sequence_mask(tmp_path, dtype):
    M1 = def_model(mx.npx, 'sequence_mask', use_sequence_length=True, axis=1, value=-5)
    M2 = def_model(mx.npx, 'sequence_mask', use_sequence_length=True, axis=0, value=-99)
    x = mx.np.array([[[[  1.,   2.,   3.,  3.5]],
                      [[  4.,   5.,   6.,  6.5]]],
                     [[[  7.,   8.,   9.,  9.5]],
                      [[ 10.,  11.,  12., 12.5]]],
                     [[[ 13.,  14.,  15., 15.5]],
                      [[ 16.,  17.,  18., 18.5]]]], dtype=dtype)
    seq_len1 = mx.np.array([1, 2, 1], dtype=dtype)
    seq_len2 = mx.np.array([1, 2], dtype=dtype)
    op_export_test('sequence_mask_1', M1, [x, seq_len1], tmp_path)
    op_export_test('sequence_mask_2', M2, [x, seq_len2], tmp_path)


@pytest.mark.skip(reason='This op will be adopted in MXNet 2.0 soon')
@pytest.mark.parametrize('dtype', ['float32'])
def test_onnx_export_contrib_interleaved_matmul_selfatt_qk(tmp_path, dtype):
    M1 = def_model('contrib.interleaved_matmul_selfatt_qk', heads=3)
    x1 = mx.nd.random.uniform(0, 1, (3, 3, 3*3*3), dtype=dtype)
    op_export_test('contrib_interleaved_matmul_selfatt_qk_1', M1, [x1], tmp_path)
    M2 = def_model('contrib.interleaved_matmul_selfatt_qk', heads=5)
    x2 = mx.nd.random.uniform(0, 1, (7, 5, 4*5*6), dtype=dtype)
    op_export_test('contrib_interleaved_matmul_selfatt_qk_2', M2, [x2], tmp_path)


@pytest.mark.skip(reason='This op will be adopted in MXNet 2.0 soon')
@pytest.mark.parametrize('dtype', ['float32'])
def test_onnx_export_contrib_interleaved_matmul_selfatt_valatt(tmp_path, dtype):
    M = def_model('contrib.interleaved_matmul_selfatt_valatt', heads=6)
    x = mx.nd.random.uniform(0, 1, (4, 5, 6*7*3), dtype=dtype)
    att = mx.nd.random.uniform(0, 1, (5*6, 4, 4), dtype=dtype)
    op_export_test('contrib_interleaved_matmul_selfatt_valatt', M, [x, att], tmp_path)


@pytest.mark.skip(reason='slice_axis is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32'])
def test_onnx_export_slice_axis(tmp_path, dtype):
    x = mx.nd.array([[  1.,   2.,   3.,   4.],
                     [  5.,   6.,   7.,   8.],
                     [  9.,  10.,  11.,  12.]], dtype=dtype)
    M1 = def_model('slice_axis', axis=0, begin=1, end=3)
    M2 = def_model('slice_axis', axis=0, begin=1, end=None)
    M3 = def_model('slice_axis', axis=1, begin=-3, end=-1)
    M4 = def_model('slice_axis', axis=-1, begin=-3, end=None)
    op_export_test('slice_axis_1', M1, [x], tmp_path)
    op_export_test('slice_axis_2', M2, [x], tmp_path)
    op_export_test('slice_axis_3', M3, [x], tmp_path)
    op_export_test('slice_axis_4', M4, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_onnx_export_npx_reshape(tmp_path, dtype):
    x = mx.np.ones((2, 3, 4, 5, 6), dtype=dtype)
    M1 = def_model(mx.npx, 'reshape', newshape=(6, 1, 5, -1))
    op_export_test('reshape', M1, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_onnx_export_np_reshape(tmp_path, dtype):
    x = mx.np.ones((2, 3, 4, 5, 6), dtype=dtype)
    M1 = def_model(mx.np, 'reshape', newshape=(6, 1, 5, -1))
    op_export_test('reshape', M1, [x], tmp_path)


@pytest.mark.skip(reason='Reshape is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_onnx_export_reshape(tmp_path, dtype):
    x = mx.np.ones((2, 3, 4, 5, 6), dtype=dtype)
    M1 = def_model(mx.npx, 'reshape', newshape=(6, 1, 0, -1))
    op_export_test('reshape_1', M1, [x], tmp_path)
    M2 = def_model(mx.np, 'reshape', newshape=(3, -1, 0, 0), reverse=True)
    op_export_test('reshape_2', M2, [x], tmp_path)
    M3 = def_model(mx.np, 'reshape', newshape=(5, 1, 1, 1, 1, 0 -1, 0), reverse=True)
    op_export_test('reshape_3', M3, [x], tmp_path)
    M4 = def_model(mx.np, 'reshape', newshape=(-3, -1))
    op_export_test('reshape_4', M4, [x], tmp_path)


@pytest.mark.skip(reason='Reshape is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_onnx_export_reshape_special_cases(tmp_path, dtype):
    x1 = mx.nd.ones((8, 9), dtype=dtype)
    M1 = def_model('reshape', shape=(0, -4, 1, -1))
    op_export_test('reshape_spec_1', M1, [x1], tmp_path)

    x2 = mx.nd.ones((8, 9, 10), dtype=dtype)

    M2 = def_model('reshape', shape=(0, -4, 3, -1, 10))
    op_export_test('reshape_spec_2', M2, [x2], tmp_path)
    M3 = def_model('reshape', shape=(-4, 2, -1, 10, 9))
    op_export_test('reshape_spec_3', M3, [x2], tmp_path)

    M4 = def_model('reshape', shape=(-3, 0))
    op_export_test('reshape_spec_4', M4, [x2], tmp_path)

    x3 = mx.nd.ones((1, 2, 3, 4, 5, 6), dtype=dtype)
    M5 = def_model('reshape', shape=(0, 0, -3, -3))
    op_export_test('reshape_spec_5', M5, [x3], tmp_path)

    x4 = mx.nd.ones((5, 8, 6, 7), dtype=dtype)
    M6 = def_model('reshape', shape=(0, -4, -1, 4, 0, 0))
    op_export_test('reshape_spec_6', M6, [x4], tmp_path)

    x5 = mx.nd.ones((2, 3, 4, 5, 6), dtype=dtype)
    M7 = def_model('reshape', shape=(0, 0, -4, 2, 2, 0, 0))
    op_export_test('reshape_spec_7', M7, [x5], tmp_path)

    x6 = mx.nd.ones((8, 7, 6, 5), dtype=dtype)
    M8 = def_model('reshape', shape=(-4, 1, -1, 0, 0, 0))
    op_export_test('reshape_spec_8', M8, [x6], tmp_path)

    x7 = mx.nd.ones((1000, 2, 3), dtype=dtype)
    M9 = def_model('reshape', shape=(-4, 1, 1000, 0, 0))
    op_export_test('reshape_spec_9', M9, [x7], tmp_path)

    x8 = mx.nd.ones((3, 96, 5), dtype=dtype)
    M10 = def_model('reshape', shape=(0, -4, 12, -1, 0))
    op_export_test('reshape_spec_10', M10, [x8], tmp_path)

    x9 = mx.nd.ones((3, 96, 5), dtype=dtype)
    M11 = def_model('reshape', shape=(0, -4, 16, -1, 0))
    op_export_test('reshape_spec_11', M11, [x9], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64'])
def test_onnx_export_npx_embedding(tmp_path, dtype):
    x = mx.np.array([[ 1.,  3.],
                     [ 0.,  2.]], dtype=dtype)
    y = mx.np.array([[  0.,   1.,   2.,   3.,   4.],
                     [  5.,   6.,   7.,   8.,   9.],
                     [ 10.,  11.,  12.,  13.,  14.],
                     [ 15.,  16.,  17.,  18.,  19.]], dtype=dtype)
    M = def_model(mx.npx, 'embedding', input_dim=4, output_dim=5)
    op_export_test('embedding', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('num_hidden', [1, 2, 7, 10, 20])
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_onnx_export_npx_fully_connected(tmp_path, dtype, num_hidden, no_bias, flatten):
    M = def_model(mx.npx, 'fully_connected', num_hidden=num_hidden, no_bias=no_bias, flatten=flatten)
    x = mx.np.random.uniform(-0.5, 0.5, (3, 4, 5))
    if (flatten):
        weight = mx.np.random.uniform(0, 1, (num_hidden, 4*5))
    else:
        weight = mx.np.random.uniform(0, 1, (num_hidden, 5))
    args = [x, weight]
    if not no_bias:
        args.append(mx.np.random.uniform(0,1,(num_hidden,)))
    op_export_test('fully_connected', M, args, tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float16'])
@pytest.mark.parametrize('shape', [(1,), (3,), (4, 5), (3, 4, 5)])
@pytest.mark.parametrize('act_type', ['elu', 'leaky', 'prelu', 'selu', 'gelu'])
def test_onnx_export_npx_leaky_relu(tmp_path, dtype, shape, act_type):
    M = def_model(mx.npx, 'leaky_relu', act_type='leaky')
    x = mx.np.random.uniform(-0.5, 0.5, shape, dtype=dtype)
    op_export_test('leaky_relu', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'float16', 'int32', 'int64'])
def test_onnx_export_np_concatenate(tmp_path, dtype):
    x = mx.np.array([[1,1],[2,2]], dtype=dtype)
    y = mx.np.array([[3,3],[4,4],[5,5]], dtype=dtype)
    z = mx.np.array([[6,6],[7,7],[8,8]], dtype=dtype)
    M1 = def_model(mx.np, 'concatenate', axis=0)
    M2 = def_model(mx.np, 'concatenate', axis=1)
    M3 = def_model(mx.np, 'concatenate')
    op_export_test('concatenate_1', M1, [(x, y, z)], tmp_path)
    op_export_test('concatenate_2', M2, [(y, z)], tmp_path)
    op_export_test('concatenate_3', M2, [(y, z)], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float16'])
@pytest.mark.parametrize('shape', [(1,), (3,), (4, 5), (3, 4, 5)])
@pytest.mark.parametrize('act_type', ['tanh', 'relu', 'sigmoid', 'softrelu', 'softsign'])
def test_onnx_export_npx_activation(tmp_path, dtype, shape, act_type):
    M = def_model(mx.npx, 'activation', act_type=act_type)
    x = mx.np.random.uniform(-0.5, 0.5, shape, dtype=dtype)
    op_export_test('activation', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('axes', [None, [1,0,2]])
def test_onnx_export_np_transpose(tmp_path, dtype, axes):
    if axes != None:
        M = def_model(mx.np, 'transpose', axes=axes)
    else:
        M = def_model(mx.np, 'transpose')
    x = mx.np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=dtype)
    op_export_test('transpose', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_onnx_export_np_expand_dims(tmp_path, dtype, axis):
    M = def_model(mx.np, 'expand_dims', axis=axis)
    x = mx.np.random.uniform(0, 1, (2,3,4), dtype=dtype)
    op_export_test('expand_dims', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_onnx_export_np_add(tmp_path, dtype):
    M = def_model(mx.np, 'add')
    x = mx.np.array([[1,1,1],[1,1,1]], dtype=dtype)
    y = mx.np.array([[0],[1]], dtype=dtype)
    op_export_test('add', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
def test_onnx_export_np_minimum(tmp_path, dtype):
    M = def_model(mx.np, 'minimum')
    if 'int' in dtype:
        x = mx.np.random.randint(0, 1000, (4, 5, 6), dtype=dtype)
        y = mx.np.random.randint(0, 1000, (4, 5, 6), dtype=dtype)
    else:
        x = mx.np.random.uniform(0, 1000, (4, 5, 6), dtype=dtype)
        y = mx.np.random.uniform(0, 1000, (4, 5, 6), dtype=dtype)
    op_export_test('minimum', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
def test_onnx_export_np_maximum(tmp_path, dtype):
    M = def_model(mx.np, 'maximum')
    if 'int' in dtype:
        x = mx.np.random.randint(0, 1000, (4, 5, 6), dtype=dtype)
        y = mx.np.random.randint(0, 1000, (4, 5, 6), dtype=dtype)
    else:
        x = mx.np.random.uniform(0, 1000, (4, 5, 6), dtype=dtype)
        y = mx.np.random.uniform(0, 1000, (4, 5, 6), dtype=dtype)
    op_export_test('maximum', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('axis', [0, 1, 2, -1])
def test_onnx_export_npx_stack(tmp_path, dtype, axis):
    M = def_model(mx.np, 'stack', axis=axis)
    if 'int' in dtype:
        x = mx.np.random.randint(0, 10*9, (3,4,5), dtype=dtype)
        y = mx.np.random.randint(0, 10*9, (3,4,5), dtype=dtype)
    else:
        x = mx.np.random.normal(0, 10*9, (3,4,5), dtype=dtype)
        y = mx.np.random.normal(0, 10*9, (3,4,5), dtype=dtype)
    op_export_test('stack', M, [(x, y)], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (1, 2, 3, 2, 1)])
@pytest.mark.parametrize('p', [0, 0.1, 0.5, 1])
def test_onnx_export_npx_dropout(tmp_path, dtype, shape, p):
    x = mx.np.random.uniform(-100, 100, size=shape).astype(dtype)
    M = def_model(mx.npx, 'dropout', p=p)
    op_export_test('dropuout', M, [x], tmp_path)


@pytest.mark.parametrize('src_dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('dst_dtype', ['bool', 'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'])
@pytest.mark.parametrize('shape', [(2,3), (4,5,6)])
def test_onnx_export_npx_cast(tmp_path, src_dtype, dst_dtype, shape):
    M = def_model(mx.npx, 'cast', dtype=dst_dtype)
    x = mx.np.random.uniform(0, 1, size=shape, dtype=src_dtype)
    op_export_test('cast', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('temperature', [None, .1, 1., 10.])
def test_onnx_export_softmax(tmp_path, dtype, temperature):
    x = mx.np.random.uniform(0, 1, (4, 5, 6), dtype=dtype)
    M1 = def_model(mx.npx, 'softmax')
    op_export_test('softmax_1', M1, [x], tmp_path)
    l2 = mx.np.random.uniform(0, 4, (5, 6)).astype('int32')
    M2 = def_model(mx.npx, 'softmax', use_length=True, axis=0, temperature=temperature)
    op_export_test('softmax_2', M2, [x, l2], tmp_path)
    M3 = def_model(mx.npx, 'softmax', use_length=True, axis=-1, temperature=temperature)
    # note that the axis==-1 case uses negative value masking + ONNX softmax
    # when valid_len==0 the masked values will NOT be 0
    l3 = mx.np.random.uniform(1, 6, (4, 5)).astype('int32')
    op_export_test('softmax_3', M3, [x, l3], tmp_path)
    M4 = def_model(mx.npx, 'softmax', use_length=True, axis=1, temperature=temperature)
    l4 = mx.np.random.uniform(0, 5, (4, 6)).astype('int32')
    op_export_test('softmax_4', M4, [x, l4], tmp_path)


@pytest.mark.skip(reason='reverse is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('axis', [0, 1, 2, 3])
def test_onnx_export_reverse(tmp_path, dtype, axis):
    x = mx.np.arange(0, 120, dtype=dtype).reshape((2, 3, 4, 5))
    M = def_model('reverse', axis=axis)
    op_export_test('reverse', M, [x], tmp_path)


@pytest.mark.skip(reason='this version of repeat is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('axis', [None, 0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize('repeats', [2, 1, 3])
def test_onnx_export_repeat(tmp_path, dtype, axis, repeats):
    x = mx.np.arange(0, 27, dtype=dtype).reshape((3, 3, 3))
    M = def_model(mx.np, 'repeat', axis=axis, repeats=repeats)
    op_export_test('repeat', M, [x], tmp_path)


@pytest.mark.skip(reason='BilinearResize2D is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1, 3, 224, 224), (2, 2, 5, 8), (2, 4, 17, 23)])
@pytest.mark.parametrize('params', [{'height': 7, 'width': 13},
                                    {'height': 10, 'width': 16},
                                    {'height': 3, 'width': 5},
                                    {'height': 2, 'width': 4},
                                    {'scale_height': 3, 'scale_width': 2},
                                    {'scale_height': 1.7, 'scale_width': 2.3},
                                    {'scale_height': 0.5, 'scale_width': 0.6},
                                    {'scale_height': 0.8, 'scale_width': 0.13},
                                    {'scale_height': 2.5, 'scale_width': 0.5},
                                    {'scale_height': 3, 'scale_width': 0.2},
                                    ])
def test_onnx_export_contrib_BilinearResize2D(tmp_path, dtype, shape, params):
    x = mx.random.uniform(0, 1, shape)
    M = def_model('contrib.BilinearResize2D', **params)
    op_export_test('contrib_BilinearResize2D', M, [x], tmp_path)


@pytest.mark.parametrize('topk', [-1, 2, 3, 4])
@pytest.mark.parametrize('valid_thresh', [0.3, 0.4, 0.8])
@pytest.mark.parametrize('overlap_thresh', [0.4, 0.7, 1.0])
def test_onnx_export_npx_box_nms(tmp_path, topk, valid_thresh, overlap_thresh):
    # Note that ONNX NMS op only supports float32

    # Also note that onnxruntime's nms has slightly different implementation in handling
    # overlaps and score ordering when certain boxes are suppressed than that of mxnet
    # the following test tensors are manually tweaked to avoid such diferences
    # The purpose of theses tests cases are to show that the high level conversion logic is
    # laid out correctly

    A = mx.np.array([[
                    [[[[0.5, 0.1, 0.1, 0.2, 0.2],
                    [0.4, 0.1, 0.1, 0.2, 0.2],
                    [0.7, 0.5, 0.5, 0.9, 0.9],
                    [0.8, 0.1, 0.9, 0.11, 0.91],
                    [0.001, 0.01, 0.01, 0.02, 0.02]]]],

                    [[[[0.5, 0.1, 0.1, 0.2, 0.2],
                    [0.4, 0.1, 0.1, 0.2, 0.2],
                    [0.7, 0.5, 0.5, 0.9, 0.9],
                    [0.8, 0.1, 0.9, 0.11, 0.91],
                    [0.001, 0.01, 0.01, 0.02, 0.02]]]],

                    [[[[0.4, 0.1, 0.1, 0.2, 0.2],
                    [0.3, 0.1, 0.1, 0.2, 0.2],
                    [0.7, 0.5, 0.5, 0.9, 0.9],
                    [0.8, 0.1, 0.9, 0.11, 0.91],
                    [0.001, 0.01, 0.01, 0.02, 0.02]]]],
                    ]])
    M = def_model(mx.npx, 'box_nms', coord_start=1, force_suppress=True,
                  overlap_thresh=overlap_thresh, valid_thresh=valid_thresh, score_index=0,
                  topk=topk, in_format='corner', out_format='corner')
    op_export_test('box_nms_manual_coner', M, [A], tmp_path)
    
    B = mx.np.array([
                    [[[[0.7, 0.5, 0.5, 0.2, 0.2],
                    [0.6, 0.48, 0.48, 0.2, 0.2],
                    [0.8, 0.76, 0.76, 0.2, 0.2],
                    [0.9, 0.7, 0.7, 0.2, 0.2],
                    [0.001, 0.5, 0.1, 0.02, 0.02]]]],

                    [[[[0.5, 0.2, 0.2, 0.2, 0.2],
                    [0.6, 0.4, 0.4, 0.21, 0.21],
                    [0.7, 0.5, 0.5, 0.9, 0.9],
                    [0.8, 0.1, 0.9, 0.01, 0.01],
                    [0.001, 0.6, 0.1, 0.02, 0.02]]]],
                    ])
    M = def_model(mx.npx, 'box_nms', coord_start=1, force_suppress=True,
                  overlap_thresh=overlap_thresh, valid_thresh=valid_thresh, score_index=0,
                  topk=topk, in_format='center', out_format='center')
    op_export_test('box_nms_manual_center', M, [B], tmp_path)


@pytest.mark.skip(reason='greater_scalar is deprecated in MXNet 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("scalar", [0., 0.1, 0.5, 1., 5, 555.])
def test_onnx_export_greater_scalar(tmp_path, dtype, scalar):
    if 'int' in dtype:
        scalar = int(scalar)
        x = mx.np.arange(0, 12, dtype=dtype).reshape((3, 4))
    else:
        x = mx.np.random.uniform(0, 9999, (5,10), dtype=dtype)
    M = def_model(mx.np, 'greater', scalar=scalar)
    op_export_test('greater', M, [x], tmp_path)


@pytest.mark.skip(reason='lesser_scalar is deprecated in MXNet 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("scalar", [0., 0.1, 0.5, 1., 5, 555.])
def test_onnx_export_lesser_scalar(tmp_path, dtype, scalar):
    if 'int' in dtype:
        scalar = int(scalar)
        x = mx.nd.arange(0, 12, dtype=dtype).reshape((3, 4))
    else:
        x = mx.random.uniform(0, 9999, (5,10), dtype=dtype)
    M = def_model('_internal._lesser_scalar', scalar=scalar)
    op_export_test('_internal._lesser_scalar', M, [x], tmp_path)


@pytest.mark.skip(reason='equal_scalar is deprecated in MXNet 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("scalar", [0., 0.1, 0.5, 1., 5, 555.])
def test_onnx_export_equal_scalar(tmp_path, dtype, scalar):
    if 'int' in dtype:
        scalar = int(scalar)
        x = mx.nd.arange(0, 12, dtype=dtype).reshape((3, 4))
    else:
        x = mx.random.uniform(0, 9999, (5,10), dtype=dtype)
    M = def_model('_internal._equal_scalar', scalar=scalar)
    op_export_test('_internal._equal_scalar', M, [x], tmp_path)


@pytest.mark.parametrize('op', ['equal', 'not_equal', 'greater', 'less', 'greater_equal', 'less_equal'])
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(2,3), (4,5,6)])
def test_onnx_export_np_comparison(tmp_path, op, dtype, shape):
    M = def_model(mx.np, op)
    x = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    y = mx.np.random.uniform(-100, 100, shape[1:]).astype(dtype)
    op_export_test(op, M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(5,), (3,3), (10,2), (20,30,40)])
def test_onnx_export_np_where(tmp_path, dtype, shape):
    M = def_model(mx.np, 'where')
    x = mx.np.zeros(shape, dtype=dtype)
    y = mx.np.ones(shape, dtype=dtype)
    cond = mx.np.random.randint(low=0, high=1, size=shape, dtype='int32')
    op_export_test('where', M, [cond, x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int64'])
@pytest.mark.parametrize('axis', [0, 2, -1, -2, -3])
@pytest.mark.parametrize('is_ascend', [True, False, 0, 1, None])
@pytest.mark.parametrize('k', [1, 4])
@pytest.mark.parametrize('dtype_i', ['float32', 'int32', 'int64'])
@pytest.mark.parametrize('ret_typ', ['value', 'indices', 'both'])
def test_onnx_export_npx_topk(tmp_path, dtype, axis, is_ascend, k, dtype_i, ret_typ):
    A = mx.np.random.uniform(0, 100, (4, 5, 6)).astype(dtype)
    kwargs = {}
    if is_ascend is not None:
        kwargs['is_ascend'] = is_ascend
    M = def_model(mx.npx, 'topk', axis=axis, k=k, dtype=dtype_i, ret_typ=ret_typ, **kwargs)
    op_export_test('topk', M, [A], tmp_path)


def test_onnx_link_op_with_multiple_outputs(tmp_path):
    A = mx.np.random.uniform(0, 100, (4, 5, 6))
    class Model1(HybridBlock):
        def __init__(self, **kwargs):
            super(Model1, self).__init__(**kwargs)

        def forward(self, x):
            out1, out2 = mx.npx.topk(x, k=3, ret_typ='both')
            out11 = out1 * 2
            out22 = out2 + 3
            return out11, out22
    op_export_test('link_op_with_multiple_outputs_case1', Model1, [A], tmp_path)

    class Model2(HybridBlock):
        def __init__(self, **kwargs):
            super(Model2, self).__init__(**kwargs)

        def forward(self, x):
            out_ = mx.npx.topk(x, k=3, ret_typ='value')
            out = out_ * 3
            return out
    op_export_test('link_op_with_multiple_outputs_case2', Model2, [A], tmp_path)

    class Model3(HybridBlock):
        def __init__(self, **kwargs):
            super(Model3, self).__init__(**kwargs)

        def forward(self, x):
            out_ = mx.npx.topk(x, k=3, ret_typ='indices')
            out = out_ * 3
            return out
    op_export_test('link_op_with_multiple_outputs_case3', Model3, [A], tmp_path)


@pytest.mark.skip(reason='maximum_scalar is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (1, 4, 1, 7)])
def test_onnx_maximum_scalar(tmp_path, dtype, shape):
    x = mx.random.uniform(0, 10, shape).astype(dtype)
    M = def_model('maximum', right=5)
    op_export_test('_maximum_scalar', M, [x], tmp_path)


@pytest.mark.skip(reason='minimum_scalar is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (1, 4, 1, 7)])
def test_onnx_minimum_scalar(tmp_path, dtype, shape):
    x = mx.random.uniform(0, 10, shape).astype(dtype)
    M = def_model('minimum', right=5)
    op_export_test('_minimum_scalar', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('fmt', ['corner', 'center'])
@pytest.mark.parametrize('clip', [-1., 0., .5, 5.])
def test_onnx_export_npx_box_decode(tmp_path, dtype, fmt, clip):
    # ensure data[0] < data[2] and data[1] < data[3] for corner format
    mul = mx.np.array([-1, -1, 1, 1], dtype=dtype)
    data = mx.np.random.uniform(0, 1, (2, 3, 4), dtype=dtype) * mul
    anchors = mx.np.random.uniform(0, 1, (1, 3, 4), dtype=dtype) * mul
    M1 = def_model(mx.npx, 'box_decode', format=fmt, clip=clip)
    op_export_test('contrib_box_decode', M1, [data, anchors], tmp_path)
    M2 = def_model(mx.npx, 'box_decode', format=fmt, clip=clip, std0=0.3, std1=1.4, std2=0.5, std3=1.6)
    op_export_test('contrib_box_decode', M1, [data, anchors], tmp_path)


@pytest.mark.skip(reason='AdaptiveAvgPooling2D is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_onnx_export_contrib_AdaptiveAvgPooling2D(tmp_path, dtype):
    x = mx.nd.random.uniform(0, 1, (1, 2, 3, 4), dtype=dtype)
    M1 = def_model('contrib.AdaptiveAvgPooling2D')
    op_export_test('contrib_AdaptiveAvgPooling2D', M1, [x], tmp_path)
    M2 = def_model('contrib.AdaptiveAvgPooling2D', output_size=1)
    op_export_test('contrib_AdaptiveAvgPooling2D', M2, [x], tmp_path)
    M3 = def_model('contrib.AdaptiveAvgPooling2D', output_size=[1])
    op_export_test('contrib_AdaptiveAvgPooling2D', M3, [x], tmp_path)
    M4 = def_model('contrib.AdaptiveAvgPooling2D', output_size=[1,1])
    op_export_test('contrib_AdaptiveAvgPooling2D', M4, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32', 'int64'])
@pytest.mark.parametrize('shapes', [((3, 3, 3), (1, 3)), ((4, 5, 6, 7), (6, 7))])
def test_onnx_export_np_mod(tmp_path, dtype, shapes):
    A = mx.np.random.uniform(-300, 300, shapes[0]).astype(dtype)
    B = mx.np.random.uniform(-30, 30, shapes[1]).astype(dtype)
    # test when dividend is zero
    B[-1] = 0
    M = def_model(mx.np, 'mod')
    op_export_test('mod', M, [A, B], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
def test_onnx_export_npx_reshape_like(tmp_path, dtype):
    if 'int' in dtype:
        x = mx.np.random.randint(0, 10, (2, 2, 3, 2), dtype=dtype)
        y = mx.np.random.randint(0, 10, (1, 4, 3, 2), dtype=dtype)
    else:
        x = mx.np.random.normal(0, 10, (2, 2, 3, 2), dtype=dtype)
        y = mx.np.random.normal(0, 10, (1, 4, 3, 2), dtype=dtype)
    M1 = def_model(mx.npx, 'reshape_like')
    op_export_test('reshape_like1', M1, [x, y], tmp_path)
    M2 = def_model(mx.npx, 'reshape_like', lhs_begin=0, lhs_end=2, rhs_begin=1, rhs_end=2)
    op_export_test('reshape_like2', M2, [x, y], tmp_path)
    M3 = def_model(mx.npx, 'reshape_like', lhs_begin=-4, lhs_end=-2, rhs_begin=-3, rhs_end=-2)
    op_export_test('reshape_like3', M3, [x, y], tmp_path)
    M4 = def_model(mx.npx, 'reshape_like', lhs_begin=0, lhs_end=None, rhs_begin=1, rhs_end=None)
    op_export_test('reshape_like4', M4, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
def test_onnx_export_npx_gather_nd(tmp_path, dtype):
    # y[0] == dim(x)
    x1 = mx.np.random.uniform(-100, 100, (4, 5, 6, 7)).astype(dtype)
    y1 = mx.np.random.randint(-4, 4, (4, 4, 4)).astype(dtype)
    M1 = def_model(mx.npx, 'gather_nd')
    op_export_test('gather_nd1', M1, [x1, y1], tmp_path)
    # y[0] < dim(x)
    x2 = mx.np.random.uniform(-100, 100, (4, 5, 6, 7)).astype(dtype)
    y2 = mx.np.random.randint(-4, 4, (2,3,4)).astype(dtype)
    M2 = def_model(mx.npx, 'gather_nd')
    op_export_test('gather_nd2', M2, [x2, y2], tmp_path)


@pytest.mark.skip(reason='UpSampling is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('shape', [(3, 4, 5, 6), (1, 1, 1, 1)])
@pytest.mark.parametrize('scale', [1, 2, 3])
def test_onnx_export_upsampling(tmp_path, dtype, shape, scale):
    A = mx.random.uniform(0, 1, shape).astype(dtype)
    M = def_model('UpSampling', scale=scale, sample_type='nearest', num_args=1)
    op_export_test('UpSampling', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('params', [((4, 5, 6), (0, 2)), ((4, 5, 6), (0, 1)),
                                    ((1, 2, 3, 4, 1), (0, 4)),
                                    ((4, 5, 1, 6), (0, 2))])
def test_onnx_export_np_swapaxes(tmp_path, dtype, params):
    shape = params[0]
    dim1, dim2 = params[1]
    x = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model(mx.np, 'swapaxes', axis1=dim1, axis2=dim2)
    op_export_test('swapaxes', M, [x], tmp_path)


@pytest.mark.skip(reason='slice_like is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('axes', [None, (0, 1, 2), (-2, -3), (-2, 0)])
def test_onnx_export_slice_like(tmp_path, dtype, axes):
    x = mx.nd.random.uniform(0, 1, (4, 5, 6, 7)).astype(dtype)
    if axes is None:
        M = def_model('slice_like')
        y = mx.nd.zeros((2, 3, 4, 5), dtype=dtype)
        op_export_test('slice_like', M, [x, y], tmp_path)
    else:
        M = def_model('slice_like', axes=axes)
        y1 = mx.nd.zeros((2, 3, 4), dtype=dtype)
        y2 = mx.nd.zeros((2, 3, 4, 5), dtype=dtype)
        y3 = mx.nd.zeros((2, 3, 4, 5, 6), dtype=dtype)
        op_export_test('slice_like_1', M, [x, y1], tmp_path)
        op_export_test('slice_like_2', M, [x, y2], tmp_path)
        op_export_test('slice_like_3', M, [x, y3], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32', 'int64'])
@pytest.mark.parametrize('axis', [None, 0, 2, -1])
@pytest.mark.parametrize('num_outputs', [2, 5])
def test_onnx_export_npx_slice_channel(tmp_path, dtype, axis, num_outputs):
    x = mx.np.zeros((10,20,30,40), dtype=dtype)
    if axis is None:
        M = def_model(mx.npx, 'slice_channel', num_outputs=num_outputs)
    else:
        M = def_model(mx.npx, 'slice_channel', axis=axis, num_outputs=num_outputs)
    op_export_test('slice_channel', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('lhs_axes', [[1, 3], [3, 1], [-2, -4], [-4, -2]])
@pytest.mark.parametrize('rhs_axes', [[1, 3], [3, 1], [-2, -4], [-4, -2]])
def test_onnx_export_npx_broadcast_like(tmp_path, dtype, lhs_axes, rhs_axes):
    x = mx.np.random.normal(0, 10, (2, 1, 1, 1, 6)).astype(dtype)
    y = mx.np.random.normal(0, 10, (2, 3, 4, 5, 6)).astype(dtype)
    M1 = def_model(mx.npx, 'broadcast_like')
    op_export_test('broadcast_like1', M1, [x, y], tmp_path)
    M2 = def_model(mx.npx, 'broadcast_like', lhs_axes=lhs_axes, rhs_axes=rhs_axes)
    op_export_test('broadcast_like2', M2, [x, y], tmp_path)


@pytest.mark.skip(reason='ROIAlign is deprecated in MXNet 2.0')
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('pooled_size', [(1, 1), (3, 3), (14, 14), (5, 7)])
@pytest.mark.parametrize('spatial_scale', [1, 0.5, 0.0625])
@pytest.mark.parametrize('spatial_ratio', [1, 2, 3, 5])
def test_onnx_export_contrib_ROIAlign(tmp_path, dtype, pooled_size, spatial_scale, spatial_ratio):
    data = mx.random.uniform(0, 1, (5, 3, 512, 512)).astype(dtype)
    rois = mx.nd.array([[-1, 0, 0, 0, 0],
                        [0, 0, 0, 63, 63],
                        [1, 34, 52, 25, 85],
                        [2, 50, 50, 100, 100],
                        [3, 0, 0, 127, 127],
                        [4, 12, 84, 22, 94],
                        [0, 0, 0, 1, 1]]).astype(dtype)
    M = def_model('contrib.ROIAlign', pooled_size=pooled_size, spatial_scale=spatial_scale,
                  sample_ratio=spatial_ratio)
    # according to https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/contrib/symbol/index.html#mxnet.contrib.symbol.ROIAlign
    # the returned value for when batch_id < 0 should be all 0's
    # however mxnet 1.8 does always behave this way so we set the first roi to 0's manually
    def mx_map(x):
        x[0] = 0
        return x
    op_export_test('_contrib_ROIAlign', M, [data, rois], tmp_path, mx_map=mx_map)


@pytest.mark.skip(reason='TODO: the behavior of this op has changed in MXNet 2.0. New conversion function needed')
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('transpose_a', [True, False])
@pytest.mark.parametrize('transpose_b', [True, False])
def test_onnx_export_batch_dot(tmp_path, dtype, transpose_a, transpose_b):
    x1 = mx.np.random.normal(0, 10, (2, 3, 4, 5, 6), dtype=dtype)
    y1 = mx.np.random.normal(0, 10, (2, 3, 4, 6, 5), dtype=dtype)
    M1 = def_model(mx.npx, 'batch_dot')
    op_export_test('batch_dot1', M1, [x1, y1], tmp_path)
    x2 = mx.np.random.normal(0, 10, (2, 3, 4, 5, 5), dtype=dtype)
    y2 = mx.np.random.normal(0, 10, (2, 3, 4, 5, 5), dtype=dtype)
    M2 = def_model(mx.npx, 'batch_dot', transpose_a=transpose_a, transpose_b=transpose_b)
    op_export_test('batch_dot2', M2, [x2, y2], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 1, 60, 60)])
@pytest.mark.parametrize('count_include_pad', [True, False])
@pytest.mark.parametrize('pooling_convention', ['full', 'valid'])
@pytest.mark.parametrize('kernel', [(3, 3), (4, 5), (14, 14)])
@pytest.mark.parametrize('stride', [None, (1, 1), (2, 2), (3, 4), (4, 5)])
@pytest.mark.parametrize('pad', [None, (1, 1), (3, 4), (4, 5)])
def test_onnx_export_npx_pooling_avg(tmp_path, dtype, shape, count_include_pad, pooling_convention,
                                 kernel, stride, pad):
    # mxnet and onnxruntime has different implementation of count_include_pad on the left column
    # and bottom row
    if count_include_pad == True:
        return
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1]):
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model(mx.npx, 'pooling', count_include_pad=count_include_pad, pool_type='avg',
                  pooling_convention=pooling_convention, layout='NCHW', **kwargs)
    # Note here we use np.nan_to_num to map the onnx output because onnxruntime AveragePool will
    # output NaN in some edge cases where mxnet outputs 0
    op_export_test('pooling_avg', M, [x], tmp_path, onnx_map=np.nan_to_num)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 16, 16, 16), (1, 1, 10, 18, 18)])
@pytest.mark.parametrize('count_include_pad', [True, False])
@pytest.mark.parametrize('pooling_convention', ['full', 'valid'])
@pytest.mark.parametrize('kernel', [(1, 1, 1), (3, 3, 3), (1, 7, 7)])
@pytest.mark.parametrize('stride', [None, (1, 1, 1), (1, 2, 3)])
@pytest.mark.parametrize('pad', [None, (0, 1, 1), (1, 2, 3)])
def test_onnx_export_npx_pooling_avg_3d(tmp_path, dtype, shape, count_include_pad, pooling_convention,
                                    kernel, stride, pad):
    # mxnet and onnxruntime has different implementation of count_include_pad on the left column
    # and bottom row
    if count_include_pad == True:
        return
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1] or pad[2] >= kernel[2]):
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model(mx.npx, 'pooling', count_include_pad=count_include_pad, pool_type='avg',
                  pooling_convention=pooling_convention, layout='NCDHW', **kwargs)
    # Note here we use np.nan_to_num to map the onnx output because onnxruntime AveragePool will
    # output NaN in some edge cases where mxnet outputs 0
    def mx_nan_to_num(a):
        return np.nan_to_num(a.asnumpy())
    op_export_test('pooling_avg_3d', M, [x], tmp_path, onnx_map=np.nan_to_num, mx_map=mx_nan_to_num)



@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 1, 60, 60)])
@pytest.mark.parametrize('pooling_convention', ['full', 'valid'])
@pytest.mark.parametrize('kernel', [(3, 3), (4, 5), (14, 14)])
@pytest.mark.parametrize('stride', [None, (1, 1), (2, 2), (3, 4), (4, 5)])
@pytest.mark.parametrize('pad', [None, (1, 1), (3, 4), (4, 5)])
def test_onnx_export_npx_pooling_max(tmp_path, dtype, shape, pooling_convention, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1]):
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model(mx.npx, 'pooling', pool_type='max', pooling_convention=pooling_convention,
                  layout='NCHW', **kwargs)
    op_export_test('pooling_max', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 16, 16, 16), (1, 1, 10, 18, 18)])
@pytest.mark.parametrize('pooling_convention', ['full', 'valid'])
@pytest.mark.parametrize('kernel', [(1, 1, 1), (3, 3, 3), (1, 7, 7)])
@pytest.mark.parametrize('stride', [None, (1, 1, 1), (1, 2, 3)])
@pytest.mark.parametrize('pad', [None, (0, 1, 1), (1, 2, 3)])
def test_onnx_export_npx_pooling_max_3d(tmp_path, dtype, shape, pooling_convention, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1] or pad[2] >= kernel[2]):
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model(mx.npx, 'pooling', pool_type='max', pooling_convention=pooling_convention,
                  layout='NCDHW', **kwargs)
    op_export_test('pooling_max_3d', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 1, 60, 60)])
@pytest.mark.parametrize('p_value', [1, 2])
@pytest.mark.parametrize('kernel', [(3, 3), (4, 5), (14, 14)])
@pytest.mark.parametrize('stride', [None, (1, 1), (2, 2), (3, 4), (4, 5)])
@pytest.mark.parametrize('pad', [None, (1, 1), (3, 4), (4, 5)])
def test_onnx_export_npx_pooling_lp(tmp_path, dtype, shape, p_value, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1]):
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model(mx.npx, 'pooling', pool_type='lp', pooling_convention='valid',
                  p_value=p_value, layout='NCHW', **kwargs)
    op_export_test('pooling_lp', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 16, 16, 16), (1, 1, 10, 18, 18)])
@pytest.mark.parametrize('p_value', [1, 2])
@pytest.mark.parametrize('kernel', [(1, 1, 1), (3, 3, 3), (1, 7, 7)])
@pytest.mark.parametrize('stride', [None, (1, 1, 1), (1, 2, 3)])
@pytest.mark.parametrize('pad', [None, (0, 1, 1), (1, 2, 3)])
def test_onnx_export_npx_pooling_lp_3d(tmp_path, dtype, shape, p_value, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1] or pad[2] >= kernel[2]):
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model(mx.npx, 'pooling', pool_type='lp', pooling_convention='valid',
                  p_value=p_value, layout='NCDHW', **kwargs)
    op_export_test('pooling_lp_3d', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 1, 60, 60)])
@pytest.mark.parametrize('pool_type', ['avg', 'max', 'lp'])
@pytest.mark.parametrize('p_value', [1, 2])
@pytest.mark.parametrize('kernel', [(3, 3), (14, 14)])
@pytest.mark.parametrize('stride', [None, (3, 4)])
@pytest.mark.parametrize('pad', [None, (3, 4)])
def test_onnx_export_npx_pooling_global(tmp_path, dtype, shape, pool_type, p_value, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1]):
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    # kernel, stride, and pad should have no effect on the results
    M = def_model(mx.npx, 'pooling', global_pool=True, pool_type=pool_type, pooling_convention='valid',
                  p_value=p_value, layout='NCHW', **kwargs)
    op_export_test('pooling_global', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 16, 16, 16), (1, 1, 10, 18, 18)])
@pytest.mark.parametrize('pool_type', ['avg', 'max', 'lp'])
@pytest.mark.parametrize('p_value', [1, 2])
@pytest.mark.parametrize('kernel', [(1, 1, 1), (3, 3, 3)])
@pytest.mark.parametrize('stride', [None, (1, 1, 1)])
@pytest.mark.parametrize('pad', [None, (0, 1, 1)])
def test_onnx_export_npx_pooling_global_3d(tmp_path, dtype, shape, pool_type, p_value, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1] or pad[2] >= kernel[2]):
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    # kernel, stride, and pad should have no effect on the results
    M = def_model(mx.npx, 'pooling', global_pool=True, pool_type=pool_type, pooling_convention='valid',
                  p_value=p_value, layout='NCDHW', **kwargs)
    op_export_test('pooling_global_3d', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_onnx_export_np_log2(tmp_path, dtype):
    x = mx.np.random.normal(0, 10, (2, 3, 4, 5)).astype(dtype)
    M = def_model(mx.np, 'log2')
    op_export_test('log2', M, [x], tmp_path)


@pytest.mark.skip(reason='broadcast_mul is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
def test_onnx_export_broadcast_mul(tmp_path, dtype):
    M = def_model('broadcast_mul')
    x = mx.nd.array([[1,2,3],[4,5,6]], dtype=dtype)
    y = mx.nd.array([[0],[3]], dtype=dtype)
    op_export_test('broadcast_mul', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 6, 60, 60)])
@pytest.mark.parametrize('num_filter', [2, 4, 32])
@pytest.mark.parametrize('num_group', [1, 2])
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('kernel', [(3, 3), (4, 5), (14, 14)])
@pytest.mark.parametrize('stride', [None, (1, 1), (2, 2), (3, 4), (4, 5)])
@pytest.mark.parametrize('pad', [None, (1, 1), (3, 4), (4, 5)])
@pytest.mark.parametrize('dilate', [None, (1, 1)])
def test_onnx_export_npx_convolution(tmp_path, dtype, shape, num_filter, num_group, no_bias,
                                 kernel, stride, pad, dilate):
    if shape[1] % num_group:
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    w_shape = (num_filter,) + (shape[1] // num_group,) + kernel
    w = mx.np.random.uniform(0, 1, w_shape, dtype=dtype)
    b_shape = (num_filter)
    b = mx.np.random.uniform(0, 1, b_shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    if dilate:
        kwargs['dilate'] = dilate
    M = def_model(mx.npx, 'convolution', num_filter=num_filter, num_group=num_group,  no_bias=no_bias,
                  layout='NCHW', **kwargs)
    inputs = [x, w] if no_bias else [x, w, b]
    op_export_test('convolution', M, inputs, tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 4, 16, 16, 16), (1, 3, 10, 18, 18)])
@pytest.mark.parametrize('num_filter', [2, 4, 32])
@pytest.mark.parametrize('num_group', [1, 2])
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('kernel', [(3, 3, 3), (1, 1, 1), (1, 7, 7)])
@pytest.mark.parametrize('stride', [None, (1, 1, 1), (1, 2, 3)])
@pytest.mark.parametrize('pad', [None, (0, 1, 1), (1, 2, 3)])
@pytest.mark.parametrize('dilate', [None, [2, 2, 2]])
def test_onnx_export_npx_convolution_3D(tmp_path, dtype, shape, num_filter, num_group, no_bias,
                                 kernel, stride, pad, dilate):
    if shape[1] % num_group:
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    w_shape = (num_filter,) + (shape[1] // num_group,) + kernel
    w = mx.np.random.uniform(0, 1, w_shape, dtype=dtype)
    b_shape = (num_filter)
    b = mx.np.random.uniform(0, 1, b_shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    if dilate:
        kwargs['dilate'] = dilate
    M = def_model(mx.npx, 'convolution', num_filter=num_filter, num_group=num_group,  no_bias=no_bias,
                  layout='NCDHW', **kwargs)
    inputs = [x, w] if no_bias else [x, w, b]
    op_export_test('convolution_3d', M, inputs, tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('num_outputs', [1, 3, 9])
@pytest.mark.parametrize('axis', [1, 2, -1, -2])
@pytest.mark.parametrize('squeeze_axis', [True, False, 0, 1])
def test_onnx_export_npx_slice_channel(tmp_path, dtype, num_outputs, axis, squeeze_axis):
    shape = (3, 9, 18)
    if squeeze_axis and shape[axis] != num_outputs:
        return
    M = def_model(mx.npx, 'slice_channel', num_outputs=num_outputs, axis=axis, squeeze_axis=squeeze_axis)
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    op_export_test('slice_channel', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('momentum', [0.9, 0.5, 0.1])
def test_onnx_export_npx_batchnorm(tmp_path, dtype, momentum):
    x = mx.np.random.normal(0, 10, (2, 3, 4, 5)).astype(dtype)
    gamma = mx.np.random.normal(0, 10, (3)).astype(dtype)
    beta = mx.np.random.normal(0, 10, (3)).astype(dtype)
    moving_mean = mx.np.random.normal(0, 10, (3)).astype(dtype)
    moving_var = mx.np.abs(mx.np.random.normal(0, 10, (3))).astype(dtype)
    M = def_model(mx.npx, 'batch_norm', eps=1e-5, momentum=momentum, fix_gamma=False, use_global_stats=False)
    op_export_test('batch_norm', M, [x, gamma, beta, moving_mean, moving_var], tmp_path)


@pytest.mark.skip(reason='TODO argsort changed spec in 2.0')
# onnxruntime does not seem to support float64 and int32
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
@pytest.mark.parametrize('axis', [0, 2, -1, -2, -3])
@pytest.mark.parametrize('is_ascend', [True, False, 0, 1, None])
@pytest.mark.parametrize('dtype_i', ['float32', 'int32', 'int64'])
def test_onnx_export_argsort(tmp_path, dtype, axis, is_ascend, dtype_i):
    A = mx.np.random.uniform(0, 100, (4, 5, 6)).astype(dtype)
    kwargs = {}
    if is_ascend is not None:
        kwargs['is_ascend'] = is_ascend
    M = def_model(mx.np, 'argsort', axis=axis, dtype=dtype_i, **kwargs)
    op_export_test('argsort', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('reps', [(2, 3), (2, ), (2, 3, 4)])
def test_onnx_export_np_tile(tmp_path, dtype, reps):
    x = mx.np.random.normal(0, 100, (5, 6)).astype(dtype)
    M = def_model(mx.np, 'tile', reps=reps)
    op_export_test('tile', M, [x], tmp_path)


@pytest.mark.skip(reason='TODO take changed spec in 2.0')
@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('axis', [-3, -2, -1, 0, 1, 2])
@pytest.mark.parametrize('mode', ['clip', 'wrap'])
def test_onnx_export_take(tmp_path, dtype, axis, mode):
    x = mx.np.random.normal(0, 10, (3, 4, 5)).astype(dtype)
    y = mx.np.random.randint(-100, 100, (6, 7)).astype(dtype)
    M1 = def_model(mx.np, 'take')
    op_export_test('take1', M1, [x, y], tmp_path)
    M2 = def_model(mx.np, 'take', axis=axis, mode=mode)
    op_export_test('take2', M2, [x, y], tmp_path)


@pytest.mark.skip(reason='TODO take changed spec in 2.0')
@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('axis', [-3, -2, -1, 0, 1, 2])
def test_onnx_export_take_raise(tmp_path, dtype, axis):
    x = mx.nd.random.normal(0, 10, (3, 4, 5)).astype(dtype)
    y = mx.random.randint(0, 3, (6, 7)).astype(dtype)
    M = def_model('take', axis=axis, mode='raise')
    op_export_test('take', M, [x, y], tmp_path)


# onnxruntime currently does not support int32
@pytest.mark.parametrize("dtype", ["float16", "float32", "int64"])
@pytest.mark.parametrize("depth", [1, 3, 5, 10])
@pytest.mark.parametrize("shape", [(1,1), (1,5), (5,5), (3,4,5)])
def test_onnx_export_npx_one_hot(tmp_path, dtype, depth, shape):
    M = def_model(mx.npx, 'one_hot', depth=depth, dtype=dtype)
    x = mx.np.random.randint(0, 10, shape).astype('int64')
    op_export_test('one_hot', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('params', [((6, 5, 4), [1, 2, 4, 5, 6]),
                                     ((7, 3, 5), [1, 7, 4]),
                                     ((3, 2, 1), [1, 2])])
def test_onnx_export_npx_sequence_reverse(tmp_path, dtype, params):
    x = mx.np.random.uniform(0, 10, params[0]).astype(dtype)
    M1 = def_model(mx.npx, 'sequence_reverse')
    op_export_test('SequenceReverse1', M1, [x], tmp_path)
    seq_len = mx.np.array(params[1])
    M1 = def_model(mx.npx, 'sequence_reverse', use_sequence_length=True)
    op_export_test('SequenceReverse1', M1, [x, seq_len], tmp_path)


@pytest.mark.skip(reason='TODO rnn changeed spec in 2.0')
@pytest.mark.parametrize('mode', ['lstm', 'gru', 'rnn_tanh', 'rnn_relu'])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('state_size', [16, 32])
@pytest.mark.parametrize('input_size', [16, 32, 64])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('seq_length', [16])
@pytest.mark.parametrize('bidirectional', [True, False])
def test_onnx_export_RNN(tmp_path, mode, dtype, state_size, input_size, num_layers, batch_size, seq_length, bidirectional):
    # TODO: The current implementation fails assertion checks for large parm/state_size. 
    # for num_layers >= 2, input_size must equal to state_size
    if num_layers >= 2 and input_size != state_size:
        return
    # Currently only bidirectional supports lstm with num_layers = 1
    if bidirectional and (mode != 'lstm' or num_layers != 1):
        return

    b = 1
    if bidirectional:
        b = 2

    factor = 1
    if mode == 'gru':
        factor = 3
    elif mode == 'lstm':
        factor = 4

    M = def_model(mx.npx, 'rnn', mode=mode, state_size=state_size, state_outputs=True,  num_layers=num_layers, p=0, bidirectional=bidirectional)
    x = mx.np.random.normal(0, 10, (seq_length, batch_size, input_size)).astype(dtype)
    param = mx.np.random.normal(0, 1, [b*num_layers*factor*state_size*input_size +
                                       b*num_layers*factor*state_size*state_size +
                                       b*num_layers*2*factor*state_size]).astype(dtype)
    state = mx.np.random.uniform(-1, 1, [b*num_layers, batch_size, state_size]).astype(dtype)
    if mode == 'lstm':
        cell = mx.np.random.uniform(-1, 1, [b*num_layers, batch_size, state_size]).astype(dtype)
        op_export_test('rnn', M, [x, param, state, cell], tmp_path)
    elif mode == 'rnn_relu':
        # set large atol as relu can outputs big numbers
        op_export_test('rnn', M, [x, param, state], tmp_path, atol=1e20)
    else:
        op_export_test('rnn', M, [x, param, state], tmp_path, atol=1e-2)


@pytest.mark.skip(reason='contrib_div_sqrt_dim is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (6, 7), (8,)])
def test_onnx_export_contrib_div_sqrt_dim(tmp_path, dtype, shape):
    A = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model('contrib.div_sqrt_dim')
    op_export_test('contrib_div_sqrt_dim', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (6, 7), (8,)])
@pytest.mark.parametrize('operator', ['sin', 'cos', 'tan', 'tanh', 'arcsin', 'arccos', 'arctan',
                                      'exp', 'log', 'ceil', 'floor'])#'sigmoid', 'relu', 'exp', 'identity'])
def test_onnx_export_np_ufunc(tmp_path, dtype, shape, operator):
    A = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model(mx.np, operator)
    op_export_test('np_ufunc', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (6, 7), (8,)])
@pytest.mark.parametrize('operator', ['sigmoid', 'relu'])
def test_onnx_export_npx_ufunc(tmp_path, dtype, shape, operator):
    A = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model(mx.npx, operator)
    op_export_test('npx_ufunc', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 6, 60, 60)])
@pytest.mark.parametrize('num_filter', [4, 16, 256])
@pytest.mark.parametrize('num_group', [1, 2])
@pytest.mark.parametrize('no_bias', [False, True])
@pytest.mark.parametrize('kernel', [(2, 2), (3, 4)])
@pytest.mark.parametrize('stride', [(1, 1), (2, 2)])
@pytest.mark.parametrize('pad', [None, (0, 0), (1, 1)])
@pytest.mark.parametrize('dilate', [None, (1, 1)])
@pytest.mark.parametrize('adj', [(0, 0), (1, 1)])
def test_onnx_export_npx_deconvolution(tmp_path, dtype, shape, num_filter, num_group, no_bias,
                                 kernel, stride, pad, dilate, adj):
    for i in range(len(stride)):
        if stride[i] <= adj[i]:
            return
    if shape[1] % num_group:
        return
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    w_shape = (shape[1],) + (num_filter // num_group,) + kernel
    w = mx.np.random.uniform(0, 1, w_shape, dtype=dtype)
    b_shape = (num_filter)
    b = mx.np.random.uniform(0, 1, b_shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    if dilate:
        kwargs['dilate'] = dilate
    if adj:
        kwargs['adj'] = adj
    M = def_model(mx.npx, 'deconvolution', num_filter=num_filter, num_group=num_group,  no_bias=no_bias,
                  layout='NCHW', **kwargs)
    inputs = [x, w] if no_bias else [x, w, b]
    op_export_test('deconvolution', M, inputs, tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'float64'])
@pytest.mark.parametrize('mode', ['edge', 'constant', 'reflect'])
@pytest.mark.parametrize('params', [((3, 4, 5, 6), (0, 0, 0, 0, 2, 3, 4, 5)),
                                    ((7, 6, 5, 4, 3), (0, 0, 0, 0, 4, 4, 3, 3, 2, 1))])
def test_onnx_export_npx_pad(tmp_path, dtype, mode, params):
     kwargs = {}
     kwargs['constant_value'] = 9999.55
     kwargs['pad_width'] = params[1]
     x = mx.np.random.uniform(0, 1, size=params[0], dtype=dtype)
     M = def_model(mx.npx, 'pad', mode=mode, **kwargs)
     op_export_test('pad', M, [x], tmp_path)


# Note that due to ONNX limitation, the behavior for when inputs > 2-D is different from that of
# MXNet
@pytest.mark.skip(reason='TODO MXNet 2.0 dot changed behavior')
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('params', [((4, 5), (5, 6), False, False),
                                    ((5, 4), (5, 6), True, False),
                                    ((5, 4), (6, 5), True, True),
                                    ((4, 5), (6, 5), False, True),
                                    ((4, 5), (5), False, False),
                                    ((4,), (4, 5), False, False),
                                    ((4, 5), (5,), False, False)])
def test_onnx_export_dot(tmp_path, dtype, params):
    A = mx.np.random.uniform(0, 1, params[0], dtype=dtype)
    B = mx.np.random.uniform(0, 1, params[1], dtype=dtype)
    M = def_model(mx.np, 'dot', transpose_a=params[2], transpose_b=params[3])
    op_export_test('dot', M, [A, B], tmp_path)


@pytest.mark.skip(reason='flatten is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(3, 4, 5, 6), (7, 8)])
def test_onnx_export_flatten(tmp_path, dtype, shape):
    x = mx.random.uniform(0, 1, shape, dtype='float32').astype(dtype)
    M = def_model('flatten')
    op_export_test('flatten', M, [x], tmp_path)


# Note that due to ONNX limitation, the behavior for when inputs > 2-D is different from that of
# MXNet
@pytest.mark.skip(reason='linalg.gemm2 is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('alpha', [1, 1.5])
@pytest.mark.parametrize('params', [((4, 5), (5, 4), False, False),
                                    ((4, 5, 6), (4, 6, 5), False, False),
                                    ((4, 5, 6, 7), (4, 5, 6, 7), True, False),
                                    ((4, 5, 6, 7), (4, 5, 6, 7), False, True),
                                    ((4, 5, 9, 7), (4, 5, 6, 9), True, True)])
def test_onnx_export_linalg_gemm2(tmp_path, dtype, alpha, params):
    A = mx.random.uniform(0, 1, params[0], dtype=dtype)
    B = mx.random.uniform(0, 1, params[1], dtype=dtype)
    M = def_model('linalg.gemm2', alpha=alpha, transpose_a=params[2], transpose_b=params[3])
    op_export_test('_linalg_gemm2', M, [A, B], tmp_path)


@pytest.mark.skip(reason='LogisticRegressionOutput is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (6, 7), (8,)])
def test_onnx_export_LogisticRegressionOutput(tmp_path, dtype, shape):
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    y = mx.nd.zeros(shape, dtype=dtype)
    M = def_model('LogisticRegressionOutput')
    op_export_test('LogisticRegressionOutput', M, [x, y], tmp_path)


@pytest.mark.skip(reason='SoftmaxOutput is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('shape', [(4, 5, 6), (6, 7), (3, 4, 5, 6, 7)])
def test_onnx_export_SoftmaxOutput(tmp_path, dtype, shape):
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    y = mx.nd.zeros(shape[:-1], dtype=dtype)
    M = def_model('SoftmaxOutput')
    op_export_test('SoftmaxOutput', M, [x, y], tmp_path)


# Due to ONNX limitation, L2Normalization only supports channel mode for now
@pytest.mark.skip(reason='L2Normalization is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (3, 4, 5, 6, 7)])
def test_onnx_export_L2Normalization(tmp_path, dtype, shape):
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    M = def_model('L2Normalization', mode='channel')
    op_export_test('L2Normalization', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (3, 4, 5, 6, 7)])
@pytest.mark.parametrize('eps', [0.001, 0.00001])
def test_onnx_export_npx_instance_norm(tmp_path, dtype, shape, eps):
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    gamma = mx.np.random.uniform(0, 1, shape[1:2], dtype=dtype)
    beta = mx.np.random.uniform(0, 1, shape[1:2], dtype=dtype)
    M = def_model(mx.npx, 'instance_norm', eps=eps)
    op_export_test('instance_norm', M, [x, gamma, beta], tmp_path)


# ONNXRuntime only supports 4-D inputs
@pytest.mark.skip(reason='LRN is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(4, 5, 6, 7)])
@pytest.mark.parametrize('alpha', [0.001, 0.00001])
@pytest.mark.parametrize('beta', [0.75, 0.8])
@pytest.mark.parametrize('knorm', [1, 2])
@pytest.mark.parametrize('nsize', [3, 5])
def test_onnx_export_LRN(tmp_path, dtype, shape, alpha, beta, knorm, nsize):
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    M = def_model('LRN', alpha=alpha, beta=beta, knorm=knorm, nsize=nsize)
    op_export_test('LRN', M, [x], tmp_path)


@pytest.mark.skip(reason='Crop is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 224, 224), (5, 6, 64, 64)])
@pytest.mark.parametrize('h_w', [(10, 10), (7, 11)])
@pytest.mark.parametrize('offset', [(7, 13), (10, 10)])
@pytest.mark.parametrize('shape2', [None, (10, 10, 16, 16)])
def test_onnx_export_Crop(tmp_path, dtype, shape, h_w, offset, shape2):
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    M = def_model('Crop', h_w=h_w, offset=offset, center_crop=False)
    if shape2 is not None:
        y = mx.random.uniform(0, 1, shape2, dtype=dtype)
        op_export_test('Crop', M, [x, y], tmp_path)
    else:
        op_export_test('Crop', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('shape', [(100,), (3, 4, 5), (6, 7)])
def test_onnx_export_np_reciprocal(tmp_path, dtype, shape):
    A = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model(mx.np, 'reciprocal')
    op_export_test('reciprocal', M, [A], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize('shape', [(1, 3), (3, 4, 5)])
def test_onnx_export_np_power(tmp_path, shape, dtype):
    x = mx.np.random.uniform(-5, 5, shape).astype(dtype)
    y = mx.np.random.uniform(-10, 10, shape).astype(dtype)
    M = def_model(mx.np, 'power')
    op_export_test('power', M, [x, y], tmp_path)


@pytest.mark.skip(reason='broadcast_power is deprecated in 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize('shape', [(1, 3), (3, 4, 5)])
def test_onnx_export_broadcast_power(tmp_path, shape, dtype):
    x = mx.nd.random.uniform(-5, 5, shape).astype(dtype)
    y = mx.nd.random.uniform(-10, 10, shape).astype(dtype)
    M = def_model('broadcast_power')
    op_export_test('broadcast_power', M, [x, y], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize('shape', [(3, 4, 5), (6, 7), (8,)])
def test_onnx_export_np_sqrt(tmp_path, dtype, shape):
    A = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model(mx.np, 'sqrt')
    op_export_test('sqrt', M, [A], tmp_path)


@pytest.mark.skip(reason='depth_to_space is deprecated in 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("params", [[(1,4,2,3), 1], [(1,4,2,3), 2]])
def test_onnx_export_depth_to_space(tmp_path, dtype, params):
    shape, block_size = params
    M = def_model('depth_to_space', block_size=block_size)
    x = mx.nd.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    op_export_test('depth_to_space', M, [x], tmp_path)


@pytest.mark.skip(reason='space to depth is deprecated in 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("params", [[(1,4,2,3), 1], [(1,1,4,6),2]])
def test_onnx_export_space_to_depth(tmp_path, dtype, params):
    shape, block_size = params
    M = def_model('space_to_depth', block_size=block_size)
    x = mx.nd.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    op_export_test('space_to_depth', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_np_square(tmp_path, dtype, shape):
    M = def_model(mx.np, 'square')
    x = mx.np.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    op_export_test('square', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_npx_shape_array(tmp_path, dtype, shape):
    M = def_model(mx.npx, 'shape_array')
    x = mx.np.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    op_export_test('shape_array', M, [x], tmp_path)


@pytest.mark.skip(reason='hard_sigmoid is deprecaed in 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
@pytest.mark.parametrize("alpha", [None, 0.1, 0.4567, 0.9])
@pytest.mark.parametrize("beta", [None, 0.1, 0.4567, 0.5, 0.9])
def test_onnx_export_hard_sigmoid(tmp_path, dtype, shape, alpha, beta):
    kwargs = { }
    if alpha is not None:
        kwargs['alpha'] = alpha
    if beta is not None:
        kwargs['beta'] = beta
    M = def_model('hard_sigmoid', **kwargs)
    x = mx.nd.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    op_export_test('hard_sigmoid', M, [x], tmp_path)


@pytest.mark.skip(reason='broadcast_lesser is deprecaed in 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_broadcast_lesser(tmp_path, dtype, shape):
    M = def_model('broadcast_lesser')
    x = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    y = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    op_export_test('broadcast_lesser', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_npx_broadcast_greater(tmp_path, dtype, shape):
    M = def_model(mx.npx, 'broadcast_greater')
    x = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    y = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    op_export_test('broadcast_greater', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize("shape", [(10,5), (1,2,3), (4,5,6)])
@pytest.mark.parametrize('axis', [None, 1])
def test_onnx_export_npx_log_softmax(tmp_path, dtype, shape, axis):
    x = mx.np.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    M = def_model(mx.npx, 'log_softmax', **kwargs)
    op_export_test('log_softmax', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (2,3), (4,5,6)])
def test_onnx_export_np_logical_and(tmp_path, dtype, shape):
    M = def_model(mx.np, 'logical_and')
    x = mx.np.random.uniform(-1, 1, shape).astype(dtype)
    y = mx.np.random.uniform(-1, 1, shape).astype(dtype)
    op_export_test('logical_and', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (2,3), (4,5,6)])
def test_onnx_export_np_logical_or(tmp_path, dtype, shape):
    M = def_model(mx.np, 'logical_or')
    x = mx.np.random.uniform(-1, 1, shape).astype(dtype)
    y = mx.np.random.uniform(-1, 1, shape).astype(dtype)
    op_export_test('logical_or', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (2,3), (4,5,6)])
def test_onnx_export_np_logical_xor(tmp_path, dtype, shape):
    M = def_model(mx.np, 'logical_xor')
    x = mx.np.random.uniform(-1, 1, shape).astype(dtype)
    y = mx.np.random.uniform(-1, 1, shape).astype(dtype)
    op_export_test('logical_xor', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shapes", [[(1,3),(2,3)], [(2,1,3,1),(2,8,3,9)], [(1,3,6),(5,3,6)]])
def test_onnx_export_np_broadcast_to(tmp_path, dtype, shapes):
    in_shape, to_shape = shapes
    M = def_model(np, 'broadcast_to', shape=to_shape)
    x = mx.np.random.uniform(-100, 100, in_shape).astype(dtype)
    op_export_test('broadcast_to', M, [x], tmp_path)


# onnxruntime currently does not support int32
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int64'])
@pytest.mark.parametrize('shape', [(1,), (2, 3), (4, 5, 6)])
def test_onnx_export_np_clip(tmp_path, dtype, shape):
    A = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    a_min = mx.np.min(A).astype('float32').asnumpy() + 5
    a_max = mx.np.max(A).astype('float32').asnumpy() - 5
    M = def_model(mx.np, 'clip', a_min=a_min, a_max=a_max)
    op_export_test('clip', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (6, 7), (8,)])
@pytest.mark.parametrize('func', [lambda x : x + np.random.rand(1)[0] * 100,
                                  lambda x : x * np.random.rand(1)[0] * 100,
                                  lambda x : x - np.random.rand(1)[0] * 100,
                                  lambda x : np.random.rand(1)[0] * 100 - x,
                                  lambda x : x / (np.random.rand(1)[0] * 100 + 1),
                                  lambda x : np.random.rand(1)[0] * 100 / x,
                                  lambda x : x ** np.random.rand(1)[0] * 100,
                                 ])
def test_onnx_export_np_scalar_op(tmp_path, dtype, shape, func):
    A = mx.np.random.uniform(1, 100, shape).astype(dtype)
    M = def_model_from_func(func)
    op_export_test('_scalar', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32'])
@pytest.mark.parametrize('shape', [(1, 1, 1), (2, 3, 4), (5, 6, 7, 8)])
@pytest.mark.parametrize('axis', [None, 0, 1, 2, -1, -2])
@pytest.mark.parametrize('op_name', ['argmax', 'argmin'])
def test_onnx_export_np_arg_max_min(tmp_path, dtype, shape, axis, op_name):
    A = mx.np.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model(mx.np, op_name, axis=axis)
    op_export_test(op_name, M, [A], tmp_path)


# onnx max and min have issue comparing negative float16 values
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [[(2, 3), (2, 3)], [(5, 4), (5, 4)]])
@pytest.mark.parametrize('op_name', ['maximum', 'minimum'])
def test_onnx_export_np_maximum_minimum(tmp_path, dtype, shape, op_name):
    lhs = mx.np.random.uniform(1, 100, shape[0]).astype(dtype)
    rhs = mx.np.random.uniform(1, 100, shape[1]).astype(dtype)
    M = def_model(mx.np, op_name)
    op_export_test(op_name, M, [lhs, rhs], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32','int32', 'int64'])
@pytest.mark.parametrize('shape', [(2, 3), (4, 5, 6)])
@pytest.mark.parametrize('axis', [None, 0, 1, -1, (0, 1)])
@pytest.mark.parametrize('keepdims', [True, False])
@pytest.mark.parametrize('op_name', ['max', 'min', 'mean', 'prod', 'sum'])
def test_onnx_export_np_reduce_op(tmp_path, dtype, shape, axis, keepdims, op_name):
    x = mx.np.random.uniform(1, 100, size=shape).astype(dtype)
    M = def_model(mx.np, op_name, axis=axis, keepdims=keepdims)
    op_export_test(op_name, M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1,), (3, ), (4, 5), (3, 4, 5)])
@pytest.mark.parametrize('op_name', ['power', 'add', 'subtract', 'multiply', 'divide'])
def test_onnx_export_np_arithmetic(tmp_path, dtype, shape, op_name):
    x = mx.np.random.uniform(1, 100, size=shape).astype(dtype)
    y = mx.np.random.uniform(1, 2, size=shape).astype(dtype)
    M = def_model(mx.np, op_name)
    op_export_test(op_name, M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [[(3, 4), (3, 4)], [(3, 4), (3, 1)], [(3, 4), (4)]])
@pytest.mark.parametrize('op_name', ['add', 'subtract', 'multiply', 'divide'])
def test_onnx_export_np_arithmetic_broadcast_case(tmp_path, dtype, shape, op_name):
    x = mx.np.random.uniform(1, 100, size=shape[0]).astype(dtype)
    y = mx.np.random.uniform(1, 2, size=shape[1]).astype(dtype)
    M = def_model(mx.np, op_name)
    op_export_test(op_name, M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1,), (3, ), (4, 5), (3, 4, 5)])
def test_onnx_export_np_negative(tmp_path, dtype, shape):
    x = mx.np.random.uniform(-100, 100, size=shape).astype(dtype)
    M = def_model(mx.np, 'negative')
    op_export_test('negative', M, [x], tmp_path)


@pytest.mark.skip(reason='addn is deprecated in 2.0')
@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('shape', [(1,), (3, ), (4, 5), (3, 4, 5)])
def test_onnx_export_addn(tmp_path, dtype, shape):
    x = mx.nd.random.uniform(-100, 100, shape=shape).astype(dtype)
    M = def_model('add_n')
    op_export_test('add_n', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape_axis', [[(1, 1), None], [(3, 1, 2, 1), (None)], [(3, 1, 2, 1), (1)], 
                            [(3, 1, 2, 1), (1, 3)]])
def test_onnx_export_np_squeeze(tmp_path, dtype, shape_axis):
    x = mx.np.random.uniform(1, 100, size=shape_axis[0]).astype(dtype)
    M = def_model(mx.np, 'squeeze', axis=shape_axis[1])
    op_export_test('squeeze', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, -1, (0, 2), (0, 1, 2)])
@pytest.mark.parametrize("shape", [(4, 5, 6), (3, 4, 5, 6)])
def test_onnx_export_npx_norm(tmp_path, dtype, order, axis, shape, keepdims):
    kwargs = {}
    if order is not None:
        kwargs['ord'] = order
    if axis is not None:
        kwargs['axis'] = axis
    if keepdims is not None:
        kwargs['keepdims'] = keepdims
    M = def_model(mx.npx, 'norm', **kwargs)
    x = mx.np.random.normal(0, 10, shape).astype(dtype)
    op_export_test('norm', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (2,3), (4,5,6)])
def test_onnx_export_np_logical_not(tmp_path, dtype, shape):
    M = def_model(mx.np, 'logical_not')
    x = mx.np.random.uniform(-1, 1, shape).astype(dtype)
    op_export_test('logical_not', M, [x], tmp_path)


@pytest.mark.skip(reason='TODO random uniform_like changed spec in 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_random_uniform_like(tmp_path, dtype, shape):
    M = def_model('random.uniform_like')
    low = -10
    high = 10
    x = mx.nd.zeros(shape=shape).astype(dtype)
    def rand_check(out):
        for i in out:
            if i.any() < low or i.any() >= high:
                raise Exception("Invalid value")
        return np.zeros_like(out)
    def rand_check_nd(out):
        return rand_check(out.asnumpy())
    op_export_test('random.uniform_like', M, [x], tmp_path, mx_map=rand_check_nd, onnx_map=rand_check)


@pytest.mark.skip(reason='TODO random uniform changed spec in 2.0')
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_random_uniform(tmp_path, dtype, shape):
    low = -10
    high = 10
    M = def_model('random_uniform', low=low, high=high, shape=shape, dtype=dtype, dummy_input=True)
    x = mx.nd.array([1], dtype='float32')
    def rand_check(out):
        for i in out:
            if i.any() < low or i.any() >= high:
                raise Exception("Invalid value")
        return np.zeros_like(out)
    def rand_check_nd(out):
        return rand_check(out.asnumpy())
    op_export_test('random_uniform', M, [x], tmp_path, mx_map=rand_check_nd, onnx_map=rand_check, dummy_input=True)


@pytest.mark.skip(reason='TODO random normal changed spec in 2.0')
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
@pytest.mark.parametrize("loc", [None, 0, 1, 2])
@pytest.mark.parametrize("scale", [None, 1, 2])
def test_onnx_export_random_normal(tmp_path, dtype, loc, scale, shape):
    kwargs = {
        'dtype': dtype,
        'size': shape,
        'dummy_input': True
    }
    if loc is not None:
        kwargs['loc'] = loc
    if scale is not None:
        kwargs['scale'] = scale
    M = def_model(mx.np.random, 'normal', **kwargs)
    x = mx.np.array([1], dtype='float32')
    def rand_check(out):
        return np.zeros_like(out)
    def rand_check_nd(out):
        return rand_check(out.asnumpy())
    op_export_test('random_normal', M, [x], tmp_path, mx_map=rand_check_nd, onnx_map=rand_check, dummy_input=True)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("spatial_scale", [0.7, 1.0])
def test_onnx_export_npx_roi_pooling(tmp_path, dtype, spatial_scale):
    M = def_model(mx.npx, 'roi_pooling', pooled_size=(2,2), spatial_scale=spatial_scale)
    x = mx.np.arange(start=0, stop=48, dtype=dtype).reshape((1,1,8,6))
    y = mx.np.array([[0,0,0,4,4]], dtype=dtype)
    op_export_test('roi_pooling', M, [x, y], tmp_path)


@pytest.mark.skip(reason='rnn_param_concat is deprecated in 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(1,2,3), (1,10)])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_onnx_export_rnn_param_concat(tmp_path, dtype, shape, axis):
    kwargs = {}
    if axis is not None:
        kwargs['dim'] = axis
    M = def_model('_internal._rnn_param_concat', **kwargs)
    x = mx.nd.random.uniform(-1, 1, shape).astype(dtype)
    y = mx.nd.random.uniform(-1, 1, shape).astype(dtype)
    op_export_test('_internal._rnn_param_concat', M, [x, y], tmp_path)


@pytest.mark.skip(reason='size_array is deprecated in 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_size_array(tmp_path, dtype, shape):
    M = def_model(mx.np, 'size_array')
    x = mx.nd.random.uniform(-1, 1, shape).astype(dtype)
    op_export_test('size_array', M, [x], tmp_path)


@pytest.mark.skip(reason='sample_multinomial is deprecated in 2.0')
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("shape", [(1,5), (2,10), (4,5)])
@pytest.mark.parametrize("sample_shape", [(1), (2)])
def test_onnx_export_sample_multinomial(tmp_path, dtype, shape, sample_shape):
    kwargs = {}
    if sample_shape is not None:
        kwargs['shape'] = sample_shape
    M = def_model('sample_multinomial', **kwargs)
    a = mx.nd.random.uniform(0, 1, shape).astype(dtype)
    x = a/a.sum(axis=1, keepdims=1)
    def rand_check(out):
        return np.zeros_like(out)
    def rand_check_nd(out):
        return rand_check(out.asnumpy())
    op_export_test('sample_multinomial', M, [x], tmp_path, mx_map=rand_check_nd, onnx_map=rand_check)


@pytest.mark.skip(reason='split_v2 is deprecated in 2.0')
@pytest.mark.parametrize("dtype", ['float32', 'int32', 'int64'])
@pytest.mark.parametrize('params', [((2, 4, 6), (1, ), 0, True),
                                    ((4, 5, 6), (2, 4), 1, False),
                                    ((4, 5, 6, 7), (0, 2, 4), 2, False),
                                    ((4, 5, 6, 7), 3, -2, False),
                                    ((2, 6, 8), 8, -1, True)])
def test_onnx_export_split_v2(tmp_path, dtype, params):
    from onnx.defs import onnx_opset_version
    if onnx_opset_version() < 13 and not isinstance(params[1], int):
        # opset12 only supports sections. indices is supported since opset13
        return
    M = def_model('split_v2', indices_or_sections=params[1], axis=params[2], squeeze_axis=params[3])
    x = mx.nd.random.uniform(0, 10, params[0]).astype(dtype)
    op_export_test('split_v2', M, [x], tmp_path)
