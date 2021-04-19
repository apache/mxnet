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

def def_model(op_name, dummy_input=False, **params):
    class Model(HybridBlock):
        def __init__(self, **kwargs):
            super(Model, self).__init__(**kwargs)

        def hybrid_forward(self, F, *inputs):
            names = op_name.split('.')
            func = F
            for name in names:
                func = getattr(func, name)
            if dummy_input:
                return func(**params), inputs[0]
            else:
                return func(*inputs, **params)
    return Model

def def_model_from_func(func, dummy_input=False, **params):
    class Model(HybridBlock):
        def __init__(self, **kwargs):
            super(Model, self).__init__(**kwargs)

        def hybrid_forward(self, F, *inputs):
            if dummy_input:
                return func(**params), inputs[0]
            else:
                return func(*inputs, **params)
    return Model

def op_export_test(model_name, Model, inputs, tmp_path, dummy_input=False, onnx_map=None, mx_map=None):
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
    onnx_file = export_to_onnx(model, model_name, inputs)
    pred_onx = onnx_rt(onnx_file, inputs)
    if dummy_input:
        pred_mx = pred_mx[0]
    if isinstance(pred_mx, list):
        for i in range(len(pred_mx)):
            pred_onx_i = onnx_map(pred_onx[i]) if onnx_map else pred_onx[i]
            pred_mx_i = mx_map(pred_mx[i]) if mx_map else pred_mx[i]
            assert_almost_equal(pred_onx_i, pred_mx_i, equal_nan=True)
    else:
        pred_onx = onnx_map(pred_onx[0]) if onnx_map else pred_onx[0]
        pred_mx = mx_map(pred_mx) if mx_map else pred_mx
        assert_almost_equal(pred_onx, pred_mx, equal_nan=True)


def test_onnx_export_abs(tmp_path):
    M = def_model('abs')
    x = mx.nd.array([[-2, -1], [0, 99]], dtype='float32')
    op_export_test('abs', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'float16', 'int32', 'int64'])
@pytest.mark.parametrize('params', [[(0, 1), (2,3), (1, 1)],
                                    [(None, 1), (2, None), None],
                                    [(0, 0, 0), (None, 4, 5), (None, 1, 2)]])
def test_onnx_export_slice(tmp_path, dtype, params):
    M = def_model('slice', begin=params[0], end=params[1], step=params[2])
    x = mx.nd.arange(start=0, stop=60, dtype=dtype).reshape((3, 4, 5))
    op_export_test('slice', M, [x], tmp_path)


def test_onnx_export_stack(tmp_path):
    M = def_model('stack')
    x = mx.nd.array([1, 2], dtype='float32')
    y = mx.nd.array([3, 4], dtype='float32')
    op_export_test('stack', M, [x, y], tmp_path)

@pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(1), (1,2), (2,3,4), (5,6,7)])
def test_onnx_export_zeros(tmp_path, dtype, shape):
    M = def_model('zeros', shape=shape, dtype=dtype, dummy_input=True)
    x = mx.nd.array([1])
    op_export_test('zeros', M, [x], tmp_path, dummy_input=True)

@pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(1), (1,2), (2,3,4), (5,6,7)])
def test_onnx_export_ones(tmp_path, dtype, shape):
    M = def_model('ones', shape=shape, dtype=dtype, dummy_input=True)
    x = mx.nd.array([0])
    op_export_test('ones', M, [x], tmp_path, dummy_input=True)


@pytest.mark.parametrize('dtype', [None, 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1), (1,2), (2,3,4), (5,6,7)])
def test_onnx_export_zeros_like(tmp_path, dtype, shape):
    M = def_model('zeros_like', dtype=dtype)
    x = mx.random.uniform(0, 1, shape, dtype='float32')
    op_export_test('zeros_like', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', [None, 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1), (1,2), (2,3,4), (5,6,7)])
def test_onnx_export_ones_like(tmp_path, dtype, shape):
    M = def_model('ones_like', dtype=dtype)
    x = mx.random.uniform(0, 1, shape, dtype='float32')
    op_export_test('ones_like', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("axis", [None,0,1])
@pytest.mark.parametrize("start", [0, 0.5, 1])
@pytest.mark.parametrize("step", [0.01, 0.1, 0.5, 1])
@pytest.mark.parametrize("test_data", [ mx.random.uniform(0, 1, (10,20)), [[0,1,2,3,4,5],[4,5,6,7,8,9],[8,9,10,11,12,13]]])
def test_onnx_export_arange_like(tmp_path, dtype, axis, start, step, test_data):
    M = def_model('contrib.arange_like', axis=axis, start=start, step=step)
    x = mx.nd.array(test_data, dtype=dtype)
    op_export_test('arange_like', M, [x], tmp_path)


@pytest.mark.parametrize("params", [[0, 2, 1], [0, 50, 0.25], [-100, 100, 0.5], [5, None, 1], [-5, None, -1]])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_onnx_export_arange(tmp_path, dtype, params):
    start, stop, step = params[0], params[1], params[2]
    if "int" in dtype:
        start = int(start)
        stop = int(stop) if stop != None else None
        step = int(step)
        if step == 0:
            step = 1
    M = def_model('arange', dummy_input=True, start=start, stop=stop, step=step, dtype=dtype)
    x = mx.nd.array([1], dtype='float32')
    op_export_test('arange', M, [x], tmp_path, dummy_input=True)


@pytest.mark.parametrize('dtype', ['float32'])
def test_onnx_export_layernorm(tmp_path, dtype):
    x = mx.nd.random.uniform(1, 2, (3, 4, 5), dtype=dtype)
    axes = list(range(np.shape(np.shape(x))[0]))
    axes.append(-1)
    for axis in axes:
        M = def_model('LayerNorm', axis=axis)
        gamma = mx.random.uniform(0, 1, [np.shape(x)[axis]], dtype=dtype)
        beta = mx.random.uniform(0, 1, [np.shape(x)[axis]], dtype=dtype)
        op_export_test('LayerNorm', M, [x, gamma, beta], tmp_path)


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


#TODO: onnxruntime does not support float64 for Where
@pytest.mark.parametrize('dtype', ['float32'])
def test_onnx_export_SequenceMask(tmp_path, dtype):
    M1 = def_model('SequenceMask', use_sequence_length=True, axis=1, value=-5)
    M2 = def_model('SequenceMask', use_sequence_length=True, axis=0, value=-99)
    x = mx.nd.array([[[[  1.,   2.,   3.,  3.5]],
                      [[  4.,   5.,   6.,  6.5]]],
                     [[[  7.,   8.,   9.,  9.5]],
                      [[ 10.,  11.,  12., 12.5]]],
                     [[[ 13.,  14.,  15., 15.5]],
                      [[ 16.,  17.,  18., 18.5]]]], dtype=dtype)
    seq_len1 = mx.nd.array([1, 2, 1], dtype=dtype)
    seq_len2 = mx.nd.array([1, 2], dtype=dtype)
    op_export_test('SequenceMask_1', M1, [x, seq_len1], tmp_path)
    op_export_test('SequenceMask_2', M2, [x, seq_len2], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
def test_onnx_export_contrib_interleaved_matmul_selfatt_qk(tmp_path, dtype):
    M1 = def_model('contrib.interleaved_matmul_selfatt_qk', heads=3)
    x1 = mx.nd.random.uniform(0, 1, (3, 3, 3*3*3), dtype=dtype)
    op_export_test('contrib_interleaved_matmul_selfatt_qk_1', M1, [x1], tmp_path)
    M2 = def_model('contrib.interleaved_matmul_selfatt_qk', heads=5)
    x2 = mx.nd.random.uniform(0, 1, (7, 5, 4*5*6), dtype=dtype)
    op_export_test('contrib_interleaved_matmul_selfatt_qk_2', M2, [x2], tmp_path)

@pytest.mark.parametrize('dtype', ['float32'])
def test_onnx_export_contrib_interleaved_matmul_selfatt_valatt(tmp_path, dtype):
    M = def_model('contrib.interleaved_matmul_selfatt_valatt', heads=6)
    x = mx.nd.random.uniform(0, 1, (4, 5, 6*7*3), dtype=dtype)
    att = mx.nd.random.uniform(0, 1, (5*6, 4, 4), dtype=dtype)
    op_export_test('contrib_interleaved_matmul_selfatt_valatt', M, [x, att], tmp_path)


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
def test_onnx_export_reshape(tmp_path, dtype):
    x = mx.nd.ones((2, 3, 4, 5, 6), dtype=dtype)
    M1 = def_model('reshape', shape=(6, 1, 0, -1))
    op_export_test('reshape_1', M1, [x], tmp_path)
    M2 = def_model('reshape', shape=(3, -1, 0, 0), reverse=True)
    op_export_test('reshape_2', M2, [x], tmp_path)
    M3 = def_model('reshape', shape=(5, 1, 1, 1, 1, 0 -1, 0), reverse=True)
    op_export_test('reshape_3', M3, [x], tmp_path)


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
def test_onnx_export_embedding(tmp_path, dtype):
    x = mx.nd.array([[ 1.,  3.],
                     [ 0.,  2.]], dtype=dtype)
    y = mx.nd.array([[  0.,   1.,   2.,   3.,   4.],
                     [  5.,   6.,   7.,   8.,   9.],
                     [ 10.,  11.,  12.,  13.,  14.],
                     [ 15.,  16.,  17.,  18.,  19.]], dtype=dtype)
    M = def_model('Embedding', input_dim=4, output_dim=5)
    op_export_test('Embedding', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('num_hidden', [1, 2, 7, 10, 20])
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_onnx_export_fully_connected(tmp_path, dtype, num_hidden, no_bias, flatten):
    M = def_model('FullyConnected', num_hidden=num_hidden, no_bias=no_bias, flatten=flatten)
    x = mx.nd.random.uniform(-0.5, 0.5, (3, 4, 5))
    if (flatten):
        weight = mx.nd.random.uniform(0, 1, (num_hidden, 4*5))
    else:
        weight = mx.nd.random.uniform(0, 1, (num_hidden, 5))
    args = [x, weight]
    if not no_bias:
        args.append(mx.nd.random.uniform(0,1,(num_hidden,)))
    op_export_test('FullyConnected', M, args, tmp_path)


#TODO: onnxruntime does not support float64 for the relu opertors
@pytest.mark.parametrize('dtype', ['float32', 'float16'])
@pytest.mark.parametrize('shape', [(1,), (3,), (4, 5), (3, 4, 5)])
@pytest.mark.parametrize('act_type', ['elu', 'leaky', 'prelu', 'selu', 'gelu'])
def test_onnx_export_LeakyReLU(tmp_path, dtype, shape, act_type):
    M = def_model('LeakyReLU', act_type='leaky')
    x = mx.nd.random.uniform(-0.5, 0.5, shape=shape, dtype=dtype)
    op_export_test('LeakyReLU', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'float16', 'int32', 'int64'])
def test_onnx_export_Concat(tmp_path, dtype):
    x = mx.nd.array([[1,1],[2,2]], dtype=dtype)
    y = mx.nd.array([[3,3],[4,4],[5,5]], dtype=dtype)
    z = mx.nd.array([[6,6],[7,7],[8,8]], dtype=dtype)
    M1 = def_model('Concat', dim=0)
    M2 = def_model('Concat', dim=1)
    op_export_test('Concat_1', M1, [x, y, z], tmp_path)
    op_export_test('Concat_2', M2, [y, z], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float16'])
@pytest.mark.parametrize('shape', [(1,), (3,), (4, 5), (3, 4, 5)])
@pytest.mark.parametrize('act_type', ['tanh', 'relu', 'sigmoid', 'softrelu', 'softsign'])
def test_onnx_export_Activation(tmp_path, dtype, shape, act_type):
    M = def_model('Activation', act_type=act_type)
    x = mx.nd.random.uniform(-0.5, 0.5, shape=shape, dtype=dtype)
    op_export_test('Activation', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('axes', [None, [1,0,2]])
def test_onnx_export_transpose(tmp_path, dtype, axes):
    if axes != None:
        M = def_model('transpose', axes=axes)
    else:
        M = def_model('transpose')
    x = mx.nd.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=dtype)
    op_export_test('transpose', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_onnx_export_expand_dims(tmp_path, dtype, axis):
    M = def_model('expand_dims', axis=axis)
    x = mx.nd.random.uniform(0, 1, (2,3,4), dtype=dtype)
    op_export_test('expand_dims', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_onnx_export_broadcast_add(tmp_path, dtype):
    M = def_model('broadcast_add')
    x = mx.nd.array([[1,1,1],[1,1,1]], dtype=dtype)
    y = mx.nd.array([[0],[1]], dtype=dtype)
    op_export_test('broadcast_add', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_onnx_export_broadcast_equal(tmp_path, dtype):
    M = def_model('broadcast_equal')
    x = mx.nd.zeros((4,5,6), dtype=dtype)
    y = mx.nd.ones((4,5,6), dtype=dtype)
    op_export_test('broadcast_equal', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
def test_onnx_export_broadcast_minimum(tmp_path, dtype):
    M = def_model('broadcast_minimum')
    if 'int' in dtype:
        x = mx.nd.random.randint(0, 1000, (4, 5, 6), dtype=dtype)
        y = mx.nd.random.randint(0, 1000, (4, 5, 6), dtype=dtype)
    else:
        x = mx.nd.random.uniform(0, 1000, (4, 5, 6), dtype=dtype)
        y = mx.nd.random.uniform(0, 1000, (4, 5, 6), dtype=dtype)
    op_export_test('broadcast_minimum', M, [x, y], tmp_path)

@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('axis', [0, 1, 2, -1])
def test_onnx_export_stack(tmp_path, dtype, axis):
    M = def_model('stack', axis=axis)
    if 'int' in dtype:
        x = mx.nd.random.randint(0, 10*9, (3,4,5), dtype=dtype)
        y = mx.nd.random.randint(0, 10*9, (3,4,5), dtype=dtype)
    else:
        x = mx.nd.random.normal(0, 10*9, (3,4,5), dtype=dtype)
        y = mx.nd.random.normal(0, 10*9, (3,4,5), dtype=dtype)
    op_export_test('stack', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('p', [0.1, 0.2, 0.5, 0.8])
def test_onnx_export_dropout(tmp_path, dtype, p):
    M = def_model('Dropout', p=p)
    x = mx.nd.array([[3,0.5,-0.5,2,7],[2,-0.4,7,3,0.2]], dtype=dtype)
    op_export_test('Dropout', M, [x], tmp_path)


@pytest.mark.parametrize('src_dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('dst_dtype', ['bool', 'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'])
@pytest.mark.parametrize('shape', [(2,3), (4,5,6)])
def test_onnx_export_cast(tmp_path, src_dtype, dst_dtype, shape):
    M = def_model('Cast', dtype=dst_dtype)
    x = mx.nd.ones(shape, dtype=src_dtype)
    op_export_test('Cast', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('temperature', [None, .1, 1., 10.])
def test_onnx_export_softmax(tmp_path, dtype, temperature):
    x = mx.nd.random.uniform(0, 1, (4, 5, 6), dtype=dtype)
    M1 = def_model('softmax')
    op_export_test('softmax_1', M1, [x], tmp_path)
    M2 = def_model('softmax', use_length=True, axis=0, temperature=temperature)
    l2 = mx.random.uniform(0, 4, (5, 6)).astype('int32')
    op_export_test('softmax_2', M2, [x, l2], tmp_path)
    M3 = def_model('softmax', use_length=True, axis=-1, temperature=temperature)
    # note that the axis==-1 case uses negative value masking + ONNX softmax
    # when valid_len==0 the masked values will NOT be 0
    l3 = mx.random.uniform(1, 6, (4, 5)).astype('int32')
    op_export_test('softmax_3', M3, [x, l3], tmp_path)
    M4 = def_model('softmax', use_length=True, axis=1, temperature=temperature)
    l4 = mx.random.uniform(0, 5, (4, 6)).astype('int32')
    op_export_test('softmax_4', M4, [x, l4], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('axis', [0, 1, 2, 3])
def test_onnx_export_reverse(tmp_path, dtype, axis):
    x = mx.nd.arange(0, 120, dtype=dtype).reshape((2, 3, 4, 5))
    M = def_model('reverse', axis=axis)
    op_export_test('reverse', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('axis', [None, 0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize('repeats', [2, 1, 3])
def test_onnx_export_repeat(tmp_path, dtype, axis, repeats):
    x = mx.nd.arange(0, 27, dtype=dtype).reshape((3, 3, 3))
    M = def_model('repeat', axis=axis, repeats=repeats)
    op_export_test('repeat', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('params', [{'height': 7, 'width': 13},
                                    {'height': 10, 'width': 16},
                                    {'height': 3, 'width': 5},
                                    {'height': 2, 'width': 4},
                                    {'scale_height': 3, 'scale_width': 2},
                                    {'scale_height': 1.7, 'scale_width': 2.3},
                                    {'scale_height': 0.5, 'scale_width': 0.6},
                                    {'scale_height': 0.8, 'scale_width': 0.1},
                                    {'scale_height': 2.5, 'scale_width': 0.5},
                                    {'scale_height': 3, 'scale_width': 0.00001},
                                    ])
def test_onnx_export_contrib_BilinearResize2D(tmp_path, dtype, params):
    x = mx.nd.arange(0, 160).reshape((2, 2, 5, 8))
    M = def_model('contrib.BilinearResize2D', **params)
    op_export_test('contrib_BilinearResize2D', M, [x], tmp_path)


@pytest.mark.parametrize('topk', [2, 3, 4])
@pytest.mark.parametrize('valid_thresh', [0.3, 0.4, 0.8])
@pytest.mark.parametrize('overlap_thresh', [0.4, 0.7, 1.0])
def test_onnx_export_contrib_box_nms(tmp_path, topk, valid_thresh, overlap_thresh):
    # Note that ONNX NMS op only supports float32

    # Also note that onnxruntime's nms has slightly different implementation in handling
    # overlaps and score ordering when certain boxes are suppressed than that of mxnet
    # the following test tensors are manually tweaked to avoid such diferences
    # The purpose of theses tests cases are to show that the high level conversion logic is
    # laid out correctly

    A = mx.nd.array([[
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
    M = def_model('contrib.box_nms', coord_start=1, force_suppress=True,
                  overlap_thresh=overlap_thresh, valid_thresh=valid_thresh, score_index=0,
                  topk=topk, in_format='corner', out_format='corner')
    op_export_test('contrib_nms_manual_coner', M, [A], tmp_path)
    
    B = mx.nd.array([
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
    M = def_model('contrib.box_nms', coord_start=1, force_suppress=True,
                  overlap_thresh=overlap_thresh, valid_thresh=valid_thresh, score_index=0,
                  topk=topk, in_format='center', out_format='center')
    op_export_test('contrib_nms_manual_center', M, [B], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("scalar", [0., 0.1, 0.5, 1., 5, 555.])
def test_onnx_export_greater_scalar(tmp_path, dtype, scalar):
    if 'int' in dtype:
        scalar = int(scalar)
        x = mx.nd.arange(0, 12, dtype=dtype).reshape((3, 4))
    else:
        x = mx.random.uniform(0, 9999, (5,10), dtype=dtype)
    M = def_model('_internal._greater_scalar', scalar=scalar)
    op_export_test('_internal._greater_scalar', M, [x], tmp_path)


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


@pytest.mark.parametrize("dtype", ["float16", "float32", "int32", "int64"])
@pytest.mark.parametrize("shape", [(1,1), (3,3), (10,2), (20,30,40)])
def test_onnx_export_where(tmp_path, dtype, shape):
    M = def_model('where')
    x = mx.nd.zeros(shape, dtype=dtype)
    y = mx.nd.ones(shape, dtype=dtype)
    cond = mx.nd.random.randint(low=0, high=1, shape=shape, dtype='int32')
    op_export_test('where', M, [cond, x, y], tmp_path)


# onnxruntime does not seem to support float64 and int32
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int64'])
@pytest.mark.parametrize('axis', [0, 2, -1, -2, -3])
@pytest.mark.parametrize('is_ascend', [True, False, 0, 1, None])
@pytest.mark.parametrize('k', [1, 4])
@pytest.mark.parametrize('dtype_i', ['float32', 'int32', 'int64'])
@pytest.mark.parametrize('ret_typ', ['value', 'indices', 'both'])
def test_onnx_export_topk(tmp_path, dtype, axis, is_ascend, k, dtype_i, ret_typ):
    A = mx.random.uniform(0, 100, (4, 5, 6)).astype(dtype)
    kwargs = {}
    if is_ascend is not None:
        kwargs['is_ascend'] = is_ascend
    M = def_model('topk', axis=axis, k=k, dtype=dtype_i, ret_typ=ret_typ, **kwargs)
    op_export_test('topk', M, [A], tmp_path)


def test_onnx_link_op_with_multiple_outputs(tmp_path):
    A = mx.random.uniform(0, 100, (4, 5, 6))
    class Model1(HybridBlock):
        def __init__(self, **kwargs):
            super(Model1, self).__init__(**kwargs)

        def hybrid_forward(self, F, x):
            out1, out2 = F.topk(x, k=3, ret_typ='both')
            out11 = out1 ** 2
            out22 = out2 ** 3
            return out11, out22
    op_export_test('link_op_with_multiple_outputs_case1', Model1, [A], tmp_path)

    class Model2(HybridBlock):
        def __init__(self, **kwargs):
            super(Model2, self).__init__(**kwargs)

        def hybrid_forward(self, F, x):
            out_ = F.topk(x, k=3, ret_typ='value')
            out = out_ ** 3
            return out
    op_export_test('link_op_with_multiple_outputs_case2', Model2, [A], tmp_path)

    class Model3(HybridBlock):
        def __init__(self, **kwargs):
            super(Model3, self).__init__(**kwargs)

        def hybrid_forward(self, F, x):
            out_ = F.topk(x, k=3, ret_typ='indices')
            out = out_ ** 3
            return out
    op_export_test('link_op_with_multiple_outputs_case3', Model3, [A], tmp_path)


# opset 8 MAX only supports float types
# opset 12 and up suppots float and int
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (1, 4, 1, 7)])
def test_onnx_maximum_scalar(tmp_path, dtype, shape):
    x = mx.random.uniform(0, 10, shape).astype(dtype)
    M = def_model('maximum', right=5)
    op_export_test('_maximum_scalar', M, [x], tmp_path)


# opset 8 Min only supports float types
# opset 12 and up suppots float and int
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (1, 4, 1, 7)])
def test_onnx_minimum_scalar(tmp_path, dtype, shape):
    x = mx.random.uniform(0, 10, shape).astype(dtype)
    M = def_model('minimum', right=5)
    op_export_test('_minimum_scalar', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('fmt', ['corner', 'center'])
@pytest.mark.parametrize('clip', [-1., 0., .5, 5.])
def test_onnx_export_contrib_box_decode(tmp_path, dtype, fmt, clip):
    # ensure data[0] < data[2] and data[1] < data[3] for corner format
    mul = mx.nd.array([-1, -1, 1, 1], dtype=dtype)
    data = mx.nd.random.uniform(0, 1, (2, 3, 4), dtype=dtype) * mul
    anchors = mx.nd.random.uniform(0, 1, (1, 3, 4), dtype=dtype) * mul
    M1 = def_model('contrib.box_decode', format=fmt, clip=clip)
    op_export_test('contrib_box_decode', M1, [data, anchors], tmp_path)
    M2 = def_model('contrib.box_decode', format=fmt, clip=clip, std0=0.3, std1=1.4, std2=0.5, std3=1.6)
    op_export_test('contrib_box_decode', M1, [data, anchors], tmp_path)


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
def test_onnx_export_broadcast_mod(tmp_path, dtype, shapes):
    A = mx.nd.random.uniform(-300, 300, shapes[0]).astype(dtype)
    B = mx.nd.random.uniform(-30, 30, shapes[1]).astype(dtype)
    # test when dividend is zero
    B[-1] = 0
    M = def_model('broadcast_mod')
    op_export_test('broadcast_mod', M, [A, B], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
def test_onnx_export_reshape_like(tmp_path, dtype):
    if 'int' in dtype:
        x = mx.nd.random.randint(0, 10, (2, 2, 3, 2), dtype=dtype)
        y = mx.nd.random.randint(0, 10, (1, 4, 3, 2), dtype=dtype)
    else:
        x = mx.nd.random.normal(0, 10, (2, 2, 3, 2), dtype=dtype)
        y = mx.nd.random.normal(0, 10, (1, 4, 3, 2), dtype=dtype)
    M1 = def_model('reshape_like')
    op_export_test('reshape_like1', M1, [x, y], tmp_path)
    M2 = def_model('reshape_like', lhs_begin=0, lhs_end=2, rhs_begin=1, rhs_end=2)
    op_export_test('reshape_like2', M2, [x, y], tmp_path)
    M3 = def_model('reshape_like', lhs_begin=-4, lhs_end=-2, rhs_begin=-3, rhs_end=-2)
    op_export_test('reshape_like3', M3, [x, y], tmp_path)
    M4 = def_model('reshape_like', lhs_begin=0, lhs_end=None, rhs_begin=1, rhs_end=None)
    op_export_test('reshape_like4', M4, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
def test_onnx_export_gather_nd(tmp_path, dtype):
    # y[0] == dim(x)
    x1 = mx.random.uniform(-100, 100, (4, 5, 6, 7)).astype(dtype)
    y1 = mx.random.randint(-4, 4, (4, 4, 4)).astype(dtype)
    M1 = def_model('gather_nd')
    op_export_test('gather_nd1', M1, [x1, y1], tmp_path)
    # y[0] < dim(x)
    x2 = mx.random.uniform(-100, 100, (4, 5, 6, 7)).astype(dtype)
    y2 = mx.random.randint(-4, 4, (2,3,4)).astype(dtype)
    M2 = def_model('gather_nd')
    op_export_test('gather_nd2', M2, [x2, y2], tmp_path)


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
def test_onnx_export_swap_axis(tmp_path, dtype, params):
    shape = params[0]
    dim1, dim2 = params[1]
    x = mx.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model('SwapAxis', dim1=dim1, dim2=dim2)
    op_export_test('SwapAxis', M, [x], tmp_path)


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
def test_onnx_export_slice_channel(tmp_path, dtype, axis, num_outputs):
    x = mx.nd.zeros((10,20,30,40), dtype=dtype)
    if axis is None:
        M = def_model('SliceChannel', num_outputs=num_outputs)
    else:
        M = def_model('SliceChannel', axis=axis, num_outputs=num_outputs)
    op_export_test('SliceChannel', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('lhs_axes', [[1, 3], [3, 1], [-2, -4], [-4, -2]])
@pytest.mark.parametrize('rhs_axes', [[1, 3], [3, 1], [-2, -4], [-4, -2]])
def test_onnx_export_broadcast_like(tmp_path, dtype, lhs_axes, rhs_axes):
    x = mx.random.normal(0, 10, (2, 1, 1, 1, 6)).astype(dtype)
    y = mx.random.normal(0, 10, (2, 3, 4, 5, 6)).astype(dtype)
    M1 = def_model('broadcast_like')
    op_export_test('broadcast_like1', M1, [x, y], tmp_path)
    M2 = def_model('broadcast_like', lhs_axes=lhs_axes, rhs_axes=rhs_axes)
    op_export_test('broadcast_like2', M2, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('pooled_size', [(1, 1), (3, 3), (14, 14), (5, 7)])
@pytest.mark.parametrize('spatial_scale', [1, 0.5, 0.0625])
@pytest.mark.parametrize('spatial_ratio', [1, 2, 3, 5])
def test_onnx_export_contrib_ROIAlign(tmp_path, dtype, pooled_size, spatial_scale, spatial_ratio):
    data = mx.random.uniform(0, 1, (5, 3, 128, 128)).astype(dtype)
    rois = mx.nd.array([[0, 0, 0, 63, 63],
                        [1, 34, 52, 25, 85],
                        [2, 50, 50, 100, 100],
                        [3, 0, 0, 127, 127],
                        [4, 12, 84, 22, 94],
                        [0, 0, 0, 1, 1]]).astype(dtype)
    M = def_model('contrib.ROIAlign', pooled_size=pooled_size, spatial_scale=spatial_scale,
                  sample_ratio=spatial_ratio)
    op_export_test('_contrib_ROIAlign', M, [data, rois], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('transpose_a', [True, False])
@pytest.mark.parametrize('transpose_b', [True, False])
def test_onnx_export_batch_dot(tmp_path, dtype, transpose_a, transpose_b):
    x1 = mx.nd.random.normal(0, 10, (2, 3, 4, 5, 6), dtype=dtype)
    y1 = mx.nd.random.normal(0, 10, (2, 3, 4, 6, 5), dtype=dtype)
    M1 = def_model('batch_dot')
    op_export_test('batch_dot1', M1, [x1, y1], tmp_path)
    x2 = mx.nd.random.normal(0, 10, (2, 3, 4, 5, 5), dtype=dtype)
    y2 = mx.nd.random.normal(0, 10, (2, 3, 4, 5, 5), dtype=dtype)
    M2 = def_model('batch_dot', transpose_a=transpose_a, transpose_b=transpose_b)
    op_export_test('batch_dot2', M2, [x2, y2], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 1, 60, 60)])
@pytest.mark.parametrize('count_include_pad', [True, False])
@pytest.mark.parametrize('pooling_convention', ['full', 'valid'])
@pytest.mark.parametrize('kernel', [(3, 3), (4, 5), (14, 14)])
@pytest.mark.parametrize('stride', [None, (1, 1), (2, 2), (3, 4), (4, 5)])
@pytest.mark.parametrize('pad', [None, (1, 1), (3, 4), (4, 5)])
def test_onnx_export_pooling_avg(tmp_path, dtype, shape, count_include_pad, pooling_convention,
                                 kernel, stride, pad):
    # mxnet and onnxruntime has different implementation of count_include_pad on the left column
    # and bottom row
    if pooling_convention == 'full' and count_include_pad == True:
        return
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1]):
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model('Pooling', count_include_pad=count_include_pad, pool_type='avg',
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
def test_onnx_export_pooling_avg_3d(tmp_path, dtype, shape, count_include_pad, pooling_convention,
                                    kernel, stride, pad):
    # mxnet and onnxruntime has different implementation of count_include_pad on the left column
    # and bottom row
    if pooling_convention == 'full' and count_include_pad == True:
        return
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1] or pad[2] >= kernel[2]):
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model('Pooling', count_include_pad=count_include_pad, pool_type='avg',
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
def test_onnx_export_pooling_max(tmp_path, dtype, shape, pooling_convention, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1]):
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model('Pooling', pool_type='max', pooling_convention=pooling_convention,
                  layout='NCHW', **kwargs)
    op_export_test('pooling_max', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 16, 16, 16), (1, 1, 10, 18, 18)])
@pytest.mark.parametrize('pooling_convention', ['full', 'valid'])
@pytest.mark.parametrize('kernel', [(1, 1, 1), (3, 3, 3), (1, 7, 7)])
@pytest.mark.parametrize('stride', [None, (1, 1, 1), (1, 2, 3)])
@pytest.mark.parametrize('pad', [None, (0, 1, 1), (1, 2, 3)])
def test_onnx_export_pooling_max_3d(tmp_path, dtype, shape, pooling_convention, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1] or pad[2] >= kernel[2]):
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model('Pooling', pool_type='max', pooling_convention=pooling_convention,
                  layout='NCDHW', **kwargs)
    op_export_test('pooling_max_3d', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 1, 60, 60)])
@pytest.mark.parametrize('p_value', [1, 2])
@pytest.mark.parametrize('kernel', [(3, 3), (4, 5), (14, 14)])
@pytest.mark.parametrize('stride', [None, (1, 1), (2, 2), (3, 4), (4, 5)])
@pytest.mark.parametrize('pad', [None, (1, 1), (3, 4), (4, 5)])
def test_onnx_export_pooling_lp(tmp_path, dtype, shape, p_value, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1]):
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model('Pooling', pool_type='lp', pooling_convention='valid',
                  p_value=p_value, layout='NCHW', **kwargs)
    op_export_test('pooling_lp', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 16, 16, 16), (1, 1, 10, 18, 18)])
@pytest.mark.parametrize('p_value', [1, 2])
@pytest.mark.parametrize('kernel', [(1, 1, 1), (3, 3, 3), (1, 7, 7)])
@pytest.mark.parametrize('stride', [None, (1, 1, 1), (1, 2, 3)])
@pytest.mark.parametrize('pad', [None, (0, 1, 1), (1, 2, 3)])
def test_onnx_export_pooling_lp_3d(tmp_path, dtype, shape, p_value, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1] or pad[2] >= kernel[2]):
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    M = def_model('Pooling', pool_type='lp', pooling_convention='valid',
                  p_value=p_value, layout='NCDHW', **kwargs)
    op_export_test('pooling_lp_3d', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 1, 60, 60)])
@pytest.mark.parametrize('pool_type', ['avg', 'max', 'lp'])
@pytest.mark.parametrize('p_value', [1, 2])
@pytest.mark.parametrize('kernel', [(3, 3), (14, 14)])
@pytest.mark.parametrize('stride', [None, (3, 4)])
@pytest.mark.parametrize('pad', [None, (3, 4)])
def test_onnx_export_pooling_global(tmp_path, dtype, shape, pool_type, p_value, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1]):
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    # kernel, stride, and pad should have no effect on the results
    M = def_model('Pooling', global_pool=True, pool_type=pool_type, pooling_convention='valid',
                  p_value=p_value, layout='NCHW', **kwargs)
    op_export_test('pooling_global', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 16, 16, 16), (1, 1, 10, 18, 18)])
@pytest.mark.parametrize('pool_type', ['avg', 'max', 'lp'])
@pytest.mark.parametrize('p_value', [1, 2])
@pytest.mark.parametrize('kernel', [(1, 1, 1), (3, 3, 3)])
@pytest.mark.parametrize('stride', [None, (1, 1, 1)])
@pytest.mark.parametrize('pad', [None, (0, 1, 1)])
def test_onnx_export_pooling_global_3d(tmp_path, dtype, shape, pool_type, p_value, kernel, stride, pad):
    # onnxruntime requires that pad is smaller than kernel
    if pad and (pad[0] >= kernel[0] or pad[1] >= kernel[1] or pad[2] >= kernel[2]):
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    # kernel, stride, and pad should have no effect on the results
    M = def_model('Pooling', global_pool=True, pool_type=pool_type, pooling_convention='valid',
                  p_value=p_value, layout='NCDHW', **kwargs)
    op_export_test('pooling_global_3d', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_onnx_export_log2(tmp_path, dtype):
    x = mx.random.normal(0, 10, (2, 3, 4, 5)).astype(dtype)
    M = def_model('log2')
    op_export_test('log2', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('axis', [None, 1, [1,2], -1])
def test_onnx_export_sum(tmp_path, dtype, axis):
    if 'int' in dtype:
        x = mx.nd.random.randint(0, 10, (5, 6, 7, 8), dtype=dtype)
    else:
        x = mx.nd.random.normal(0, 10, (5, 6, 7, 8), dtype=dtype)
    if axis is not None:
        M = def_model('sum', axis=axis)
    else:
        M = def_model('sum')
    op_export_test('sum', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
def test_onnx_export_broadcast_mul(tmp_path, dtype):
    M = def_model('broadcast_mul')
    x = mx.nd.array([[1,2,3],[4,5,6]], dtype=dtype)
    y = mx.nd.array([[0],[3]], dtype=dtype)
    op_export_test('broadcast_mul', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (1, 2, 3, 2, 1)])
@pytest.mark.parametrize('p', [0, 0.1, 0.5, 1])
def test_onnx_export_dropout(tmp_path, dtype, shape, p):
    x = mx.random.uniform(-100, 100, shape=shape).astype(dtype)
    M = def_model('Dropout', p=p)
    op_export_test('Dropuout', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (2, 6, 60, 60)])
@pytest.mark.parametrize('num_filter', [2, 4, 32])
@pytest.mark.parametrize('num_group', [1, 2])
@pytest.mark.parametrize('no_bias', [True, False])
@pytest.mark.parametrize('kernel', [(3, 3), (4, 5), (14, 14)])
@pytest.mark.parametrize('stride', [None, (1, 1), (2, 2), (3, 4), (4, 5)])
@pytest.mark.parametrize('pad', [None, (1, 1), (3, 4), (4, 5)])
@pytest.mark.parametrize('dilate', [None, (1, 1)])
def test_onnx_export_convolution(tmp_path, dtype, shape, num_filter, num_group, no_bias,
                                 kernel, stride, pad, dilate):
    if shape[1] % num_group:
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    w_shape = (num_filter,) + (shape[1] // num_group,) + kernel
    w = mx.random.uniform(0, 1, w_shape, dtype=dtype)
    b_shape = (num_filter)
    b = mx.random.uniform(0, 1, b_shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    if dilate:
        kwargs['dilate'] = dilate
    M = def_model('Convolution', num_filter=num_filter, num_group=num_group,  no_bias=no_bias,
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
def test_onnx_export_convolution_3D(tmp_path, dtype, shape, num_filter, num_group, no_bias,
                                 kernel, stride, pad, dilate):
    if shape[1] % num_group:
        return
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    w_shape = (num_filter,) + (shape[1] // num_group,) + kernel
    w = mx.random.uniform(0, 1, w_shape, dtype=dtype)
    b_shape = (num_filter)
    b = mx.random.uniform(0, 1, b_shape, dtype=dtype)
    kwargs = {}
    if kernel:
        kwargs['kernel'] = kernel
    if stride:
        kwargs['stride'] = stride
    if pad:
        kwargs['pad'] = pad
    if dilate:
        kwargs['dilate'] = dilate
    M = def_model('Convolution', num_filter=num_filter, num_group=num_group,  no_bias=no_bias,
                  layout='NCDHW', **kwargs)
    inputs = [x, w] if no_bias else [x, w, b]
    op_export_test('convolution', M, inputs, tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('num_outputs', [1, 3, 9])
@pytest.mark.parametrize('axis', [1, 2, -1, -2])
@pytest.mark.parametrize('squeeze_axis', [True, False, 0, 1])
def test_onnx_export_slice_channel(tmp_path, dtype, num_outputs, axis, squeeze_axis):
    shape = (3, 9, 18)
    if squeeze_axis and shape[axis] != num_outputs:
        return
    M = def_model('SliceChannel', num_outputs=num_outputs, axis=axis, squeeze_axis=squeeze_axis)
    x = mx.random.uniform(0, 1, shape, dtype=dtype)
    op_export_test('slice_channel', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('momentum', [0.9, 0.5, 0.1])
def test_onnx_export_batchnorm(tmp_path, dtype, momentum):
    x = mx.nd.random.normal(0, 10, (2, 3, 4, 5)).astype(dtype)
    gamma = mx.nd.random.normal(0, 10, (3)).astype(dtype)
    beta = mx.nd.random.normal(0, 10, (3)).astype(dtype)
    moving_mean = mx.nd.random.normal(0, 10, (3)).astype(dtype)
    moving_var = mx.nd.abs(mx.nd.random.normal(0, 10, (3))).astype(dtype)
    M = def_model('BatchNorm', eps=1e-5, momentum=momentum, fix_gamma=False, use_global_stats=False)
    op_export_test('BatchNorm1', M, [x, gamma, beta, moving_mean, moving_var], tmp_path)


# onnxruntime does not seem to support float64 and int32
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
@pytest.mark.parametrize('axis', [0, 2, -1, -2, -3])
@pytest.mark.parametrize('is_ascend', [True, False, 0, 1, None])
@pytest.mark.parametrize('dtype_i', ['float32', 'int32', 'int64'])
def test_onnx_export_argsort(tmp_path, dtype, axis, is_ascend, dtype_i):
    A = mx.random.uniform(0, 100, (4, 5, 6)).astype(dtype)
    kwargs = {}
    if is_ascend is not None:
        kwargs['is_ascend'] = is_ascend
    M = def_model('argsort', axis=axis, dtype=dtype_i, **kwargs)
    op_export_test('argsort', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('reps', [(2, 3), (2, ), (2, 3, 4)])
def test_onnx_export_tile(tmp_path, dtype, reps):
    x = mx.nd.random.normal(0, 100, (5, 6)).astype(dtype)
    M = def_model('tile', reps=reps)
    op_export_test('tile', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('axis', [-3, -2, -1, 0, 1, 2])
@pytest.mark.parametrize('mode', ['clip', 'wrap'])
def test_onnx_export_take(tmp_path, dtype, axis, mode):
    x = mx.nd.random.normal(0, 10, (3, 4, 5)).astype(dtype)
    y = mx.random.randint(-100, 100, (6, 7)).astype(dtype)
    M1 = def_model('take')
    op_export_test('take1', M1, [x, y], tmp_path)
    M2 = def_model('take', axis=axis, mode=mode)
    op_export_test('take2', M2, [x, y], tmp_path)


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
def test_onnx_export_one_hot(tmp_path, dtype, depth, shape):
    M = def_model('one_hot', depth=depth, dtype=dtype)
    x = mx.random.randint(0, 10, shape).astype('int64')
    op_export_test('one_hot', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('params', [((6, 5, 4), [1, 2, 4, 5, 6]),
                                     ((7, 3, 5), [1, 7, 4]),
                                     ((3, 2, 1), [1, 2])])
def test_onnx_export_sequence_reverse(tmp_path, dtype, params):
    x = mx.nd.random.uniform(0, 10, params[0]).astype(dtype)
    M1 = def_model('SequenceReverse')
    op_export_test('SequenceReverse1', M1, [x], tmp_path)
    seq_len = mx.nd.array(params[1])
    M1 = def_model('SequenceReverse', use_sequence_length=True)
    op_export_test('SequenceReverse1', M1, [x, seq_len], tmp_path)


# onnx LSTM from opset 11 does not support float64
@pytest.mark.parametrize('mode', ['lstm', 'gru'])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('state_size', [16, 32])
@pytest.mark.parametrize('input_size', [16, 32, 64])
@pytest.mark.parametrize('num_layers', [1, 2])
@pytest.mark.parametrize('batch_size', [1, 2, 4])
@pytest.mark.parametrize('seq_length', [16, 32])
def test_onnx_export_RNN(tmp_path, mode, dtype, state_size, input_size, num_layers, batch_size, seq_length):
    # TODO: The current implementation fails assertion checks for large parm/state_size. 

    # for num_layers >= 2, input_size must equal to state_size
    if num_layers >= 2 and input_size != state_size:
        return
    factor = 3
    if mode == 'lstm':
        factor = 4

    M = def_model('RNN', mode=mode, state_size=state_size, state_outputs=True,  num_layers=num_layers, p=0)
    x = mx.nd.random.normal(0, 10, (seq_length, batch_size, input_size), dtype=dtype)
    param = mx.nd.random.normal(0, 1, [num_layers*factor*state_size*input_size +
                                       num_layers*factor*state_size*state_size +
                                       num_layers*2*factor*state_size], dtype=dtype)
    state = mx.nd.random.uniform(-1, 1, [num_layers, batch_size, state_size], dtype=dtype)
    if mode == 'lstm':
        cell = mx.nd.random.uniform(-1, 1, [num_layers, batch_size, state_size], dtype=dtype)
        op_export_test('rnn', M, [x, param, state, cell], tmp_path)
    else:
        op_export_test('rnn', M, [x, param, state], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32', 'int64'])
@pytest.mark.parametrize('shapes', [((3, 3, 3), (1, 3)), ((4, 5, 6, 7), (6, 7))])
def test_onnx_export_broadcast_lesser_equal(tmp_path, dtype, shapes):
    A = mx.nd.random.uniform(0, 5, shapes[0]).astype('int32').astype(dtype)
    B = mx.nd.random.uniform(0, 5, shapes[1]).astype('int32').astype(dtype)
    M = def_model('broadcast_lesser_equal')
    op_export_test('broadcast_lesser_equal', M, [A, B], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32', 'int64'])
@pytest.mark.parametrize('shapes', [((3, 3, 3), (1, 3)), ((4, 5, 6, 7), (6, 7))])
def test_onnx_export_broadcast_greater_equal(tmp_path, dtype, shapes):
    A = mx.nd.random.uniform(0, 5, shapes[0]).astype('int32').astype(dtype)
    B = mx.nd.random.uniform(0, 5, shapes[1]).astype('int32').astype(dtype)
    M = def_model('broadcast_greater_equal')
    op_export_test('broadcast_greater_equal', M, [A, B], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (6, 7), (8,)])
def test_onnx_export_contrib_div_sqrt_dim(tmp_path, dtype, shape):
    A = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model('contrib.div_sqrt_dim')
    op_export_test('contrib_div_sqrt_dim', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
@pytest.mark.parametrize('shape', [(100,), (3, 4, 5), (6, 7)])
def test_onnx_export_reciprocal(tmp_path, dtype, shape):
    A = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model('reciprocal')
    op_export_test('reciprocal', M, [A], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize('shape', [(1, 3), (3, 4, 5)])
def test_onnx_export_power(tmp_path, shape, dtype):
    x = mx.nd.random.uniform(-5, 5, shape).astype(dtype)
    y = mx.nd.random.uniform(-10, 10, shape).astype(dtype)
    M = def_model('_internal._power')
    op_export_test('_internal._power', M, [x, y], tmp_path)

@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize('shape', [(1, 3), (3, 4, 5)])
def test_onnx_export_broadcast_power(tmp_path, shape, dtype):
    x = mx.nd.random.uniform(-5, 5, shape).astype(dtype)
    y = mx.nd.random.uniform(-10, 10, shape).astype(dtype)
    M = def_model('broadcast_power')
    op_export_test('broadcast_power', M, [x, y], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize('shape', [(3, 4, 5), (6, 7), (8,)])
def test_onnx_export_sqrt(tmp_path, dtype, shape):
    A = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model('sqrt')
    op_export_test('sqrt', M, [A], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("params", [[(1,4,2,3), 1], [(1,4,2,3), 2]])
def test_onnx_export_depth_to_space(tmp_path, dtype, params):
    shape, block_size = params
    M = def_model('depth_to_space', block_size=block_size)
    x = mx.nd.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    op_export_test('depth_to_space', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("params", [[(1,4,2,3), 1], [(1,1,4,6),2]])
def test_onnx_export_space_to_depth(tmp_path, dtype, params):
    shape, block_size = params
    M = def_model('space_to_depth', block_size=block_size)
    x = mx.nd.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    op_export_test('space_to_depth', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_square(tmp_path, dtype, shape):
    M = def_model('square')
    x = mx.nd.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    op_export_test('square', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_shape_array(tmp_path, dtype, shape):
    M = def_model('shape_array')
    x = mx.nd.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    op_export_test('shape_array', M, [x], tmp_path)


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


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_broadcast_lesser(tmp_path, dtype, shape):
    M = def_model('broadcast_lesser')
    x = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    y = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    op_export_test('broadcast_lesser', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (1,2,3), (4,5,6)])
def test_onnx_export_broadcast_greater(tmp_path, dtype, shape):
    M = def_model('broadcast_greater')
    x = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    y = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    op_export_test('broadcast_greater', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize("shape", [(10,5), (1,2,3), (4,5,6)])
@pytest.mark.parametrize('axis', [None, 1])
def test_onnx_export_log_softmax(tmp_path, dtype, shape, axis):
    x = mx.nd.random.uniform(0, 1, shape, dtype=dtype)
    kwargs = {}
    if axis is not None:
        kwargs['axis'] = axis
    M = def_model('log_softmax', **kwargs)
    op_export_test('log_softmax', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (2,3), (4,5,6)])
def test_onnx_export_broadcast_logical_and(tmp_path, dtype, shape):
    M = def_model('broadcast_logical_and')
    x = mx.nd.random.uniform(-1, 1, shape).astype(dtype)
    y = mx.nd.random.uniform(-1, 1, shape).astype(dtype)
    op_export_test('broadcast_logical_and', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (2,3), (4,5,6)])
def test_onnx_export_broadcast_logical_or(tmp_path, dtype, shape):
    M = def_model('broadcast_logical_or')
    x = mx.nd.random.uniform(-1, 1, shape).astype(dtype)
    y = mx.nd.random.uniform(-1, 1, shape).astype(dtype)
    op_export_test('broadcast_logical_or', M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize("shape", [(10,), (2,3), (4,5,6)])
def test_onnx_export_broadcast_logical_xor(tmp_path, dtype, shape):
    M = def_model('broadcast_logical_xor')
    x = mx.nd.random.uniform(-1, 1, shape).astype(dtype)
    y = mx.nd.random.uniform(-1, 1, shape).astype(dtype)
    op_export_test('broadcast_logical_xor', M, [x, y], tmp_path)


# onnxruntime currently does not support int32
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int64'])
@pytest.mark.parametrize('shape', [(1,), (2, 3), (4, 5, 6)])
def test_onnx_export_clip(tmp_path, dtype, shape):
    A = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    a_min = mx.nd.min(A).astype('float32').asnumpy()[0] + 5
    a_max = mx.nd.max(A).astype('float32').asnumpy()[0] - 5
    print(a_min)
    M = def_model('clip', a_min=a_min, a_max=a_max)
    op_export_test('clip', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(3, 4, 5), (6, 7), (8,)])
@pytest.mark.parametrize('func', [lambda x : x + np.random.rand(1)[0]*100,
                                  lambda x : x * np.random.rand(1)[0]*100,
                                  lambda x : x - np.random.rand(1)[0]*100,
                                  lambda x : np.random.rand(1)[0]*100 - x,
                                  lambda x : x / (np.random.rand(1)[0]*100),
                                  lambda x : np.random.rand(1)[0]*100 / x,
                                  lambda x : x ** np.random.rand(1)[0]*10,
                                 ])
def test_onnx_export_scalar_op(tmp_path, dtype, shape, func):
    A = mx.nd.random.uniform(1, 100, shape).astype(dtype)
    M = def_model_from_func(func)
    op_export_test('_scalar', M, [A], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32'])
@pytest.mark.parametrize('shape', [(1, 1, 1), (2, 3, 4), (5, 6, 7, 8)])
@pytest.mark.parametrize('axis', ['None', 0, 1, 2, -1, -2])
@pytest.mark.parametrize('keepdims', [True, False])
@pytest.mark.parametrize('op_name', ['argmax', 'argmin'])
def test_onnx_export_arg_max_min(tmp_path, dtype, shape, axis, keepdims, op_name):
    A = mx.nd.random.uniform(-100, 100, shape).astype(dtype)
    M = def_model(op_name, axis=axis, keepdims=keepdims)
    op_export_test(op_name, M, [A], tmp_path)


# onnx max and min have issue comparing negative float16 values
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [[(2, 3), (2, 3)], [(5, 4), (5, 4)]])
@pytest.mark.parametrize('op_name', ['maximum', 'minimum'])
def test_onnx_export_maximum_minimum(tmp_path, dtype, shape, op_name):
    lhs = mx.nd.random.uniform(1, 100, shape[0]).astype(dtype)
    rhs = mx.nd.random.uniform(1, 100, shape[1]).astype(dtype)
    M = def_model(op_name)
    op_export_test(op_name, M, [lhs, rhs], tmp_path)



# onnx reduce ops do not support float64
@pytest.mark.parametrize('dtype', ['float16', 'float32','int32', 'int64'])
@pytest.mark.parametrize('shape', [(2, 3), (4, 5, 6)])
@pytest.mark.parametrize('axis', [None, 0, 1, -1, (0, 1)])
@pytest.mark.parametrize('keepdims', [True, False])
@pytest.mark.parametrize('op_name', ['max', 'min', 'mean', 'prod'])
def test_onnx_export_reduce_op(tmp_path, dtype, shape, axis, keepdims, op_name):
    if dtype != 'int64' or op_name != 'mean':
        # onnx ReduceMean does not support int 64
        x = mx.nd.random.uniform(1, 100, shape=shape).astype(dtype)
        M = def_model(op_name, axis=axis, keepdims=keepdims)
        op_export_test(op_name, M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1,), (3, ), (4, 5), (3, 4, 5)])
@pytest.mark.parametrize('op_name', ['elemwise_add', 'elemwise_sub', 'elemwise_mul', 'elemwise_div'])
def test_onnx_export_elemwise_op(tmp_path, dtype, shape, op_name):
    x = mx.nd.random.uniform(1, 100, shape=shape).astype(dtype)
    y = mx.nd.random.uniform(1, 100, shape=shape).astype(dtype)
    M = def_model(op_name)
    op_export_test(op_name, M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [[(3, 4), (3, 4)], [(3, 4), (3, 1)], [(3, 4), (4)]])
@pytest.mark.parametrize('op_name', ['broadcast_sub', 'broadcast_div'])
def test_onnx_export_broadcast_op(tmp_path, dtype, shape, op_name):
    x = mx.nd.random.uniform(1, 100, shape=shape[0]).astype(dtype)
    y = mx.nd.random.uniform(1, 100, shape=shape[1]).astype(dtype)
    M = def_model(op_name)
    op_export_test(op_name, M, [x, y], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape', [(1,), (3, ), (4, 5), (3, 4, 5)])
def test_onnx_export_negative(tmp_path, dtype, shape):
    x = mx.nd.random.uniform(-100, 100, shape=shape).astype(dtype)
    M = def_model('negative')
    op_export_test('negative', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('shape', [(1,), (3, ), (4, 5), (3, 4, 5)])
def test_onnx_export_addn(tmp_path, dtype, shape):
    x = mx.nd.random.uniform(-100, 100, shape=shape).astype(dtype)
    M = def_model('add_n')
    op_export_test('add_n', M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('shape', [(1,), (3, ), (4, 5), (3, 4, 5)])
@pytest.mark.parametrize('op_name', ['ceil', 'floor', 'log'])
def test_onnx_export_ufunc(tmp_path, dtype, shape, op_name):
    x = mx.nd.random.uniform(-100, 100, shape=shape).astype(dtype)
    M = def_model(op_name)
    op_export_test(op_name, M, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('shape_axis', [[(1, 1), None], [(3, 1, 2, 1), (None)], [(3, 1, 2, 1), (1)], 
                            [(3, 1, 2, 1), (1, 3)]])
def test_onnx_export_squeeze(tmp_path, dtype, shape_axis):
    x = mx.nd.random.uniform(1, 100, shape=shape_axis[0]).astype(dtype)
    M = def_model('squeeze', axis=shape_axis[1])
    op_export_test('squeeze', M, [x], tmp_path)
