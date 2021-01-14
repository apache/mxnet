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

def op_export_test(model_name, Model, inputs, tmp_path, dummy_input=False):
    def export_to_onnx(model, model_name, inputs):
        model_path = '{}/{}'.format(tmp_path, model_name)
        model.export(model_path, epoch=0)
        sym_file = '{}-symbol.json'.format(model_path)
        params_file = '{}-0000.params'.format(model_path)
        dtype = inputs[0].dtype
        onnx_file = '{}/{}.onnx'.format(tmp_path, model_name)
        mx.contrib.onnx.export_model(sym_file, params_file, [inp.shape for inp in inputs],
                                     dtype, onnx_file)
        return onnx_file

    def onnx_rt(onnx_file, inputs):
        sess = rt.InferenceSession(onnx_file)
        dtype_0 = inputs[0].asnumpy().dtype
        input_dict = dict((sess.get_inputs()[i].name, inputs[i].asnumpy().astype(dtype_0)) for i in range(len(inputs)))
        pred = sess.run(None, input_dict)[0]
        return pred

    # create a new model 
    model = Model()
    model.initialize(ctx=mx.cpu(0))
    model.hybridize()
    pred_nat = model(*inputs)
    onnx_file = export_to_onnx(model, model_name, inputs)
    pred_onx = onnx_rt(onnx_file, inputs)
    if dummy_input:
        pred_nat = pred_nat[0]
    assert_almost_equal(pred_nat, pred_onx)


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


def test_onnx_export_zeros_like(tmp_path):
    M = def_model('zeros_like')
    x = mx.nd.array([[-2,-1,0],[0,50,99],[4,5,6],[7,8,9]], dtype='float32')
    op_export_test('zeros_like', M, [x], tmp_path)

def test_onnx_export_ones_like(tmp_path):
    M = def_model('ones_like')
    x = mx.nd.array([[-2,-1,0],[0,50,99],[4,5,6],[7,8,9]], dtype='float32')
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


@pytest.mark.parametrize("stop", [2, 50, 5000])
@pytest.mark.parametrize("step", [0.25, 0.5, 1, 5])
@pytest.mark.parametrize("start", [0., 1.])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_onnx_export_arange(tmp_path, dtype, start, stop, step):
    if "int" in dtype:
        start = int(start)
        stop = int(stop)
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
    op_export_test('slice_axis_1', M1, [x], tmp_path)
    op_export_test('slice_axis_2', M2, [x], tmp_path)
    op_export_test('slice_axis_3', M3, [x], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_onnx_export_reshape(tmp_path, dtype):
    x = mx.nd.ones((2, 3, 4, 5, 6), dtype=dtype)
    M1 = def_model('reshape', shape=(6, 1, 0, -1))
    op_export_test('reshape_1', M1, [x], tmp_path)
    M2 = def_model('reshape', shape=(3, -1, 0, 0), reverse=True)
    op_export_test('reshape_2', M2, [x], tmp_path)
    M3 = def_model('reshape', shape=(5, 1, 1, 1, 1, 0 -1, 0), reverse=True)
    op_export_test('reshape_3', M3, [x], tmp_path)


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


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'float16'])
@pytest.mark.parametrize('shape', [(1,), (3,), (4, 5), (3, 4, 5)])
def test_onnx_export_elemwise_add(tmp_path, dtype, shape):
    M = def_model('elemwise_add')
    x = mx.nd.random.uniform(-0.5, 0.5, shape=shape, dtype=dtype)
    y = mx.nd.random.uniform(-0.5, 0.5, shape=shape, dtype=dtype)
    op_export_test('elmwise_add', M, [x, y], tmp_path)


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
@pytest.mark.parametrize('temperature', [.1, 1., 10.])
def test_onnx_export_softmax(tmp_path, dtype, temperature):
    x = mx.nd.random.uniform(0, 1, (2, 3, 4), dtype=dtype)
    M1 = def_model('softmax', temperature=temperature)
    op_export_test('softmax_1', M1, [x], tmp_path)
    M2 = def_model('softmax', use_length=True, axis=0, temperature=temperature)
    l2 = mx.nd.array([[2,0,2,1],[1,1,2,1], [0,0,0,1]], dtype=int)
    op_export_test('softmax_2', M2, [x, l2], tmp_path)
    M3 = def_model('softmax', use_length=True, axis=-1, temperature=temperature)
    l3 = mx.nd.array([[2,0,4],[0,0,0]], dtype=int)
    op_export_test('softmax_3', M3, [x, l3], tmp_path)
    M4 = def_model('softmax', use_length=True, axis=1, temperature=temperature)
    l4 = mx.nd.array([[2,0,3,1],[0,1,0,0]], dtype=int)
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
@pytest.mark.parametrize("shape", [(1,1), (3,3), (10,2), (20,30,40)])
def test_onnx_export_where(tmp_path, dtype, shape):
    M = def_model('where')
    x = mx.nd.zeros(shape, dtype=dtype)
    y = mx.nd.ones(shape, dtype=dtype)
    cond = mx.nd.random.randint(low=0, high=1, shape=shape, dtype='int32')
    op_export_test('where', M, [cond, x, y], tmp_path)

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
