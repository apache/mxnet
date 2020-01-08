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
import numpy as np
from itertools import product
import copy

from numpy.testing import assert_allclose

import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed

def check_unsupported_single_sym(sym):
    wrapped_sym = mx.sym.Group([mx.sym.identity(s) for s in sym])
    trt_sym = wrapped_sym.get_backend_symbol('TensorRT')
    assert len(wrapped_sym.get_internals()) == len(trt_sym.get_internals())

def check_single_sym(sym, data_shapes, arg_params_shapes=None, aux_params_shapes=None,
                     rtol_fp32=1e-5, atol_fp32=0., rtol_fp16=1e-3, atol_fp16=0.):
    if arg_params_shapes is None:
        arg_params_shapes = {}
    if aux_params_shapes is None:
        aux_params_shapes = {}
    for i in range(3):
        data = {k: mx.nd.array(np.random.rand(*v) + 0.01, dtype='float32', ctx=mx.cpu())
                for k, v in data_shapes.items()}
        arg_params = {k: mx.nd.array(np.random.rand(*v) + 0.01, dtype='float32', ctx=mx.cpu())
                      for k, v in arg_params_shapes.items()}
        aux_params = {k: mx.nd.array(np.random.rand(*v) + 0.01, dtype='float32', ctx=mx.cpu())
                      for k, v in aux_params_shapes.items()}
        wrapped_sym = mx.sym.Group([mx.sym.identity(s) for s in sym])

        # Test FP32 MXNet Native
        shapes = {}
        shapes.update(data_shapes)
        shapes.update(arg_params_shapes)
        shapes.update(aux_params_shapes)
        orig_executor = wrapped_sym.simple_bind(ctx=mx.gpu(0), grad_req='null',
                                                force_rebind=True, **shapes)
        orig_executor.copy_params_from(arg_params, aux_params)
        orig_executor.forward(is_train=False, **data)
        orig_outputs = [arr.asnumpy() for arr in orig_executor.outputs]

        # Test FP32 MXNet-TRT
        mx.contrib.tensorrt.set_use_fp16(False)
        trt_sym = wrapped_sym.get_backend_symbol('TensorRT')
        assert len(trt_sym.get_internals()) < len(wrapped_sym.get_internals())
        remaining_arg_params, remaining_aux_params = \
            mx.contrib.tensorrt.init_tensorrt_params(trt_sym, arg_params, aux_params)
        shapes = {}
        shapes.update(data_shapes)
        shapes.update({k: v.shape for k, v in remaining_arg_params.items()})
        shapes.update({k: v.shape for k, v in remaining_aux_params.items()})
        trt_fp32_executor = trt_sym.simple_bind(ctx=mx.gpu(0), grad_req='null',
                                                force_rebind=True, **shapes)
        trt_fp32_executor.copy_params_from(remaining_arg_params, remaining_aux_params)
        trt_fp32_executor.forward(is_train=False, **data)
        trt_fp32_outputs = [arr.asnumpy() for arr in trt_fp32_executor.outputs]

        # Test FP16 MXNet-TRT
        mx.contrib.tensorrt.set_use_fp16(True)
        data = {k: v.astype('float16') for k, v in data.items()}
        arg_params = {k: v.astype('float16') for k, v in arg_params.items()}
        aux_params = {k: v.astype('float16') for k, v in aux_params.items()}
        trt_sym = wrapped_sym.get_backend_symbol('TensorRT')
        assert len(trt_sym.get_internals()) < len(wrapped_sym.get_internals())
        remaining_arg_params, remaining_aux_params = \
            mx.contrib.tensorrt.init_tensorrt_params(trt_sym, arg_params, aux_params)
        shapes = {}
        shapes.update(data_shapes)
        shapes.update({k: v.shape for k, v in remaining_arg_params.items()})
        shapes.update({k: v.shape for k, v in remaining_aux_params.items()})

        trt_fp16_executor = trt_sym.simple_bind(ctx=mx.gpu(0),
                                                type_dict={k: 'float16' for k in shapes.keys()},
                                                grad_req='null', force_rebind=True, **shapes)
        trt_fp16_executor.copy_params_from(remaining_arg_params, remaining_aux_params)
        trt_fp16_executor.forward(is_train=False, **data)
        trt_fp16_outputs = [arr.asnumpy() for arr in trt_fp16_executor.outputs]

        for j, (orig, fp16, fp32) in enumerate(zip(orig_outputs, trt_fp16_outputs, trt_fp32_outputs)):
            abs_orig = abs(orig)
            diff32 = abs(fp32 - orig)
            diff16 = abs(fp16.astype('float32') - orig)
            _atol32 = diff32 - rtol_fp32 * abs_orig
            _atol16 = diff16 - rtol_fp16 * abs_orig
            print("{}: diff32({:.2E}) | diff16({:.2E}) | atol32({:.2E}) | atol16({:.2E}) | orig.min({:.2E})".format(
                  j, diff32.max(), diff16.max(), _atol32.max(), _atol16.max(), abs_orig.min()))
            assert_allclose(fp32, orig, rtol=rtol_fp32, atol=atol_fp32)
            assert_allclose(fp16, orig, rtol=rtol_fp16, atol=atol_fp16)

@with_seed()
def test_noop():
    data = mx.sym.Variable('data')
    check_unsupported_single_sym(data)


@with_seed()
def test_identity():
    data = mx.sym.Variable('data')
    sym = mx.sym.identity(data)
    check_single_sym(sym, data_shapes={'data': (8,3,32,32)},
                     rtol_fp32=0., atol_fp32=0., rtol_fp16=1e-3, atol_fp16=1e-7)


@with_seed()
def test_convolution2d():
    data = mx.sym.Variable('data')
    weight = mx.sym.Variable('weight')
    bias = mx.sym.Variable('bias')
    data_shape = (8,3,16,16)
    num_filter = 7
    for kernel in [(3, 3), (1, 1), (3, 1)]:
        for stride in [(1, 1), (2, 2), (2, 1)]:
            if stride[0] > kernel[0] or stride[1] > kernel[1]: # doesn't make any sense
                continue
            if kernel == (3, 3) and stride == (1, 1):
                atol_fp32 = 0.
                rtol_fp32 = 1e-5
                atol_fp16 = 0.
                rtol_fp16 = 1e-2
            else:
                atol_fp32 = 0.
                rtol_fp32 = 0.
                atol_fp16 = 0.
                rtol_fp16 = 1e-2
            for pad in [(1, 1), (0, 0), (1, 0)]:
                for group in [1, 2]:
                    for layout in ['NCHW', 'NHWC']:
                        weight_shape = (num_filter, data_shape[1]) + kernel
                        bias_shape = (num_filter,)
                        sym = mx.sym.Convolution(data, weight=weight, bias=bias, kernel=kernel,
                                                 stride=stride, pad=pad, num_filter=num_filter,
                                                 no_bias=False, layout=layout)
                        if layout == 'NCHW':
                            print("kernel: {} | stride: {} | pad: {} | group: {} | layout: {} | with_bias".format(
                                  kernel, stride, pad, group, layout))
                            check_single_sym(sym, {'data': data_shape},
                                             {'weight': weight_shape, 'bias': bias_shape},
                                             rtol_fp32=rtol_fp32, atol_fp32=atol_fp32,
                                             rtol_fp16=rtol_fp16, atol_fp16=atol_fp16)
                        else:
                            check_unsupported_single_sym(sym)
                        sym = mx.sym.Convolution(data, weight=weight, kernel=kernel, stride=stride,
                                                 pad=pad, num_filter=num_filter, no_bias=True,
                                                 layout=layout)
                        if layout == 'NCHW':
                            print("kernel: {} | stride: {} | pad: {} | group: {} | layout: {} | without_bias".format(
                                  kernel, stride, pad, group, layout))
                            check_single_sym(sym, {'data': data_shape},
                                             {'weight': weight_shape},
                                             rtol_fp32=rtol_fp32, atol_fp32=atol_fp32,
                                             rtol_fp16=rtol_fp16, atol_fp16=atol_fp16)
                        else:
                            check_unsupported_single_sym(sym)

@with_seed()
def test_deconvolution2d():
    data = mx.sym.Variable('data')
    weight = mx.sym.Variable('weight')
    bias = mx.sym.Variable('bias')
    data_shape = (8,3,16,16)
    num_filter = 7
    for kernel in [(3, 3), (1, 1), (3, 1)]:
        for stride in [(1, 1), (2, 2), (2, 1)]:
            if stride[0] > kernel[0] or stride[1] > kernel[1]: # doesn't make any sense
                continue
            if kernel == (3, 3) and stride == (1, 1):
                atol_fp32 = 0.
                rtol_fp32 = 5e-5
                atol_fp16 = 0.
                rtol_fp16 = 1e-2
            else:
                atol_fp32 = 0.
                rtol_fp32 = 1e-6
                atol_fp16 = 0.
                rtol_fp16 = 1e-2
            for pad in [(1, 1), (0, 0), (1, 0)]:
                for group in [1, 2]:
                    for layout in ['NCHW', 'NHWC']:
                        weight_shape = (data_shape[1], num_filter) + kernel
                        bias_shape = (num_filter,)
                        sym = mx.sym.Deconvolution(data, weight=weight, bias=bias, kernel=kernel,
                                                 stride=stride, pad=pad, num_filter=num_filter,
                                                 no_bias=False, layout=layout)
                        if layout == 'NCHW':
                            print("kernel: {} | stride: {} | pad: {} | group: {} | layout: {} | with_bias".format(
                                  kernel, stride, pad, group, layout))
                            check_single_sym(sym, {'data': data_shape},
                                             {'weight': weight_shape, 'bias': bias_shape},
                                             rtol_fp32=rtol_fp32, atol_fp32=atol_fp32,
                                             rtol_fp16=rtol_fp16, atol_fp16=atol_fp16)
                        else:
                            check_unsupported_single_sym(sym)
                        sym = mx.sym.Deconvolution(data, weight=weight, kernel=kernel, stride=stride,
                                                 pad=pad, num_filter=num_filter, no_bias=True,
                                                 layout=layout)
                        if layout == 'NCHW':
                            print("kernel: {} | stride: {} | pad: {} | group: {} | layout: {} | without_bias".format(
                                  kernel, stride, pad, group, layout))
                            check_single_sym(sym, {'data': data_shape},
                                             {'weight': weight_shape},
                                             rtol_fp32=rtol_fp32, atol_fp32=atol_fp32,
                                             rtol_fp16=rtol_fp16, atol_fp16=atol_fp16)
                        else:
                            check_unsupported_single_sym(sym)

@with_seed()
def test_fully_connected(): # TODO(cfujitsang): take care of flatten option
    data = mx.sym.Variable('data')
    weight = mx.sym.Variable('weight')
    bias = mx.sym.Variable('bias')
    data_shape = (8,64)
    num_hidden = 7
    weight_shape = (num_hidden, data_shape[1])
    bias_shape = (num_hidden,)
    sym = mx.sym.FullyConnected(data, weight=weight, bias=bias, no_bias=False,
                                num_hidden=num_hidden)
    check_single_sym(sym, {'data': data_shape}, {'weight': weight_shape, 'bias': bias_shape},
                     rtol_fp16=5e-3, atol_fp16=0.)
    sym = mx.sym.FullyConnected(data, weight=weight, no_bias=True, num_hidden=num_hidden)
    check_unsupported_single_sym(sym)


@with_seed()
def test_relu():
    data = mx.sym.Variable('data')
    sym = mx.sym.relu(data)
    for data_shape in [(10, 32), (10, 3, 32), (10, 3, 32, 32), (10, 3, 7, 32, 32)]:
        check_single_sym(sym, {'data': data_shape}, rtol_fp32=0., atol_fp32=0.,
                         rtol_fp16=1e-3, atol_fp16=1e-7)


@with_seed()
def test_activation():
    data = mx.sym.Variable('data')
    for act_type in ['relu', 'sigmoid', 'tanh']:
        sym = mx.sym.Activation(data, act_type=act_type)
        for data_shape in [(10, 32), (10, 3, 32), (10, 3, 32, 32), (10,3,7,32,32)]:
            check_single_sym(sym, {'data': data_shape}, rtol_fp32=0., atol_fp32=0.,
                             rtol_fp16=1e-3, atol_fp16=1e-7)
    for act_type in ['softrelu', 'softsign']:
        sym = mx.sym.Activation(data, act_type=act_type)
        check_unsupported_single_sym(sym)


@with_seed()
def test_pooling2d():
    data = mx.sym.Variable('data')
    data_shape = (4, 3, 32,32)
    for pool_type in ['max', 'avg', 'lp', 'sum']:
        if pool_type == 'max':
            rtol_fp32 = 1e-6
            atol_fp32 = 0.
            rtol_fp16 = 1e-3
            atol_fp16 = 0.
        else:
            rtol_fp32 = 5e-6
            atol_fp32 = 0.
            rtol_fp16 = 1e-3
            atol_fp16 = 0.
        for layout in ['NHWC', 'NCHW']:
            for (stride, pad, kernel, count_include_pad, pooling_convention) \
                 in product([(2,2), (2,1)], [(0,0), (1,1)], [(2,2), (3,2)],
                            [True, False], ['valid', 'full']):
                print("pool_type: {} | layout: {} | stride: {} | pad: {} | ".format(
                      pool_type, layout, stride, pad) +
                      "kernel: {} | count_include_pad: {} | pooling_convention: {}".format(
                      kernel, count_include_pad, pooling_convention))
                sym = mx.sym.Pooling(data, kernel=kernel, pool_type=pool_type, stride=stride,
                                     pad=pad, layout=layout, count_include_pad=count_include_pad,
                                     pooling_convention=pooling_convention)
                if (layout == 'NHWC') or \
                    pool_type not in ('max', 'avg') or \
                    pooling_convention != 'valid' or \
                    (pool_type == 'avg' and count_include_pad):
                    check_unsupported_single_sym(sym)
                else:
                    check_single_sym(sym, {'data': data_shape},
                                     rtol_fp32=rtol_fp32, atol_fp32=atol_fp32,
                                     rtol_fp16=rtol_fp16, atol_fp16=atol_fp16)
            print("pool_type: {} | layout: {} | global_pool".format(pool_type, layout))
            sym = mx.sym.Pooling(data, global_pool=True, pool_type=pool_type, layout=layout)
            if layout == 'NHWC' or pool_type not in ('max', 'avg'):
                check_unsupported_single_sym(sym)
            else:
                if pool_type == 'max':
                    rtol_fp32 = 0.
                    atol_fp32 = 0.
                    rtol_fp16 = 1e-3
                    atol_fp16 = 0.
                else:
                    rtol_fp32 = 1e-5
                    atol_fp32 = 0.
                    rtol_fp16 = 1e-3
                    atol_fp16 = 0.
                check_single_sym(sym, {'data': data_shape}, rtol_fp32=rtol_fp32,
                                 atol_fp32=atol_fp32, rtol_fp16=rtol_fp16, atol_fp16=atol_fp16)


@with_seed()
def test_softmax_output():
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    data_shape = (8, 100)
    label_shape = (8, 100)
    sym = mx.sym.SoftmaxOutput(data, label)
    check_single_sym(sym, {'data': data_shape, 'label': label_shape},
                     rtol_fp32=1e-6, atol_fp32=0., rtol_fp16=5e-3, atol_fp16=0.)
    sym = mx.sym.SoftmaxOutput(data)
    check_single_sym(sym, {'data': data_shape},
                     rtol_fp32=1e-6, atol_fp32=0., rtol_fp16=5e-3, atol_fp16=0.)



def check_batch_norm(sym, data_shapes, arg_params_shapes=None, aux_params_shapes=None,
                     rtol_fp32=1e-5, atol_fp32=1e-7, rtol_fp16=1e-2, atol_fp16=1e-3):
    if arg_params_shapes is None:
        arg_params_shapes = {}
    if aux_params_shapes is None:
        aux_params_shapes = {}
    for i in range(3):
        data = {
            'data': mx.nd.array(np.random.rand(*data_shapes['data']) + 0.01,
                                dtype='float32', ctx=mx.cpu())
        }
        arg_params = {
            'gamma': mx.nd.array(np.random.rand(*arg_params_shapes['gamma']) * 0.1 + 1.,
                                 dtype='float32', ctx=mx.cpu()),
            'beta': mx.nd.array(np.random.rand(*arg_params_shapes['beta']),
                                dtype='float32', ctx=mx.cpu())
        }
        aux_params = {
            'moving_mean': mx.nd.array(
                0.45 + np.random.rand(*aux_params_shapes['moving_mean']) * 0.1 + 0.01,
                                      dtype='float32', ctx=mx.cpu()),
            'moving_var': mx.nd.array(
                0.95 + np.random.rand(*aux_params_shapes['moving_var']) * 0.1,
                                      dtype='float32', ctx=mx.cpu())
        }
        wrapped_sym = mx.sym.Group([mx.sym.identity(s) for s in sym])

        # Test FP32 MXNet Native
        shapes = {}
        shapes.update(data_shapes)
        shapes.update(arg_params_shapes)
        shapes.update(aux_params_shapes)
        orig_executor = wrapped_sym.simple_bind(ctx=mx.gpu(0), grad_req='null',
                                                force_rebind=True, **shapes)
        orig_executor.copy_params_from(arg_params, aux_params)
        orig_executor.forward(is_train=False, **data)
        orig_outputs = [arr.asnumpy() for arr in orig_executor.outputs]

        # Test FP32 MXNet-TRT
        mx.contrib.tensorrt.set_use_fp16(False)
        trt_sym = wrapped_sym.get_backend_symbol('TensorRT')
        assert len(trt_sym.get_internals()) < len(wrapped_sym.get_internals())
        remaining_arg_params, remaining_aux_params = \
            mx.contrib.tensorrt.init_tensorrt_params(trt_sym, arg_params, aux_params)
        shapes = {}
        shapes.update(data_shapes)
        shapes.update({k: v.shape for k, v in remaining_arg_params.items()})
        shapes.update({k: v.shape for k, v in remaining_aux_params.items()})
        trt_fp32_executor = trt_sym.simple_bind(ctx=mx.gpu(0), grad_req='null',
                                                force_rebind=True, **shapes)
        trt_fp32_executor.copy_params_from(remaining_arg_params, remaining_aux_params)
        trt_fp32_executor.forward(is_train=False, **data)
        trt_fp32_outputs = [arr.asnumpy() for arr in trt_fp32_executor.outputs]

        # Test FP16 MXNet-TRT
        mx.contrib.tensorrt.set_use_fp16(True)
        data = {k: v.astype('float16') for k, v in data.items()}
        arg_params = {k: v.astype('float32') for k, v in arg_params.items()}
        aux_params = {k: v.astype('float32') for k, v in aux_params.items()}
        trt_sym = wrapped_sym.get_backend_symbol('TensorRT')
        remaining_arg_params, remaining_aux_params = \
            mx.contrib.tensorrt.init_tensorrt_params(trt_sym, arg_params, aux_params)
        shapes = {}
        shapes.update(data_shapes)
        shapes.update({k: v.shape for k, v in remaining_arg_params.items()})
        shapes.update({k: v.shape for k, v in remaining_aux_params.items()})

        trt_fp16_executor = trt_sym.simple_bind(ctx=mx.gpu(0),
                                                type_dict={k: 'float16' for k in shapes.keys()},
                                                grad_req='null', force_rebind=True, **shapes)
        trt_fp16_executor.copy_params_from(remaining_arg_params, remaining_aux_params)
        trt_fp16_executor.forward(is_train=False, **data)
        trt_fp16_outputs = [arr.asnumpy() for arr in trt_fp16_executor.outputs]


        for j, (orig, fp16, fp32) in enumerate(zip(orig_outputs,
                                                   trt_fp16_outputs,
                                                   trt_fp32_outputs)):
            abs_orig = abs(orig)
            diff32 = abs(fp32 - orig)
            diff16 = abs(fp16.astype('float32') - orig)
            _atol32 = diff32 - rtol_fp32 * abs_orig
            _atol16 = diff16 - rtol_fp16 * abs_orig
            print("{}: diff32({:.2E}) | diff16({:.2E}) | atol32({:.2E}) | atol16({:.2E}) | orig.min({:.2E})".format(
                  j, diff32.max(), diff16.max(), _atol32.max(), _atol16.max(), abs_orig.min()))
            assert_allclose(fp32, orig, rtol=rtol_fp32, atol=atol_fp32)
            assert_allclose(fp16.astype('float32'), orig, rtol=rtol_fp16, atol=atol_fp16)

@with_seed()
def test_batch_norm():
    data = mx.sym.Variable('data')
    gamma = mx.sym.Variable('gamma')
    beta = mx.sym.Variable('beta')
    moving_mean = mx.sym.Variable('moving_mean')
    moving_var = mx.sym.Variable('moving_var')
    data_shape = (4,3,32,32)
    gamma_shape = (3,)
    beta_shape = (3,)
    moving_mean_shape = (3,)
    moving_var_shape = (3,)
    for fix_gamma in [True, False]:
        for use_global_stats in [True, False]:
            for axis in [0, 1, 2, 3]:
                sym = mx.sym.BatchNorm(data, gamma=gamma, beta=beta, moving_mean=moving_mean,
                                       fix_gamma=fix_gamma, moving_var=moving_var, momentum=0.9,
                                       axis=axis, use_global_stats=use_global_stats, eps=1e-5)
                if axis == 1:
                    check_batch_norm(sym,
                        {'data': data_shape}, {'gamma': gamma_shape, 'beta': beta_shape},
                        {'moving_mean': moving_mean_shape, 'moving_var': moving_var_shape},
                        atol_fp32=2e-7)
                else:
                    check_unsupported_single_sym(sym)


@with_seed()
def test_clip():
    data = mx.sym.Variable('data')
    sym = mx.sym.clip(data, 0.25, 0.75)
    for data_shape in [(10, 32), (10, 3, 32), (10, 3, 32, 32), (10,3,7,32,32)]:
        check_single_sym(sym, {'data': data_shape},
                         rtol_fp32=0., atol_fp32=0.,
                         rtol_fp16=1e-3, atol_fp16=0.)


@with_seed()
def test_concat():
    lhs = mx.sym.Variable('lhs')
    rhs = mx.sym.Variable('rhs')
    shape = [3, 5, 7, 9]
    lhs_shape = tuple(shape)
    for axis in range(1, 4):
        sym = mx.sym.concat(lhs, rhs, dim=axis)
        rhs_shape = copy.copy(shape)
        rhs_shape[axis] = 1
        rhs_shape = tuple(rhs_shape)
        check_single_sym(sym, {'lhs': lhs_shape, 'rhs': rhs_shape},
                         rtol_fp32=0., atol_fp32=0., rtol_fp16=1e-3, atol_fp16=1e-7)


@with_seed()
def test_elemwise_ops():
    lhs = mx.sym.Variable('lhs')
    rhs = mx.sym.Variable('rhs')
    shape = (3, 5, 7, 9)
    lhs_shape = tuple(shape)
    sym = mx.sym.elemwise_add(lhs, rhs)
    check_single_sym(sym, {'lhs': shape, 'rhs': shape},
                     rtol_fp32=0., atol_fp32=0.)

    sym = mx.sym.elemwise_sub(lhs, rhs)
    # TODO(cfujitsang): is atol_fp16 ok ?
    check_single_sym(sym, {'lhs': shape, 'rhs': shape},
                     rtol_fp32=0., atol_fp32=0., rtol_fp16=1e-3, atol_fp16=1e-3)

    sym = mx.sym.elemwise_mul(lhs, rhs)
    check_single_sym(sym, {'lhs': shape, 'rhs': shape},
                     rtol_fp32=0., atol_fp32=0., rtol_fp16=5e-3, atol_fp16=1e-7)

@with_seed()
def test_flatten():
    data = mx.sym.Variable('data')
    sym = mx.sym.flatten(data)
    for data_shape in [(3, 5, 7), (3, 5, 7, 9), (3, 5, 7, 9, 11)]:
        check_single_sym(sym, {'data': data_shape},
                         rtol_fp32=0., atol_fp32=0., atol_fp16=1e-7)

@with_seed()
def test_dropout():
    data = mx.sym.Variable('data')
    for data_shape in [(3, 5), (3, 5, 7), (3, 5, 7, 9)]:
        for mode in ['training', 'always']:
            sym = mx.sym.Dropout(data, p=0.7, mode=mode)
            if mode == 'training':
                check_single_sym(sym, {'data': data_shape},
                                 rtol_fp32=0., atol_fp32=0., atol_fp16=1e-7)
            else:
                check_unsupported_single_sym(sym)
            sym = mx.sym.Dropout(data, p=0.7, mode=mode, axes=(0,))
            check_unsupported_single_sym(sym)

if __name__ == "__main__":
    import nose
    nose.runmodule()
