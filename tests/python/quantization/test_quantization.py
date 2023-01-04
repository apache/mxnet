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

"""Some of the tests using CUDNN require a special GPU instruction called dp4a.
Ref: http://images.nvidia.com/content/pdf/tesla/184457-Tesla-P4-Datasheet-NV-Final-Letter-Web.pdf
"""
import os
import mxnet as mx
import numpy as onp
from mxnet import npx
from mxnet.util import use_np
from mxnet.gluon.model_zoo import vision
from mxnet.test_utils import assert_almost_equal, assert_exception, rand_ndarray, rand_shape_nd, same, DummyIter
from common import xfail_when_nonstandard_decimal_separator
from mxnet.io import NDArrayIter
import unittest
import operator

npx.reset_np()

def collect_block_args_aux(block, sym):
  arg_params, aux_params = dict(), dict()
  for k, v in block.collect_params().items():
    if k in sym.list_arguments():
      arg_params[k]= v._reduce()
    elif k in sym.list_auxiliary_states():
      aux_params[k]= v._reduce()
  return arg_params, aux_params

def is_test_for_gpu():
    return mx.current_device().device_type == 'gpu'


def is_test_for_dnnl():
    return (mx.current_device().device_type == 'cpu'
            and os.environ.get('ENABLE_ONEDNN_QUANTIZATION_TEST') == '1')


def is_test_for_native_cpu():
    return (mx.current_device().device_type == 'cpu'
            and os.environ.get('ENABLE_ONEDNN_QUANTIZATION_TEST') == None)


def get_low_high(qtype):
    """ Return low and high value for given integer type as float number"""
    if qtype == 'uint8':
        return 0.0, 255.0
    else:
        return -128.0, 127.0


def test_quantize_float32_to_int8():
    shape = rand_shape_nd(4)
    data = rand_ndarray(shape, 'default', dtype='float32')
    min_range = mx.nd.min(data)
    max_range = mx.nd.max(data)
    qdata, min_val, max_val = mx.nd.contrib.quantize(data, min_range, max_range, out_type='int8')
    data_np = data.asnumpy()
    min_range = min_range.asscalar()
    max_range = max_range.asscalar()
    real_range = onp.maximum(onp.abs(min_range), onp.abs(max_range))
    quantized_range = 127.0
    scale = quantized_range / real_range
    assert qdata.dtype == onp.int8
    assert min_val.dtype == onp.float32
    assert max_val.dtype == onp.float32
    assert same(min_val.asscalar(), -real_range)
    assert same(max_val.asscalar(), real_range)
    qdata_np = (onp.sign(data_np) * onp.minimum(onp.abs(data_np) * scale + 0.5, quantized_range)).astype(onp.int8)
    assert_almost_equal(qdata.asnumpy(), qdata_np, atol = 1)

def test_calibrated_quantize_v2_bfloat16_to_int8():
    shape = rand_shape_nd(4)
    data = mx.nd.random.normal(0, 1, shape).astype('bfloat16')
    min_range = mx.nd.min(data).asscalar()
    max_range = mx.nd.max(data).asscalar()
    qdata, min_val, max_val = mx.nd.contrib.quantize_v2(data, 'int8', min_range, max_range)
    data_np = data.asnumpy()
    real_range = onp.maximum(onp.abs(min_range), onp.abs(max_range))
    quantized_range = 127.0
    scale = quantized_range / real_range
    assert qdata.dtype == onp.int8
    assert min_val.dtype == onp.float32
    assert max_val.dtype == onp.float32
    assert same(min_val.asscalar(), -real_range)
    assert same(max_val.asscalar(), real_range)
    qdata_np = (onp.sign(data_np) * onp.minimum(onp.abs(data_np) * scale + 0.5, quantized_range)).astype(onp.int8)
    assert_almost_equal(qdata.asnumpy(), qdata_np, atol=1)

def test_dequantize_int8_to_float32():

    def get_test_data(real_range, qdata_np):
        qdata = mx.nd.array(qdata_np, dtype=onp.int8)
        min_range = mx.nd.array([-real_range], dtype=onp.float32)
        max_range = mx.nd.array([real_range], dtype=onp.float32)
        return qdata, min_range, max_range

    def baseline_dequantization(qdata, real_range, qdata_np):
        quantized_range = 127.0
        scale = real_range / quantized_range
        data_np = qdata_np * scale
        return data_np

    def test_nd_array_dequantization(qdata, min_range, max_range, expected_result):
        data = mx.nd.contrib.dequantize(qdata, min_range, max_range, out_type='float32')
        assert data.dtype == onp.float32
        assert_almost_equal(data.asnumpy(), expected_result, atol = 1)

    def test_symbolic_api_dequantization(qdata, min_range, max_range, expected_result):
        sym_data = mx.sym.Variable('data')
        sym_min_range = mx.sym.Variable('min_range')
        sym_max_range = mx.sym.Variable('max_range')
        dequant = mx.sym.contrib.dequantize(sym_data, sym_min_range,
                                            sym_max_range, out_type='float32')
        out = dequant._bind(ctx=mx.current_device(),
                           args={'data':qdata, 'min_range':min_range, 'max_range':max_range})
        data = out.forward()[0]
        assert data.dtype == onp.float32
        assert_almost_equal(data.asnumpy(), expected_result, atol = 1)

    real_range = 128
    shape = rand_shape_nd(4)
    qdata_np = onp.random.uniform(low=-127, high=127, size=shape).astype(dtype=onp.int8)
    qdata, min_range, max_range = get_test_data(real_range, qdata_np)
    expected_result = baseline_dequantization(qdata, real_range, qdata_np)
    # test nd array implementation.
    test_nd_array_dequantization(qdata, min_range, max_range, expected_result)
    # test symbolic api implementaion.
    test_symbolic_api_dequantization(qdata, min_range, max_range, expected_result)


def test_requantize_int32_to_int8():
    def quantized_int32_to_float(qdata, min_range, max_range):
        assert qdata.dtype == 'int32'
        quantized_range = onp.iinfo('int32').max
        real_range = onp.maximum(onp.abs(min_range), onp.abs(max_range))
        scale = float(real_range) / float(quantized_range)
        return qdata.astype('float32') * scale

    def float_to_quantized_int8(data, min_range, max_range):
        assert data.dtype == 'float32'
        real_range = onp.maximum(onp.abs(min_range), onp.abs(max_range))
        quantized_range = onp.iinfo('int8').max
        scale = float(quantized_range) / float(real_range)
        return (onp.sign(data) * onp.minimum(onp.abs(data) * scale + 0.5, quantized_range)).astype('int8')

    def requantize(qdata, min_data, max_data, real_range):
        data = quantized_int32_to_float(qdata, min_data, max_data)
        output = float_to_quantized_int8(data, -real_range, real_range)
        return output, -real_range, real_range

    def requantize_baseline(qdata, min_data, max_data, min_calib_range=None, max_calib_range=None):
        if min_calib_range is not None and max_calib_range is not None:
            real_range = onp.maximum(onp.abs(min_calib_range), onp.abs(max_calib_range))
            return requantize(qdata, min_data, max_data, real_range)
        else:
            min_range = quantized_int32_to_float(onp.min(qdata), min_data, max_data)
            max_range = quantized_int32_to_float(onp.max(qdata), min_data, max_data)
            return requantize(qdata, min_data, max_data, onp.maximum(onp.abs(min_range), onp.abs(max_range)))

    def check_requantize(shape, min_calib_range=None, max_calib_range=None):
        qdata = mx.nd.random.uniform(low=-1000.0, high=1000.0, shape=shape).astype('int32')
        min_range = mx.nd.array([-1010.0])
        max_range = mx.nd.array([1020.0])
        if min_calib_range is None or max_calib_range is None:
            qdata_int8, min_output, max_output = mx.nd.contrib.requantize(qdata, min_range, max_range)
        else:
            qdata_int8, min_output, max_output = mx.nd.contrib.requantize(qdata, min_range, max_range,
                                                                          min_calib_range=min_calib_range,
                                                                          max_calib_range=max_calib_range)

        qdata_int8_np, min_output_np, max_output_np = requantize_baseline(qdata.asnumpy(), min_range.asscalar(),
                                                                          max_range.asscalar(),
                                                                          min_calib_range=min_calib_range,
                                                                          max_calib_range=max_calib_range)
        assert_almost_equal(qdata_int8.asnumpy(), qdata_int8_np, atol = 1)
        assert_almost_equal(min_output.asnumpy(), onp.array([min_output_np]))
        assert_almost_equal(max_output.asnumpy(), onp.array([max_output_np]))

    @use_np
    def check_requantize_with_gluon(shape, min_calib_range=None, max_calib_range=None):
        qdata = mx.np.random.uniform(low=-1000.0, high=1000.0, size=shape).astype('int32')
        min_range = mx.np.array([-1010.0])
        max_range = mx.np.array([1020.0])

        class RequantizeBlock(mx.gluon.nn.HybridBlock):
            def __init__(self, min_calib_range=None, max_calib_range=None, **kwargs):
                super(RequantizeBlock, self).__init__(**kwargs)
                self.min_calib_range = min_calib_range
                self.max_calib_range = max_calib_range

            def forward(self, x, min_range, max_range):
                if self.min_calib_range is not None and self.max_calib_range is not None:
                    out = npx.requantize(x, min_range, max_range,
                                         min_calib_range=self.min_calib_range,
                                         max_calib_range=self.max_calib_range)
                else:
                    out = npx.requantize(x, min_range, max_range)
                return out

        requant = RequantizeBlock(min_calib_range, max_calib_range)  # m*_calib_ranges can be None
        qdata_int8, min_output, max_output = requant(qdata, min_range, max_range)

        qdata_int8_np, min_output_np, max_output_np = requantize_baseline(qdata.asnumpy(), min_range.item(),
                                                                          max_range.item(),
                                                                          min_calib_range=min_calib_range,
                                                                          max_calib_range=max_calib_range)
        assert_almost_equal(qdata_int8.asnumpy(), qdata_int8_np, atol = 1)
        assert_almost_equal(min_output.asnumpy(), onp.array([min_output_np]))
        assert_almost_equal(max_output.asnumpy(), onp.array([max_output_np]))

    # test with gluon API.
    check_requantize_with_gluon((3, 4, 10, 10))
    check_requantize_with_gluon((32, 3, 23, 23))
    check_requantize_with_gluon((3, 4, 10, 10), min_calib_range=-1050.0, max_calib_range=1040.0)
    check_requantize_with_gluon((32, 3, 23, 23), min_calib_range=-134.349, max_calib_range=523.43)
    # Test with nd array API
    check_requantize((3, 4, 10, 10))
    check_requantize((32, 3, 23, 23))
    check_requantize((3, 4, 10, 10), min_calib_range=-1050.0, max_calib_range=1040.0)
    check_requantize((32, 3, 23, 23), min_calib_range=-134.349, max_calib_range=523.43)


@use_np
def test_quantized_conv():
    def check_quantized_conv(data_shape, kernel, num_filter, pad, stride, dilate, use_bias, qdtype):
        if is_test_for_native_cpu():
            print('skipped testing quantized_conv for native cpu since it is not supported yet')
            return
        elif is_test_for_dnnl():
            # (TODO)Xinyu: https://github.com/apache/mxnet/issues/16830
            print('skipped testing quantized_conv for oneDNN cpu since it is a flaky case')
            return
        elif qdtype == 'uint8' and is_test_for_gpu():
            print('skipped testing quantized_conv for gpu uint8 since it is not supported yet')
            return
        elif is_test_for_gpu() and len(data_shape) != 4:
            print('skipped testing quantized_conv for gpu 5d layout since it is not supported yet')
            return

        # run fp32 conv
        if len(data_shape) == 4:
            convfp32 = mx.gluon.nn.Conv2D(channels=num_filter, kernel_size=kernel, strides=stride,
                                          padding=pad, dilation=dilate, use_bias=use_bias)
        elif len(data_shape) == 5:
            convfp32 = mx.gluon.nn.Conv3D(channels=num_filter, kernel_size=kernel, strides=stride,
                                          padding=pad, dilation=dilate, use_bias=use_bias)
        else:
            print('unsupported shape')
            assert False

        if qdtype == 'uint8':
            data_low = 0.0
            data_high = 127.0
        else:
            data_low = -127.0
            data_high = 127.0

        convfp32.initialize()
        input_data = mx.np.random.uniform(low=data_low,
                                          high=data_high,
                                          size=data_shape
                                         ).astype('int32').astype('float32')
        convfp32(input_data) # initialize params
        npx.waitall()
        fp32_params = convfp32.collect_params()
        weight_shape = fp32_params['weight'].shape
        new_args = dict()
        new_args['weight'] = mx.np.random.uniform(low=-127.0,
                                                  high=127.0,
                                                  size=weight_shape
                                                 ).astype('int32').astype('float32')
        if use_bias:
           new_args['bias'] = mx.np.random.uniform(low=-127.0,
                                                   high=127.0,
                                                   size=fp32_params['bias'].shape
                                                  ).astype('int32').astype('float32')
        convfp32.load_dict(new_args, cast_dtype=True, dtype_source='saved')

        output = convfp32(input_data)

        # run quantized conv
        class QuantConv(mx.gluon.nn.HybridBlock):
            def __init__(self, channels, kernel_size, strides=(1, 1),
                         padding=(0, 0), dilation=(1, 1), use_bias=True, **kwargs):
                super(QuantConv, self).__init__(**kwargs)
                self.use_bias = use_bias
                self._kwargs = {'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                                'pad': padding, 'num_filter': channels, 'no_bias': not use_bias, 'num_group': 1,
                                'layout': 'NCHW'}

                self.min_data = mx.gluon.Parameter('min_data', dtype='float32', shape=(1), allow_deferred_init=True)
                self.max_data = mx.gluon.Parameter('max_data', dtype='float32', shape=(1), allow_deferred_init=True)

                self.weight = mx.gluon.Parameter('weight', dtype='int8', shape=weight_shape, allow_deferred_init=True)
                self.min_weight = mx.gluon.Parameter('min_weight', dtype='float32', shape=(1), allow_deferred_init=True)
                self.max_weight = mx.gluon.Parameter('max_weight', dtype='float32', shape=(1), allow_deferred_init=True)

                if use_bias:
                    self.bias = mx.gluon.Parameter('bias', dtype='int8', shape=(num_filter,), allow_deferred_init=True)
                    self.min_bias = mx.gluon.Parameter('min_bias', dtype='float32', shape=(1), allow_deferred_init=True)
                    self.max_bias = mx.gluon.Parameter('max_bias', dtype='float32', shape=(1), allow_deferred_init=True)

            def forward(self, x):
                device = x.device
                weight = self.weight.data().to_device(device)
                bias = self.bias.data().to_device(device) if self.use_bias else None
                min_data = self.min_data.data().to_device(device)
                max_data = self.max_data.data().to_device(device)
                min_weight = self.min_weight.data().to_device(device)
                max_weight = self.max_weight.data().to_device(device)
                min_bias = self.min_bias.data().to_device(device) if self.use_bias else None
                max_bias = self.max_bias.data().to_device(device) if self.use_bias else None
                out = npx.quantized_conv(data=x, weight=weight, bias=bias,
                                         min_data=min_data, max_data=max_data,
                                         min_weight=min_weight, max_weight=max_weight,
                                         min_bias=min_bias, max_bias=max_bias,
                                         **self._kwargs)
                return out

        convint8 = QuantConv(channels=num_filter, kernel_size=kernel, strides=stride,
                             padding=pad, dilation=dilate, use_bias=use_bias)

        quantized_range = 127.0
        qargs = {
            'weight': new_args['weight'].astype('int8'),
            'min_data': mx.np.array([-quantized_range]),
            'max_data': mx.np.array([quantized_range]),
            'min_weight': mx.np.array([-quantized_range]),
            'max_weight': mx.np.array([quantized_range])
        }
        if use_bias:
            qargs.update({
                'bias': new_args['bias'].astype('int8'),
                'min_bias': mx.np.array([-quantized_range]),
                'max_bias': mx.np.array([quantized_range]),
            })

        convint8.load_dict(qargs, cast_dtype=True, dtype_source='saved')

        qoutput, min_range, max_range = convint8(input_data.astype(qdtype))

        if use_bias:
            # with adding bias, accuracy loss should not be greater than one
            diff = mx.np.abs(output - qoutput.astype(output.dtype))
            cond = mx.np.less(2, diff).sum().item()
            assert cond == 0
        else:
            assert_almost_equal(output.asnumpy(), qoutput.asnumpy(), atol = 1)

    for qdtype in ['int8', 'uint8']:
        check_quantized_conv((3, 4, 28, 28), (3, 3), 128, (1, 1), (1, 1), (1, 1), True, qdtype)
        check_quantized_conv((3, 4, 28, 28), (3, 3), 128, (1, 1), (1, 1), (1, 1), False, qdtype)
        check_quantized_conv((1, 3, 4, 28, 28), (1, 3, 3), 128, (1, 1, 1), (1, 1, 1), (1, 1, 1), False, qdtype)
        check_quantized_conv((1, 3, 4, 28, 28), (1, 3, 3), 128, (1, 1, 1), (1, 1, 1), (1, 1, 1), True, qdtype)
        check_quantized_conv((1, 3, 4, 28, 28), (1, 3, 3), 128, (1, 1, 1), (1, 1, 1), (2, 2, 2), False, qdtype)
        check_quantized_conv((1, 3, 4, 28, 28), (1, 3, 3), 128, (1, 1, 1), (1, 1, 1), (2, 2, 2), True, qdtype)


@use_np
def test_quantized_elemwise_add():
    def check_quantized_elemwise_add(data_shape, qdtypeA, qdtypeB):
        if is_test_for_native_cpu():
            print('skipped testing quantized_elemwise_add for native cpu since it is not supported yet')
            return
        elif (qdtypeA != 'uint8' and qdtypeA != 'int8') or (qdtypeB != 'uint8' and qdtypeB != 'int8'):
            print('skipped testing quantized_elemwise_add for not supported data type')
            return
        elif is_test_for_gpu():
            print('skipped testing quantized_elemwise_add for gpu since it is not supported yet')
            return

        class ElemwiseSumBlock(mx.gluon.nn.HybridBlock):
            def __init__(self, **kwargs):
                super(ElemwiseSumBlock, self).__init__(**kwargs)

            def forward(self, dataA, dataB):
                return dataA + dataB

        class QuantElemwiseSumBlock(mx.gluon.nn.HybridBlock):
            def __init__(self, **kwargs):
                super(QuantElemwiseSumBlock, self).__init__(**kwargs)

            def forward(self, dataA, dataB, dataA_min, dataA_max, dataB_min, dataB_max):
                return npx.quantized_elemwise_add(dataA, dataB, dataA_min, dataA_max, dataB_min, dataB_max)

        elemwise_add_fp32 = ElemwiseSumBlock()

        dataA_low, dataA_high = get_low_high(qdtypeA)
        dataB_low, dataB_high = get_low_high(qdtypeB)

        dataA_val = mx.np.random.uniform(low=dataA_low, high=dataA_high, size=data_shape).astype('int32').astype('float32')
        dataB_val = mx.np.random.uniform(low=dataB_low, high=dataB_high, size=data_shape).astype('int32').astype('float32')

        output = elemwise_add_fp32(dataA_val, dataB_val)
        mx.nd.waitall()

        #run quantized
        quantized_elemwise_add = QuantElemwiseSumBlock()
        dataA_val_int8 = dataA_val.astype(qdtypeA)
        dataB_val_int8 = dataB_val.astype(qdtypeB)

        quantized_range = 127.0
        min_dataA = mx.np.array([dataA_low])
        max_dataA = mx.np.array([dataA_high])
        min_dataB = mx.np.array([dataB_low])
        max_dataB = mx.np.array([dataB_high])
        qoutput, min_range, max_range = quantized_elemwise_add(dataA_val_int8, dataB_val_int8,
                                                               min_dataA, max_dataA,
                                                               min_dataB, max_dataB)
        int8_rslt = qoutput.astype(output.dtype) * max_range / 0x7fffffff
        diff = mx.np.abs(output - int8_rslt)
        cond = mx.np.less(2, diff).sum().item()
        assert cond == 0

    check_quantized_elemwise_add((4, 6), 'uint8', 'int8')
    check_quantized_elemwise_add((13, 74, 52), 'uint8', 'uint8')
    check_quantized_elemwise_add((3, 4, 56, 56), 'int8', 'uint8')
    check_quantized_elemwise_add((32, 56, 64, 11), 'int8', 'int8')

@use_np
def test_quantized_npi_add():
    def check_quantized_npi_add(data_shape,  qdtypeA, qdtypeB, broadcast=None):
        if is_test_for_native_cpu():
            print('skipped testing quantized_npi_add for native cpu since it is not supported yet')
            return
        elif (qdtypeA != 'uint8' and qdtypeA != 'int8') or (qdtypeB != 'uint8' and qdtypeB != 'int8'):
            print('skipped testing quantized_npi_add for not supported data type')
            return
        elif is_test_for_gpu():
            print('skipped testing quantized_npi_add for gpu since it is not supported yet')
            return

        class ElemwiseSumBlock(mx.gluon.nn.HybridBlock):
            def __init__(self, **kwargs):
                super(ElemwiseSumBlock, self).__init__(**kwargs)

            def forward(self, dataA, dataB):
                return dataA + dataB

        class QuantElemwiseSumBlock(mx.gluon.nn.HybridBlock):
            def __init__(self, **kwargs):
                super(QuantElemwiseSumBlock, self).__init__(**kwargs)

            def forward(self, dataA, dataB, dataA_min, dataA_max, dataB_min, dataB_max):
                return npx.quantized_npi_add(dataA, dataB, dataA_min, dataA_max, dataB_min, dataB_max)

        elemwise_add_fp32 = ElemwiseSumBlock()

        dataA_low, dataA_high = get_low_high(qdtypeA)
        dataB_low, dataB_high = get_low_high(qdtypeB)

        data_shapeA = data_shape
        data_shapeB = data_shape

        if broadcast :
            if broadcast == 'A':
                data_shapeA = ()
                for index in range(len(data_shape)):
                    data_shapeA += (1,)
            else:
                data_shapeB = ()
                for index in range(len(data_shape)):
                    data_shapeB += (1,)

        dataA_val = mx.np.random.uniform(low=dataA_low, high=dataA_high, size=data_shapeA).astype('int32').astype('float32')
        dataB_val = mx.np.random.uniform(low=dataB_low, high=dataB_high, size=data_shapeB).astype('int32').astype('float32')

        output = elemwise_add_fp32(dataA_val, dataB_val)

        #run quantized
        quantized_elemwise_add = QuantElemwiseSumBlock()
        dataA_val_int8 = dataA_val.astype(qdtypeA)
        dataB_val_int8 = dataB_val.astype(qdtypeB)
        quantized_range = 127.0
        min_dataA = mx.np.array([dataA_low])
        max_dataA = mx.np.array([dataA_high])
        min_dataB = mx.np.array([dataB_low])
        max_dataB = mx.np.array([dataB_high])
        qoutput, min_range, max_range = quantized_elemwise_add(dataA_val_int8, dataB_val_int8,
                                                               min_dataA, max_dataA,
                                                               min_dataB, max_dataB)
        int8_rslt = qoutput.astype(output.dtype) * max_range / 0x7fffffff
        diff = mx.np.abs(output - int8_rslt)
        cond = mx.np.less(2, diff).sum().item()
        assert cond == 0

    check_quantized_npi_add((4, 6), 'uint8', 'int8')
    check_quantized_npi_add((13, 74, 52), 'uint8', 'uint8')
    check_quantized_npi_add((3, 4, 56, 56), 'int8', 'uint8')
    check_quantized_npi_add((32, 56, 64, 11), 'int8', 'int8')

    check_quantized_npi_add((4, 6), 'uint8', 'int8', 'A')
    check_quantized_npi_add((13, 74, 52), 'uint8', 'uint8', 'B')
    check_quantized_npi_add((3, 4, 56, 56), 'int8', 'uint8', 'A')
    check_quantized_npi_add((32, 56, 64, 11), 'int8', 'int8', 'B')


@use_np
def test_quantized_elemwise_mul():
    def check_quantized_elemwise_mul(data_shape, qtype):
        if is_test_for_native_cpu():
            print('skipped testing quantized_elemwise_mul for native cpu since it is not supported yet')
            return
        elif qtype != 'int8':
            print('skipped testing quantized_elemwise_mul for not supported data type')
            return
        elif is_test_for_gpu():
            print('skipped testing quantized_elemwise_mul for gpu since it is not supported yet')
            return

        class ElemwiseMulBlock(mx.gluon.nn.HybridBlock):
            def __init__(self, **kwargs):
                super(ElemwiseMulBlock, self).__init__(**kwargs)

            def forward(self, dataA, dataB):
                return mx.np.multiply(dataA, dataB)

        class QuantElemwiseMulBlock(mx.gluon.nn.HybridBlock):
            def __init__(self, **kwargs):
                super(QuantElemwiseMulBlock, self).__init__(**kwargs)

            def forward(self, dataA, dataB, dataA_min, dataA_max, dataB_min, dataB_max):
                return npx.quantized_elemwise_mul(dataA, dataB, dataA_min, dataA_max, dataB_min, dataB_max)

        elemwise_mul_fp32 = ElemwiseMulBlock()
        data_low, data_high = get_low_high(qtype)

        dataA_val = mx.np.random.uniform(low=data_low, high=data_high, size=data_shape).astype('int32').astype('float32')
        dataB_val = mx.np.random.uniform(low=data_low, high=data_high, size=data_shape).astype('int32').astype('float32')

        output = elemwise_mul_fp32(dataA_val, dataB_val)

        quantized_elemwise_mul = QuantElemwiseMulBlock()
        dataA_val_int8 = dataA_val.astype(qtype)
        dataB_val_int8 = dataB_val.astype(qtype)
        quantized_range = 127.0
        min_dataA = mx.np.array([data_low])
        max_dataA = mx.np.array([data_high])
        min_dataB = mx.np.array([data_low])
        max_dataB = mx.np.array([data_high])
        qoutput, min_range, max_range = quantized_elemwise_mul(dataA_val_int8, dataB_val_int8,
                                                               min_dataA, max_dataA,
                                                               min_dataB, max_dataB)

        fp32_rslt = output.asnumpy()
        int8_rslt = qoutput.astype(output.dtype)
        assert_almost_equal(fp32_rslt, int8_rslt, atol = 1e-4)

    for qtype in ['int8', 'uint8']:
        check_quantized_elemwise_mul((4, 6), qtype)
        check_quantized_elemwise_mul((13, 74, 52), qtype)
        check_quantized_elemwise_mul((3, 4, 56, 56), qtype)
        check_quantized_elemwise_mul((32, 56, 64, 11), qtype)


@use_np
def test_quantized_pooling():
    def check_quantized_pooling(data_shape, kernel, pool_type, pad, stride, global_pool, qdtype, convention='valid'):
        if is_test_for_native_cpu():
            print('skipped testing quantized_pooling for native cpu since it is not supported yet')
            return
        elif qdtype == 'uint8' and is_test_for_gpu():
            print('skipped testing quantized_pooling for gpu uint8 since it is not supported yet')
            return
        elif is_test_for_gpu() and len(data_shape) != 4:
            print('skipped testing quantized_pooling for gpu 5d layout since it is not supported yet')
            return

        class PoolingBlock(mx.gluon.nn.HybridBlock):
            def __init__(self, kernel=kernel, pad=pad, stride=stride,
                         pool_type=pool_type, global_pool=global_pool, cudnn_off=False,
                         pooling_convention=convention):
                super(PoolingBlock, self).__init__()
                self._kwargs = {'kernel': kernel, 'pad': pad, 'stride': stride,
                                'pool_type': pool_type, 'global_pool': global_pool,
                                'cudnn_off': False, 'pooling_convention': convention}

            def forward(self, data):
                return npx.pooling(data, **self._kwargs)

        class QuantPoolingBlock(mx.gluon.nn.HybridBlock):
            def __init__(self, kernel=kernel, pad=pad, stride=stride,
                         pool_type=pool_type, global_pool=global_pool,
                         cudnn_off=False, pooling_convention=convention):
                super(QuantPoolingBlock, self).__init__()

                self._kwargs = {'kernel': kernel, 'pad': pad, 'stride': stride,
                                'pool_type': pool_type, 'global_pool': global_pool, 'cudnn_off': False,
                                'pooling_convention':convention}

            def forward(self, data, min_data, max_data):
                return npx.quantized_pooling(data, min_data, max_data, **self._kwargs)

        pooling_fp32 = PoolingBlock()
        if qdtype == 'uint8':
            data_low = 0.0
            data_high = 127.0
        else:
            data_low = -127.0
            data_high = 127.0

        input_data = mx.np.random.uniform(low=data_low,
                                          high=data_high,
                                          size=data_shape
                                         ).astype('int32').astype('float32')
        output = pooling_fp32(input_data)

        quantized_pooling = QuantPoolingBlock(kernel=kernel, pad=pad, stride=stride,
                                              pool_type=pool_type, global_pool=global_pool,
                                              pooling_convention=convention)

        int8_input_data = input_data.astype(qdtype)
        quantized_range = 127.0
        min_data = mx.np.array([-quantized_range])
        max_data = mx.np.array([quantized_range])

        qoutput, min_range, max_range = quantized_pooling(int8_input_data, min_data, max_data)

        if pool_type == 'max':
            assert_almost_equal(output.asnumpy(), qoutput.asnumpy())
        elif pool_type == 'avg':  # for avg pooling, fp32 and int8 may be different due to rounding errors
            diff = mx.np.abs(output - qoutput.astype(output.dtype))
            cond = mx.np.less(2, diff).sum().item()
            assert cond == 0

    for qdtype in ['int8', 'uint8']:
        check_quantized_pooling((3, 4, 56, 56), (3, 3), 'max', (0, 0), (2, 2), False, qdtype)
        check_quantized_pooling((3, 4, 56, 56), (3, 3), 'max', (0, 0), (2, 2), True, qdtype)
        check_quantized_pooling((3, 512, 7, 7), (7, 7), 'avg', (0, 0), (1, 1), False, qdtype)
        check_quantized_pooling((3, 512, 7, 7), (7, 7), 'avg', (0, 0), (1, 1), True, qdtype)
        check_quantized_pooling((3, 4, 3, 56, 56), (1, 3, 3), 'max', (0, 0, 0), (1, 2, 2), False, qdtype)
        check_quantized_pooling((3, 4, 3, 56, 56), (1, 3, 3), 'max', (0, 0, 0), (1, 2, 2), True, qdtype)
        check_quantized_pooling((3, 512, 3, 7, 7), (1, 7, 7), 'avg', (0, 0, 0), (1, 2, 2), False, qdtype)
        check_quantized_pooling((3, 512, 3, 7, 7), (1, 7, 7), 'avg', (0, 0, 0), (1, 2, 2), True, qdtype)

        check_quantized_pooling((3, 4, 56, 56), (3, 3), 'max', (0, 0), (2, 2), False, qdtype, 'full')
        check_quantized_pooling((3, 4, 56, 56), (3, 3), 'max', (0, 0), (2, 2), True, qdtype, 'full')
        check_quantized_pooling((3, 512, 7, 7), (7, 7), 'avg', (0, 0), (1, 1), False, qdtype, 'full')
        check_quantized_pooling((3, 512, 7, 7), (7, 7), 'avg', (0, 0), (1, 1), True, qdtype, 'full')
        check_quantized_pooling((3, 4, 3, 56, 56), (1, 3, 3), 'max', (0, 0, 0), (1, 2, 2), False, qdtype, 'full')
        check_quantized_pooling((3, 4, 3, 56, 56), (1, 3, 3), 'max', (0, 0, 0), (1, 2, 2), True, qdtype, 'full')
        check_quantized_pooling((3, 512, 3, 7, 7), (1, 7, 7), 'avg', (0, 0, 0), (1, 2, 2), False, qdtype, 'full')
        check_quantized_pooling((3, 512, 3, 7, 7), (1, 7, 7), 'avg', (0, 0, 0), (1, 2, 2), True, qdtype, 'full')


@use_np
def test_quantized_fc():
    def check_quantized_fc(data_shape, num_hidden, use_bias, qdtype, flatten=True):
        if is_test_for_native_cpu():
            hasMKL = False
            for key in os.environ.keys():
                if operator.eq(key, "BUILD_TAG"):
                    if os.environ['BUILD_TAG'].find("MKL") != -1:
                        hasMKL = True
                    break
            if hasMKL == False:
                print('skipped testing quantized_fc on cpu since s8u8s32 is only supported by MKL BLAS library')
                return
        elif qdtype == 'uint8' and is_test_for_gpu():
            print('skipped testing quantized_fc for gpu uint8 since it is not supported yet')
            return

        def maxabs(a, b):
            return mx.np.maximum(mx.np.abs(a), mx.np.abs(b))

        int8_range = 127.0
        if qdtype == 'uint8':
            data_low = 0.0
            data_high = 63.0
            quantized_range = 255.0
        else:
            data_low = -63.0
            data_high = 63.0
            quantized_range = 127.0

        data = mx.np.random.uniform(low=data_low,
                                    high=data_high,
                                    size=data_shape
                                   ).astype('int32').astype('float32')
        fc_fp32 = mx.gluon.nn.Dense(units=num_hidden, use_bias=use_bias, flatten=flatten)
        fc_fp32.initialize()
        fc_fp32(data)
        npx.waitall()
        fp32_params = fc_fp32.collect_params()
        weight_shape = fp32_params['weight'].shape

        new_args = dict()
        new_args['weight'] = mx.np.random.uniform(low=data_low,
                                                  high=data_high,
                                                  size=fp32_params['weight'].shape
                                                 ).astype('int32').astype('float32')
        data_min = mx.np.min(data).astype('float32')
        data_max = mx.np.max(data).astype('float32')
        weight_min = mx.np.min(new_args['weight']).astype('float32')
        weight_max = mx.np.max(new_args['weight']).astype('float32')
        data_range = maxabs(data_min, data_max)
        weight_range = maxabs(weight_min, weight_max)

        if use_bias:
            bias = mx.np.random.uniform(low=data_low,
                                        high=data_high,
                                        size=fp32_params['bias'].shape
                                       ).astype('int32').astype('float32')
            bias_min = mx.np.min(bias).astype('float32')
            bias_max = mx.np.max(bias).astype('float32')
            bias_range = maxabs(bias_min, bias_max)

            bias_scale = int8_range / bias_range
            data_scale = quantized_range / data_range
            weight_scale = int8_range / weight_range
            bias_int32_rescale = data_scale * weight_scale / bias_scale
            new_bias = bias.astype('float32') * bias_int32_rescale
            new_args['bias'] = new_bias.astype('int32').astype('float32')

        fc_fp32.load_dict(new_args, cast_dtype=True, dtype_source='saved')
        output = fc_fp32(data)

        class QuantFC(mx.gluon.nn.HybridBlock):
            def __init__(self, num_hidden, use_bias, flatten, **kwargs):
                super(QuantFC, self).__init__(**kwargs)
                self.use_bias = use_bias
                self._kwargs = {'num_hidden': num_hidden, 'no_bias': not use_bias, 'flatten': flatten}

                self.min_data = mx.gluon.Parameter('min_data', dtype='float32', shape=(1), allow_deferred_init=True)
                self.max_data = mx.gluon.Parameter('max_data', dtype='float32', shape=(1), allow_deferred_init=True)

                self.weight = mx.gluon.Parameter('weight', dtype='int8', shape=weight_shape, allow_deferred_init=True)
                self.min_weight = mx.gluon.Parameter('min_weight', dtype='float32', shape=(1), allow_deferred_init=True)
                self.max_weight = mx.gluon.Parameter('max_weight', dtype='float32', shape=(1), allow_deferred_init=True)

                if use_bias:
                    self.bias = mx.gluon.Parameter('bias', dtype='int8', shape=(num_hidden,), allow_deferred_init=True)
                    self.min_bias = mx.gluon.Parameter('min_bias', dtype='float32', shape=(1), allow_deferred_init=True)
                    self.max_bias = mx.gluon.Parameter('max_bias', dtype='float32', shape=(1), allow_deferred_init=True)

            def forward(self, x):
                device = x.device
                weight = self.weight.data().to_device(device)
                bias = self.bias.data().to_device(device) if self.use_bias else None
                min_data = self.min_data.data().to_device(device)
                max_data = self.max_data.data().to_device(device)
                min_weight = self.min_weight.data().to_device(device)
                max_weight = self.max_weight.data().to_device(device)
                min_bias = self.min_bias.data().to_device(device) if self.use_bias else None
                max_bias = self.max_bias.data().to_device(device) if self.use_bias else None
                out = npx.quantized_fully_connected(data=x, weight=weight, bias=bias,
                                                    min_data=min_data, max_data=max_data,
                                                    min_weight=min_weight, max_weight=max_weight,
                                                    min_bias=min_bias, max_bias=max_bias,
                                                    **self._kwargs)
                return out

        fc_int8 = QuantFC(num_hidden=num_hidden, use_bias=use_bias, flatten=flatten)
        qargs = {
            'weight': new_args['weight'].astype('int8'),
            'min_data': mx.np.array([-data_range]),
            'max_data': mx.np.array([data_range]),
            'min_weight': mx.np.array([-weight_range]),
            'max_weight': mx.np.array([weight_range])
        }
        if use_bias:
            qargs.update({
                'bias': bias.astype('int8'),
                'min_bias': mx.np.array([-bias_range]),
                'max_bias': mx.np.array([bias_range]),
            })

        fc_int8.load_dict(qargs, cast_dtype=True, dtype_source='saved')

        qoutput, min_range, max_range = fc_int8(data.astype(qdtype))

        if use_bias:
            # with adding bias, accuracy loss should not be greater than one
            diff = mx.np.abs(output - qoutput.astype(output.dtype))
            cond = mx.np.less(2, diff).sum().item()
            assert cond == 0
        else:
            assert_almost_equal(output.asnumpy(), qoutput.asnumpy())

    for qdtype in ['int8', 'uint8']:
        if is_test_for_dnnl():
            check_quantized_fc((32, 512, 2), 100, False, qdtype, flatten=False)
            check_quantized_fc((32, 512, 2), 100, True, qdtype, flatten=False)
            check_quantized_fc((32, 512, 2, 2), 100, False, qdtype, flatten=False)
            check_quantized_fc((32, 512, 2, 2), 100, True, qdtype, flatten=False)
        check_quantized_fc((32, 512, 2, 2), 100, False, qdtype)
        check_quantized_fc((32, 111, 2, 2), 100, False, qdtype)
        check_quantized_fc((32, 512, 2, 2), 100, True, qdtype)
        check_quantized_fc((32, 111, 2, 2), 100, True, qdtype)
        check_quantized_fc((256, 2048, 2, 2), 800, True, qdtype)
        check_quantized_fc((256, 111, 2, 2), 800, True, qdtype)
        check_quantized_fc((256, 2048, 2, 2), 800, False, qdtype)
        check_quantized_fc((256, 111, 2, 2), 800, False, qdtype)

@use_np
def test_quantized_transpose():
    def check_quantized_transpose(shape, qdtype, axes):
        data_low, data_high = get_low_high(qdtype)
        data = mx.np.random.uniform(low=data_low, high=data_high, size=shape).astype(qdtype).astype('float32')
        min_data = mx.np.array([mx.np.min(data).astype('float32').item()])
        max_data = mx.np.array([mx.np.max(data).astype('float32').item()])
        qdata = data.astype(qdtype)
        output = mx.np.transpose(data, axes=axes)
        qoutput, min_output, max_output = npx.quantized_transpose(qdata, min_data, max_data, axes=axes)
        assert_almost_equal(output.asnumpy(), qoutput.asnumpy())
        assert_almost_equal(min_output.item(), min_data.item())
        assert_almost_equal(max_output.item(), max_data.item())

    for qtype in ['int8', 'uint8']:
        check_quantized_transpose((), qtype, ())
        check_quantized_transpose((2,3), qtype, (1,0))
        check_quantized_transpose((8,21), qtype, (1,0))
        check_quantized_transpose((7,3,9), qtype, (2,1,0))
        check_quantized_transpose((5,3,6,8), qtype, (2,3,0,1))


@use_np
def test_quantized_embedding():
    def check_quantized_embedding(data_shape, input_dim, output_dim):
        if is_test_for_gpu():
            print('skipped testing test_quantized_embedding for gpu since it is not supported yet')
            return

        def maxabs(a, b):
            return mx.np.maximum(mx.np.abs(a), mx.np.abs(b))

        data = mx.np.random.uniform(low=0,
                                    high=input_dim,
                                    size=data_shape
                                   ).astype('int32').astype('float32')
        embedding_fp32 = mx.gluon.nn.Embedding(input_dim=input_dim, output_dim=output_dim)
        embedding_fp32.initialize()
        embedding_fp32(data)
        npx.waitall()
        fp32_params = embedding_fp32.collect_params()
        weight_shape = fp32_params['weight'].shape
        int8_range = 127.0
        new_params = dict()
        weight = mx.np.random.uniform(low=-int8_range,
                                      high=int8_range,
                                      size=weight_shape
                                     ).astype('int32').astype('float32')
        new_params['weight'] = weight
        embedding_fp32.load_dict(new_params, cast_dtype=True, dtype_source='saved')

        output = embedding_fp32(data)

        weight_min = mx.np.min(weight).astype('float32')
        weight_max = mx.np.max(weight).astype('float32')
        weight_range = maxabs(weight_min, weight_max)

        class QuantEmbedding(mx.gluon.nn.HybridBlock):
            def __init__(self, input_dim=input_dim, output_dim=output_dim, **kwargs):
                super(QuantEmbedding, self).__init__(**kwargs)
                self._kwargs = {'input_dim': input_dim, 'output_dim': output_dim}

                self.weight = mx.gluon.Parameter('weight', dtype='float32', shape=weight_shape, allow_deferred_init=True)
                self.min_weight = mx.gluon.Parameter('min_weight', dtype='float32', shape=(1), allow_deferred_init=True)
                self.max_weight = mx.gluon.Parameter('max_weight', dtype='float32', shape=(1), allow_deferred_init=True)

            def forward(self, x):
                device = x.device
                weight = self.weight.data().to_device(device)
                min_weight = self.min_weight.data().to_device(device)
                max_weight = self.max_weight.data().to_device(device)
                out = npx.quantized_embedding(data=x, weight=weight,
                                              min_weight=min_weight,
                                              max_weight=max_weight,
                                              **self._kwargs)
                return out

        embedding_int8 = QuantEmbedding(input_dim=input_dim, output_dim=output_dim)
        qargs = {
            'weight': weight.astype('int8'),
            'min_weight': mx.np.array([-weight_range]),
            'max_weight': mx.np.array([weight_range])
        }

        embedding_int8.load_dict(qargs, cast_dtype=True, dtype_source='saved')

        qoutput, min_range, max_range = embedding_int8(data)


        assert_almost_equal(output.asnumpy(), qoutput.asnumpy())

    check_quantized_embedding((1,), 1000, 256)
    check_quantized_embedding((1,), 1024, 512)
    check_quantized_embedding((32,), 1000, 256)
    check_quantized_embedding((32,), 1024, 512)


@use_np
def test_quantized_flatten():
    def check_quantized_flatten(shape, qdtype):
        if qdtype == 'uint8':
            data_low = 0.0
            data_high = 127.0
        else:
            data_low = -127.0
            data_high = 127.0
        qdata = mx.np.random.uniform(low=data_low, high=data_high, size=shape).astype(qdtype)
        min_data = mx.np.array([-1023.343], dtype='float32')
        max_data = mx.np.array([2343.324275], dtype='float32')
        qoutput, min_output, max_output = npx.quantized_flatten(qdata, min_data, max_data)
        assert qoutput.ndim == 2
        assert qoutput.shape[0] == qdata.shape[0]
        assert qoutput.shape[1] == onp.prod(qdata.shape[1:])
        assert same(qdata.asnumpy().flatten(), qoutput.asnumpy().flatten())
        assert same(min_data.asnumpy(), min_output.asnumpy())
        assert same(max_data.asnumpy(), max_output.asnumpy())

    for qdtype in ['int8', 'uint8']:
        check_quantized_flatten((10,), qdtype)
        check_quantized_flatten((10, 15), qdtype)
        check_quantized_flatten((10, 15, 18), qdtype)
        check_quantized_flatten((3, 4, 23, 23), qdtype)


@use_np
def test_quantized_act():
    def check_quantized_act(data_shape, qdtype):
        if is_test_for_native_cpu():
            print('skipped testing quantized_act for native cpu since it is not supported yet')
            return
        elif qdtype == 'int8' and is_test_for_dnnl():
            print('skipped testing quantized_act for oneDNN cpu int8 since it is not supported yet')
            return
        elif is_test_for_gpu():
            print('skipped testing quantized_act for gpu since it is not supported yet')
            return

        act_fp32 = mx.gluon.nn.Activation(activation='relu')

        if qdtype == 'uint8':
            data_low = 0.0
            data_high = 127.0
        else:
            data_low = -127.0
            data_high = 127.0

        data = mx.np.random.uniform(low=data_low,
                                    high=data_high,
                                    size=data_shape
                                   ).astype(qdtype).astype('float32')
        output = act_fp32(data)

        class QuantActivation(mx.gluon.nn.HybridBlock):
            def __init__(self, activation, **kwargs):
                super(QuantActivation, self).__init__(**kwargs)
                self._kwargs = {'act_type': activation}

            def forward(self, x, min_data, max_data):
                out = npx.quantized_act(data=x, min_data=min_data, max_data=max_data, **self._kwargs)
                return out

        quantized_act = QuantActivation(activation='relu')

        qdata = data.astype(qdtype)
        quantized_range_min = mx.np.array([mx.np.min(data).astype('float32').item()])
        quantized_range_max = mx.np.array([mx.np.max(data).astype('float32').item()])
        qoutput, min_range, max_range = quantized_act(qdata, quantized_range_min, quantized_range_max)

        assert_almost_equal(output.asnumpy(), qoutput.asnumpy())
        assert_almost_equal(min_range.item(), quantized_range_min.item())
        assert_almost_equal(max_range.item(), quantized_range_max.item())

    for qdtype in ['int8', 'uint8']:
        check_quantized_act((10,), qdtype)
        check_quantized_act((10, 15), qdtype)
        check_quantized_act((10, 15, 18), qdtype)
        check_quantized_act((3, 4, 23, 23), qdtype)


@use_np
def test_quantized_bn():
    def get_mean_var(data):
        axes = list(range(data.ndim))
        del axes[1]
        mean = mx.np.mean(data, axis=axes)
        mean_broad = mx.np.expand_dims(mean, axis=0)
        mean_broad = mx.np.expand_dims(mean_broad, axis=2)
        mean_broad = mx.np.expand_dims(mean_broad, axis=3)
        mean_broad = mx.npx.broadcast_like(mean_broad, data)
        var = mx.np.multiply(data - mean_broad, data - mean_broad)
        axes = list(range(var.ndim))
        del axes[1]
        var = mx.np.mean(var, axis=axes)
        return mean, var

    def check_quantized_bn(data_shape, qdtype):
        if is_test_for_native_cpu():
            print('skipped testing quantize_bn for native cpu since it is not supported yet')
            return
        elif is_test_for_gpu():
            print('skipped testing quantize_bn for gpu since it is not supported yet')
            return

        data_low, data_high = get_low_high(qdtype)

        # run fp32 bn
        bn_fp32 = mx.gluon.nn.BatchNorm(use_global_stats=True, scale=True)
        data = mx.np.random.uniform(low=data_low, high=data_high, size=data_shape)
        bn_fp32.initialize()
        bn_fp32.hybridize()
        bn_fp32(data)
        fp32_params = bn_fp32.collect_params()

        data = mx.np.random.uniform(low=data_low, high=data_high, size=data_shape)
        gamma = mx.np.random.uniform(low=data_low, high=data_high, size=fp32_params['gamma'].shape)
        beta = mx.np.random.uniform(low=data_low, high=data_high, size=fp32_params['beta'].shape)
        running_mean, running_var = get_mean_var(data)
        new_params = {
            'gamma':gamma,
            'beta':beta,
            'running_mean': running_mean,
            'running_var': running_var
        }

        bn_fp32.load_dict(new_params)
        output = bn_fp32(data)

        # generate int8 bn from fp32 bn
        calib_data = mx.gluon.data.DataLoader(data, batch_size=data_shape[0])
        quant_bn = mx.contrib.quant.quantize_net(bn_fp32,
                                                 quantized_dtype=qdtype,
                                                 quantize_mode='full',
                                                 calib_data=calib_data,
                                                 calib_mode='naive',
                                                 num_calib_batches=1,
                                                 device=mx.current_device())

        output_int8_to_fp32 = quant_bn(data)

        assert_almost_equal(output.asnumpy(), output_int8_to_fp32.asnumpy(), rtol=1e-1, atol=8)

    for qdtype in ['int8', 'uint8']:
      check_quantized_bn((32, 512, 4, 4), qdtype)
      check_quantized_bn((32, 1024, 8, 8), qdtype)
      check_quantized_bn((32, 3, 224, 224), qdtype)


def test_quantized_reshape():
    test_cases = [((2, 3, 5, 5),  (-2, -1),         False, (2, 75)),
                  ((2, 3, 5, 5),  (-2, -2, -1),     False, (2, 3, 25)),
                  ((5, 3, 4, 5),  (-2, -1, -2),     False, (5, 15, 4)),
                  ((2, 3, 5, 4),  (-1, -2, -2),     False, (8, 3, 5)),
                  ((2, 3, 5, 5),  (-2, -2, -2, -2), False, (2, 3, 5, 5)),
                  ((2, 1, 4, 5),  (-2, -3, -2, -2), False, (2, 4, 5)),
                  ((1, 1, 4, 1),  (-3, -3, -2, -2), False, (4, 1)),
                  ((1, 1, 1, 1),  (-3, -3, -3, -3), False, ()),
                  ((2, 4, 5, 3),  (-1, 2, 2, 1),    False, (30, 2, 2, 1)),
                  ((2, 3, 5, 6),  (-4,),            False, (2, 3, 5, 6)),
                  ((2, 3, 5, 6),  (6, 1, -4),       False, (6, 1, 5, 6)),
                  ((2, 3, 5, 6),  (-5, -5),         False, (6, 30)),
                  ((2, 3, 5, 6),  (-5, -1),         False, (6, 30)),
                  ((64,),         (-6, 16, 4),      False, (16, 4)),
                  ((64,),         (-6, 16, -1),     False, (16, 4)),
                  ((64, 1, 2, 3), (-6, 16, -1, -4), False, (16, 4, 1, 2, 3)),
                  ((8, 5, 4, 6),  (-4, -1, 3, -6),  True,  (8, 5, 4, 2, 3))]

    def check_quantized_reshape(shape, qdtype, newshape, reverse, expected_ret_shape):
        data_low, data_high = get_low_high(qdtype)
        qdata = mx.np.random.uniform(low=data_low, high=data_high, size=shape).astype(qdtype)
        min_data = mx.np.array([-1023.343], dtype='float32')
        max_data = mx.np.array([2343.324275], dtype='float32')
        qoutput, min_output, max_output = npx.quantized_reshape(qdata, min_data, max_data, newshape=newshape, reverse=reverse)
        assert qoutput.shape == expected_ret_shape
        assert same(qdata.asnumpy().flatten(), qoutput.asnumpy().flatten())
        assert same(min_data.asnumpy(), min_output.asnumpy())
        assert same(max_data.asnumpy(), max_output.asnumpy())

    for qdtype in ['int8', 'uint8']:
        for shape, newshape, reverse, expected_ret_shape in test_cases:
            check_quantized_reshape(shape, qdtype, newshape, reverse, expected_ret_shape)


def test_quantize_params():
    if is_test_for_native_cpu():
        print('skipped testing quantized_params for native cpu since it is not supported yet')
        return

    data = mx.sym.Variable('data')
    conv = mx.sym.Convolution(data, kernel=(1, 1), num_filter=2048, name='conv')
    sym = mx.sym.BatchNorm(data=conv, eps=2e-05, fix_gamma=False, momentum=0.9, use_global_stats=False, name='bn')
    offline_params = [name for name in sym.list_arguments()
                      if not name.startswith('data') and not name.endswith('label')]
    params = {}
    for name in offline_params:
        params[name] = mx.nd.uniform(shape=(2, 2))
    qsym, _ = mx.contrib.quant._quantize_symbol(sym, device=mx.current_device(),
                                                offline_params=offline_params, quantize_mode='full')
    qparams = mx.contrib.quant._quantize_params(qsym, params, min_max_dict = {})
    param_names = params.keys()
    qparam_names = qparams.keys()
    for name in qparam_names:
        if name.startswith('bn'):
            assert name in param_names
        elif name.startswith('conv'):
            assert name not in param_names
            assert name.find('quantize') != -1


class FP32Net(mx.gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(FP32Net, self).__init__(**kwargs)
        self.conv = mx.gluon.nn.Conv2D(channels=16, kernel_size=(1,1))
        self.bn = mx.gluon.nn.BatchNorm(epsilon=2e-05, scale=True, momentum=0.9, use_global_stats=False)
        self.act = mx.gluon.nn.Activation(activation='relu')
        self.pool = mx.gluon.nn.AvgPool2D(pool_size=(4,4))
        self.fc = mx.gluon.nn.Dense(units=10, flatten=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool(out)
        out = self.fc(out)
        return npx.softmax(out)


class FP32MultipleOutputs(mx.gluon.nn.HybridBlock):
    def __init__(self, length, **kwargs):
        super(FP32MultipleOutputs, self).__init__(**kwargs)
        self.length = length
        self.convs = mx.gluon.nn.Conv2D(channels=16, kernel_size=(1,1))
        self.fc = mx.gluon.nn.Dense(units=10, flatten=True)

    def forward(self, x):
        res = npx.slice_channel(x, num_outputs=self.length,
                                axis=1, squeeze_axis=1)
        out = []
        for i in range(self.length):
            out.append(self.convs(res[i]))
            out[i] = mx.np.expand_dims(out[i], axis=0)
        out = mx.np.concatenate(out)
        out = mx.np.reshape(out, ((self.length, -1)))
        out = self.fc(out)
        return npx.softmax(out)

class FP32MultipleInputs(mx.gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(FP32MultipleInputs, self).__init__(**kwargs)
        self.conv1 = mx.gluon.nn.Conv2D(channels=64, kernel_size=(1,1), use_bias=False)
        self.bn1 = mx.gluon.nn.BatchNorm()
        self.conv2 = mx.gluon.nn.Conv2D(channels=64, kernel_size=(1,1), use_bias=False)
        self.bn2 = mx.gluon.nn.BatchNorm()

    def forward(self, data0, data1):
        out0 = self.conv1(data0)
        out0 = self.bn1(out0)
        out1 = self.conv2(data1)
        out1 = self.bn2(out1)
        return out1 + out0

@use_np
@xfail_when_nonstandard_decimal_separator
def test_quantize_model():
    def check_params(params, qparams, qsym=None):
        if qsym is None:
            assert len(params) == len(qparams)
            for k, v in params.items():
                assert k in qparams
                assert same(v.asnumpy(), qparams[k].asnumpy())
        else:
            qparams_ground_truth = mx.contrib.quant._quantize_params(qsym, params, min_max_dict = {})
            assert len(qparams) == len(qparams_ground_truth)
            for k, v in qparams_ground_truth.items():
                assert k in qparams
                assert same(v.asnumpy(), qparams[k].asnumpy())

    def check_qsym_calibrated(qsym):
        attrs = qsym.attr_dict()
        for k, v in attrs.items():
            if k.find('requantize_') != -1:
                assert 'min_calib_range' in v
                assert 'max_calib_range' in v

    def check_qsym_qdtype(qsym, qdtype):
        attrs = qsym.attr_dict()
        for k, v in attrs.items():
            if k.find('_quantize') != -1:
                assert 'out_type' in v
                assert v['out_type'] == qdtype

    def skip_not_supported():
        if is_test_for_native_cpu():
            print('skipped testing quantize_model for native cpu since it is not supported yet')
            return True
        elif qdtype == 'int8' and is_test_for_dnnl():
            print('skipped testing quantize_model for oneDNN cpu int8 since it is not supported yet')
            return True
        elif qdtype == 'uint8' and is_test_for_gpu():
            print('skipped testing quantize_model for gpu uint8 since it is not supported yet')
            return True
        return False

    def check_quantize_model(qdtype):
        if is_test_for_native_cpu():
            print('skipped testing quantize_model for native cpu since it is not supported yet')
            return
        elif qdtype == 'int8' and is_test_for_dnnl():
            print('skipped testing quantize_model for oneDNN cpu int8 since it is not supported yet')
            return
        elif qdtype == 'uint8' and is_test_for_gpu():
            print('skipped testing quantize_model for gpu uint8 since it is not supported yet')
            return

        standard_net = FP32Net()
        standard_net.initialize()
        batch_size = 4
        data_shape = (batch_size, 4, 10, 10)

        length = batch_size  # specify num of outputs from split op
        multi_out_net = FP32MultipleOutputs(length)
        multi_out_net.initialize()
        multi_out_data_shape = (length, 4, 4, 10, 10)

        for net, dshape in zip((standard_net, multi_out_net), (data_shape, multi_out_data_shape)):
            data = mx.np.random.uniform(low=0, high=1, size=dshape)
            net.hybridize()
            net(data)
            sym, _ = net.export(None)
            arg_params, aux_params = collect_block_args_aux(net, sym)

            qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym,
                                                                             arg_params=arg_params,
                                                                             aux_params=aux_params,
                                                                             device=mx.current_device(),
                                                                             quantized_dtype=qdtype,
                                                                             calib_mode='none',
                                                                             quantize_mode='full')
            check_params(arg_params, qarg_params, qsym)
            check_params(aux_params, qaux_params)

            calib_data = mx.np.random.uniform(size=dshape)
            calib_data = mx.gluon.data.DataLoader(calib_data, batch_size=batch_size)
            qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym,
                                                                             arg_params=arg_params,
                                                                             aux_params=aux_params,
                                                                             device=mx.current_device(),
                                                                             quantized_dtype=qdtype,
                                                                             calib_mode='naive',
                                                                             calib_data=calib_data,
                                                                             num_calib_batches=1,
                                                                             quantize_mode='full')
            check_params(arg_params, qarg_params, qsym)
            check_params(aux_params, qaux_params)
            check_qsym_calibrated(qsym)
            check_qsym_qdtype(qsym, qdtype)

    def check_quantize_model_multiple_inputs(qdtype):
        if skip_not_supported():
            return

        net = FP32MultipleInputs()
        net.initialize()
        net.hybridize()
        dshape = (64, 4, 10, 10)
        data = [mx.np.random.uniform(low=0, high=1, size=dshape),
                mx.np.random.uniform(low=0, high=1, size=dshape)]
        net(*data)
        sym, _ = net.export(None)
        arg_params, aux_params = collect_block_args_aux(net, sym)

        qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym,
                                                                         arg_params=arg_params,
                                                                         aux_params=aux_params,
                                                                         device=mx.current_device(),
                                                                         quantized_dtype=qdtype,
                                                                         calib_mode='none',
                                                                         quantize_mode='full')
        check_params(arg_params, qarg_params, qsym)
        check_params(aux_params, qaux_params)

        calib_data = [mx.np.random.uniform(size=dshape),
                      mx.np.random.uniform(size=dshape)]
        calib_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(*calib_data), batch_size=4)
        qsym, qarg_params, qaux_params = mx.contrib.quant.quantize_model(sym=sym,
                                                                         arg_params=arg_params,
                                                                         aux_params=aux_params,
                                                                         device=mx.current_device(),
                                                                         quantized_dtype=qdtype,
                                                                         calib_mode='naive',
                                                                         calib_data=calib_data,
                                                                         data_names=["data0","data1"],
                                                                         num_calib_batches=1,
                                                                         quantize_mode='full')
        check_params(arg_params, qarg_params, qsym)
        check_params(aux_params, qaux_params)
        check_qsym_calibrated(qsym)
        check_qsym_qdtype(qsym, qdtype)


    for qdtype in ['int8', 'uint8']:
        check_quantize_model(qdtype)
        check_quantize_model_multiple_inputs(qdtype)


@mx.util.use_np
def test_quantize_gluon_with_forward():
    def check_quantize_net(qdtype):
        if is_test_for_native_cpu():
            print('skipped testing test_quantize_model_with_forward for native cpu since it is not supported yet')
            return
        elif is_test_for_gpu():
            print('skipped testing test_quantize_model_with_forward for gpu uint8 since it is not supported yet')
            return

        data_shape = (32, 3, 224, 224)
        batch_size = 1
        resnet18_v1 = vision.resnet18_v1(pretrained=True)
        resnet18_v1.reset_device(mx.current_device())
        excluded_names_match = []
        if mx.current_device() == mx.gpu():
            excluded_names_match += ['activation', 'relu', 'conv0']
        num_calib_batches = 1

        random_data = mx.np.random.uniform(size=data_shape)
        calib_data = mx.gluon.data.DataLoader(random_data, batch_size=batch_size)

        quantized_resnet18_v1 = mx.contrib.quant.quantize_net(resnet18_v1, quantized_dtype=qdtype,
                                                              exclude_layers=None,
                                                              exclude_layers_match=excluded_names_match,
                                                              calib_mode='none',
                                                              data_shapes=[data_shape],
                                                              device=mx.current_device())
        quantized_resnet18_v1.hybridize(static_alloc=True, static_shape=True)
        quantized_resnet18_v1(random_data)

        for mode in ['naive', 'entropy']:
            for quantize_granularity in ['tensor-wise', 'channel-wise']:
                qdtype = qdtype if mode == 'naive' else 'auto'
                quantized_resnet18_v1 = mx.contrib.quant.quantize_net(resnet18_v1, quantized_dtype=qdtype,
                                                                    exclude_layers=None,
                                                                    exclude_layers_match=excluded_names_match,
                                                                    calib_data=calib_data,
                                                                    calib_mode=mode,
                                                                    quantize_granularity=quantize_granularity,
                                                                    num_calib_batches=num_calib_batches,
                                                                    device=mx.current_device())

                quantized_resnet18_v1.hybridize(static_alloc=True, static_shape=True)
                quantized_resnet18_v1(random_data)

    for qdtype in ['int8', 'uint8']:
        check_quantize_net(qdtype)


@xfail_when_nonstandard_decimal_separator
def test_quantize_sym_with_calib():
    if is_test_for_native_cpu():
        print('skipped testing quantized_pooling for native cpu since it is not supported yet')
        return

    def get_fp32_sym():
        data = mx.sym.Variable('data')
        conv = mx.sym.Convolution(data, kernel=(1, 1), num_filter=16, name='conv')
        bn = mx.sym.BatchNorm(data=conv, eps=2e-05, fix_gamma=False, momentum=0.9, use_global_stats=False, name='bn')
        act = mx.sym.Activation(data=bn, act_type='relu', name='relu')
        pool = mx.sym.Pooling(act, kernel=(4, 4), pool_type='avg', name='pool')
        fc = mx.sym.FullyConnected(pool, num_hidden=10, flatten=True, name='fc')
        sym = mx.sym.softmax(fc, name='softmax')
        return sym

    sym = get_fp32_sym()
    offline_params = [name for name in sym.list_arguments()
                      if not name.startswith('data') and not name.endswith('label')]
    qsym, _ = mx.contrib.quant._quantize_symbol(sym, device=mx.current_device(),
                                             offline_params=offline_params, quantize_mode='full')
    requantize_op_names = ['requantize_conv', 'requantize_fc']
    min_max_dict = {'conv_output': (onp.random.uniform(low=100.0, high=200.0), onp.random.uniform(low=100.0, high=200.0)),
                    'fc_output': (onp.random.uniform(low=100.0, high=200.0), onp.random.uniform(low=100.0, high=200.0))}
    op_name_to_th_name = {'requantize_conv': 'conv_output', 'requantize_fc': 'fc_output'}
    cqsym = mx.contrib.quant._calibrate_quantized_sym(qsym, min_max_dict)
    attr_dict = cqsym.attr_dict()
    for name in requantize_op_names:
        assert name in attr_dict
        lhs = float(attr_dict[name]['min_calib_range'])
        rhs = min_max_dict[op_name_to_th_name[name]][0]
        assert_almost_equal(onp.array([lhs]), onp.array([rhs]))
        lhs = float(attr_dict[name]['max_calib_range'])
        rhs = min_max_dict[op_name_to_th_name[name]][1]
        assert_almost_equal(onp.array([lhs]), onp.array([rhs]), rtol=1e-3, atol=1e-4)


@use_np
def test_quantization_net_with_different_data_inputs_options():
    if is_test_for_native_cpu():
        print('skipped testing test_quantization_net_with_different_data_inputs_options for native cpu since it is not supported yet')
        return
    elif is_test_for_gpu():
        print('skipped testing test_quantization_net_with_different_data_inputs_options for gpu since it is not supported yet')
        return

    net = FP32Net()
    net.initialize()

    batch_size = 32
    data_shape = (batch_size, 3, 224, 224)
    random_data = mx.np.random.uniform(size=data_shape)

    # pass data_shapes as list of tuples
    quantized_net = mx.contrib.quant.quantize_net(net,
                                                  quantized_dtype='auto',
                                                  data_shapes=[data_shape],
                                                  device=mx.current_device())
    out = quantized_net(random_data)
    out.wait_to_read()


    # pass data_shapes as list of DataDescs
    net2 = FP32Net()
    net2.initialize()
    data_desc = mx.io.DataDesc('data', data_shape)
    quantized_net2 = mx.contrib.quant.quantize_net(net2,
                                                   quantized_dtype='auto',
                                                   data_shapes=[data_desc],
                                                   device=mx.current_device())
    out2 = quantized_net2(random_data)
    out2.wait_to_read()


    # pass data as DataLoader
    net3 = FP32Net()
    net3.initialize()
    data_loader = mx.gluon.data.DataLoader(random_data, batch_size=batch_size)
    quantized_net3 = mx.contrib.quant.quantize_net(net3,
                                                   quantized_dtype='auto',
                                                   calib_data=data_loader,
                                                   device=mx.current_device())
    out3 = quantized_net3(random_data)
    out3.wait_to_read()



def test_optimal_threshold_adversarial_case():
    # The worst case for the optimal_threshold function is when the values are concentrated
    # at one edge: [0, 0, ..., 1000]. (histogram)
    # We want to make sure that the optimal threshold in this case is the max.
    hist = []
    hist_edges = []
    min_val = -2
    max_val = 2
    for _ in range(0, 998):
        hist.append(0)
    for i in range(0, 999):
        hist_edges.append((max_val - min_val) / 999 * i + min_val)
    hist.append(1000)
    hist_edges.append(max_val)
    hist_data = (hist, hist_edges, min_val, max_val, max_val)
    for dtype in ['uint8', 'int8', 'auto']:
        res = mx.contrib.quant._LayerHistogramCollector.get_optimal_threshold(hist_data, dtype, num_quantized_bins=5)
        # The threshold should be 2.
        print (res)
        assert abs(res[2] - 2) < 1e-5


def test_get_optimal_thresholds():
    # Given an ndarray with elements following a uniform distribution, the optimal threshold
    # for quantizing the ndarray should be either abs(min(nd)) or abs(max(nd)).
    def get_threshold(nd):
        min_nd = mx.nd.min(nd)
        max_nd = mx.nd.max(nd)
        return mx.nd.maximum(mx.nd.abs(min_nd), mx.nd.abs(max_nd)).asnumpy()

    for dtype in ['uint8', 'int8', 'auto']:
        nd = mx.nd.uniform(low=-10.532, high=11.3432, shape=(8, 3, 23, 23), dtype=onp.float64)
        expected_threshold = get_threshold(nd)
        arr = nd.asnumpy()
        min_range = onp.min(arr)
        max_range = onp.max(arr)
        th = max(abs(min_range), abs(max_range))
        hist, hist_edges = onp.histogram(arr, bins=8001, range=(-th, th))
        hist_dict = {'layer1' : (hist, hist_edges, min_range, max_range, th)}
        min_max_dict = mx.contrib.quant._LayerHistogramCollector.get_optimal_thresholds(hist_dict, dtype)
        assert 'layer1' in min_max_dict
        assert_almost_equal(onp.array([min_max_dict['layer1'][1]]), expected_threshold, rtol=1e-2, atol=1e-4)


@use_np
def test_rnn_quantization():
    data_low = -1
    data_high = 1
    def check_rnn_quantization(num_layers, bidirectional, seq_len, batch_size, input_dim, state_size):
        data_shape = (seq_len, batch_size, input_dim)

        rnn_fp32 = mx.gluon.rnn.LSTM(hidden_size=state_size,
                                     num_layers = num_layers,
                                     bidirectional=bidirectional)

        data = mx.np.random.uniform(low=data_low, high=data_high, size=data_shape)
        states_shape = (num_layers * 2 if bidirectional else num_layers, batch_size, state_size)
        states = [mx.np.zeros((states_shape)) for _ in range(batch_size)]

        rnn_fp32.initialize()
        rnn_fp32.hybridize()
        ref_out = rnn_fp32(data, states)

        class RNNDataLoader(mx.gluon.data.DataLoader):
            def __init__(self, data, states):
                super().__init__(mx.gluon.data.SimpleDataset([]), 1)
                self.data = data
                self.states = states

            def __iter__(self):
                return self

            def __next__(self):
                return [self.data, self.states]

            def __bool__(self):
                return bool(self.dataiter.iter_next())

        calib_data = RNNDataLoader(data, states)
        quant_rnn = mx.contrib.quant.quantize_net(rnn_fp32,
                                                  quantized_dtype='auto',
                                                  quantize_mode='full',
                                                  calib_data=calib_data,
                                                  calib_mode='naive',
                                                  num_calib_batches=1,
                                                  device=mx.current_device())
        qout = quant_rnn(data, states)

        qsym, _ = quant_rnn.export(None)
        assert qsym.tojson().find("quantized_rnn") != -1

        ref_out = [ref_out[0], ref_out[1][0], ref_out[1][1]]
        for i in range(len(qout)):
            mse = onp.mean((ref_out[i].asnumpy() - qout[i].asnumpy())**2)
            assert mse < 0.001

    check_rnn_quantization(1, False, 5, 2, 16, 16)
    check_rnn_quantization(1, True, 5, 2, 16, 16)



@use_np
def test_quantized_rnn():
    def check_quantized_rnn(num_layers, bidirectional, seq_len, batch_size, input_dim, state_size):
        ndir = 2 if bidirectional else 1
        size = ndir*state_size*4
        first_lyr_param_size = (input_dim + state_size + 2) * size
        other_lyr_param_size = (state_size * ndir + state_size + 2) * size
        full_param_size = first_lyr_param_size + (num_layers - 1) * other_lyr_param_size

        data = mx.np.random.uniform(-1, 1, (seq_len, batch_size, input_dim))
        state = mx.np.random.uniform(-1, 1, (num_layers*ndir, batch_size, state_size))
        state_cell = mx.np.random.uniform(0, 1, (num_layers*ndir, batch_size, state_size))
        params = mx.np.random.normal(0, 1, (full_param_size,))

        out = npx.rnn(data=data,
                      parameters=params,
                      mode='lstm',
                      state=state,
                      state_size=state_size,
                      state_cell=state_cell,
                      num_layers=num_layers,
                      bidirectional=bidirectional)

        data_min = mx.np.min(data)
        data_max = mx.np.max(data)
        data_scale = mx.np.array(128.0 / (data_max - data_min)).reshape((1,))
        data_shift = mx.np.array(128.0 - data_max * data_scale).reshape((1,))

        qdata = (data * data_scale + data_shift + 0.5).astype('uint8')
        qout = npx.contrib_quantized_rnn(data=qdata,
                                         parameters=params,
                                         mode='lstm',
                                         state=state,
                                         state_size=state_size,
                                         state_cell=state_cell,
                                         num_layers=num_layers,
                                         bidirectional=bidirectional,
                                         data_scale=data_scale,
                                         data_shift=data_shift)

        mse = onp.mean((out.asnumpy() - qout.asnumpy())**2)
        assert mse < 0.001

    check_quantized_rnn(1, False, 5, 2, 16, 16)
    check_quantized_rnn(1, True, 5, 2, 16, 16)
