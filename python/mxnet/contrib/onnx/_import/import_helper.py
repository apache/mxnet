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

# coding: utf-8
# pylint: disable=invalid-name
"""Operator attributes conversion"""
from .common import AttributeConverter as AttrCvt 

def _revert_caffe2_pad(attr):
    """Removing extra padding from Caffe2."""
    if len(attr) == 4:
        attr = attr[:2]
    elif len(attr) == 2:
        pass
    else:
        raise ValueError("Invalid caffe2 type padding: {}".format(attr))
    return attr

def _math_name_picker(surfix):
    def _impl(attr):
        if attr.get('broadcast', 0):
            return 'broadcast_' + surfix
        return 'elemwise_' + surfix
    return _impl

def _broadcast_constraint():
    def _broadcast_check(attrs):
        if attrs.get('axis', None):
            return False
        return True
    return _broadcast_check, "Specifying broadcast axis not allowed."

def _dimension_constraint():
    """checking dimensions for conv, deconv, pooling operators"""
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) == 2:
            return True
        return False
    return _dim_check, "Only 2d kernel supported."

def _elemwise(name):
    """converting attributes for add operator"""
    return AttrCvt(
        op_name=_math_name_picker(name),
        disables=['axis'],
        ignores=['broadcast'])

def _pooling(name):
    """converting attributes for pooling operator"""
    return AttrCvt(
        op_name='Pooling',
        transforms={
            'kernel_shape': 'kernel',
            'strides': 'stride',
            'pads': 'pad'},
        # pooling convention full to match caffe2
        extras={'pool_type': name, 'pooling_convention':'valid'},
        custom_check=_dimension_constraint())

def _conv():
    """converting attributes for convolution operator"""
    return AttrCvt(
        op_name='Convolution',
        transforms={
            'kernel_shape': 'kernel',
            'strides': 'stride',
            'dilations': ('dilate', (0, 0)),
            'pads': ('pad', (0, 0), _revert_caffe2_pad),
            'group': ('num_group', 1)},
        custom_check=_dimension_constraint())

def _conv_transpose():
    """converting attributes for deconvolution operator"""
    return AttrCvt(
        op_name='Deconvolution',
        transforms={
            'kernel_shape': 'kernel',
            'strides': 'stride',
            'dilations': ('dilate', (0, 0)),
            'pads': ('pad', (0, 0), _revert_caffe2_pad),
            'group': ('num_group', 1)},
        disables=['output_shape'],
        custom_check=_dimension_constraint())

def _batch_norm():
    """converting attributes for BatchNorm operator"""
    return AttrCvt(
        op_name='BatchNorm',
        transforms={'epsilon': 'eps'},
        extras={'cudnn_off': 1},
        ignores=['spatial', 'is_test', 'consumed_inputs'])

def _activation(name):
    """converting attributes for LeakyRelu operator"""
    return AttrCvt(
        op_name='LeakyReLU',
        transforms={
            'alpha':'slope'},
        extras={'act_type': name})

def _pad_sequence_fix(attr, kernelDim=None):
    """Changing onnx's pads sequence to match with mxnet's pad_width
    mxnet: (x1_begin, x1_end, ... , xn_begin, xn_end)
    onnx: (x1_begin, x2_begin, ... , xn_end, xn_end)"""
    new_attr = ()
    if len(attr) % 2 == 0:
        for index in range(int(len(attr) / 2)):
            new_attr = new_attr + attr[index::int(len(attr) / 2)]
        # Making sure pad values  are in the attr for all axes.
        if kernelDim is not None:
            while len(new_attr) < kernelDim*2:
                new_attr = new_attr + (0, 0)
    return new_attr

def _pad():
    """converting attributes for Pad operator"""
    return AttrCvt(
        op_name='pad',
        transforms={
            'pads': ('pad_width', (0, 0, 0, 0, 0, 0, 0, 0), _pad_sequence_fix),
            'value': 'constant_value'})

def _global_pooling(name):
    """Requires kernel attribute which is not present in onnx currently.
    So for now giving default kernel."""
    return AttrCvt(
        op_name='Pooling',
        extras={'global_pool': True,
                'kernel': (1, 1),
                'pool_type': name})

def _upsample_scale_fix(attr):
    """Scale attribute conversion from float to int"""
    return int(attr)

def _upsample_restrict_mode(attr):
    """Mxnet's current UpSampling operator doesn't work well in bilinear mode.
    New operator is coming in this PR https://github.com/apache/incubator-mxnet/pull/9688/
    Issue to track this: https://github.com/onnx/onnx-mxnet/issues/33
    For now, only nearest mode is enabled."""
    if attr.decode() != 'nearest':
        raise ValueError("Only nearest mode is supported: {}".format(attr))
    return attr.decode()

def _upsample(name):
    """converting attributes for UpSampling operator"""
    return AttrCvt(
        op_name=name,
        transforms={'height_scale': ('scale', 1, _upsample_scale_fix),
                    'mode': ('sample_type', 'nearest', _upsample_restrict_mode),
                    'width_scale': ('scale', 1, _upsample_scale_fix)})

# _convert_map defines maps of name to converter functor(callable)
_convert_map = {
    # defs/experimental
    'FC'            : AttrCvt('FullyConnected', ignores=['axis', 'axis_w']),

    # defs/generator
    'Constant': AttrCvt('identity'),
    'RandomUniform' : AttrCvt('random_uniform', ignores=['seed']),
    'RandomNormal'  : AttrCvt('random_normal', {'mean':'loc'}, ignores=['seed']),
    'RandomUniformLike' : AttrCvt('random_uniform', ignores=['seed']),
    'RandomNormalLike': AttrCvt('random_normal', {'mean':'loc'}, ignores=['seed']),

    # defs/logical

    # defs/math
    'Add'           : _elemwise('add'),
    'Sub'           : _elemwise('sub'),
    'Mul'           : _elemwise('mul'),
    'Div'           : _elemwise('div'),
    'Neg'           : AttrCvt('negative'),
    'Abs'           : AttrCvt('abs'),
    'Reciprocal'    : AttrCvt('reciprocal'),
    'Floor'         : AttrCvt('floor'),
    'Ceil'          : AttrCvt('ceil'),
    'Sqrt'          : AttrCvt('sqrt'),
    'Gemm'          : AttrCvt('linalg_gemm', {'transA':'transpose_a', 'transB':'transpose_b'},
                              ignores=['broadcast']),
    'Relu'          : AttrCvt('relu'),
    'LeakyRelu'     : AttrCvt('LeakyReLU', {'alpha': 'slope'}),
    # 'Selu'
    'Elu'           : _activation('elu'),
    'Exp'           : AttrCvt('exp'),
    'Log'           : AttrCvt('log'),
    'Tanh'          : AttrCvt('tanh'),
    'Pow'           : AttrCvt('pow', {'exponent':'exp'}),
    'Dot'           : AttrCvt('dot'),
    'MatMul'        : AttrCvt('linalg_gemm2'),
    # 'PRelu'
    'Sigmoid'       : AttrCvt('sigmoid'),
    'Max'           : AttrCvt('maximum'), #elemwise maximum
    'Min'           : AttrCvt('minimum'), #elemwise minimum
    'Sum'           : AttrCvt('add_n'), #elemwise sum
    # softmax default axis is different in onnx
    'Softmax'       : AttrCvt('softmax', extras={'axis': 1}),

    # defs/nn
    'AveragePool'   : _pooling('avg'),
    'MaxPool'       : _pooling('max'),
    'Conv'          : _conv(),
    'ConvTranspose' : _conv_transpose(),
    'GlobalAveragePool': _global_pooling('avg'),
    'GlobalMaxPool' : _global_pooling('max'),
    'BatchNormalization': _batch_norm(),
    'SpatialBN'     : _batch_norm(),
    'Dropout'       : AttrCvt('Dropout', {'ratio': 'p'}, ignores=['is_test']),
    'Flatten'       : AttrCvt('flatten'),
    'LRN'           : AttrCvt('LRN', {'bias': 'knorm', 'size' : 'nsize'}),
    # defs/reduction
    'ReduceMax'     : AttrCvt('max', {'axes': 'axis'}),
    'ReduceMin'     : AttrCvt('min', {'axes': 'axis'}),
    'ReduceSum'     : AttrCvt('sum', {'axes': 'axis'}),
    'ReduceMean'    : AttrCvt('mean', {'axes': 'axis'}),
    'ReduceProd'    : AttrCvt('prod', {'axes': 'axis'}),
    # 'ReduceLogSumExp'
    'ArgMax'        : AttrCvt('argmax'),
    'ArgMin'        : AttrCvt('argmin'),

    # defs/tensor
    'Cast'          : AttrCvt('cast', {'to': 'dtype'}),
    'Reshape'       : AttrCvt('reshape'),
    'Concat'        : AttrCvt('concat', {'axis': 'dim'}),
    'Split'         : AttrCvt('split', {'split': 'num_outputs'}),
    'Pad'           : _pad(),
    'Slice'         : AttrCvt('slice_axis', {'axes': 'axis', 'ends': 'end', 'starts': 'begin'}),
    'Transpose'     : AttrCvt('transpose', {'perm': 'axes'}),
    'Squeeze'       : AttrCvt('split', {'axes': 'axis'}),
    # 'Gather'
    'Upsample'      : _upsample('UpSampling')
}
