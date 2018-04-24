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
""" Module for translating ONNX operators into Mxnet operatoes"""
# pylint: disable=unused-argument,protected-access
from . import translation_utils
from .... import symbol

# Method definitions for the callable objects mapped in the import_helper module

def identity(attrs, inputs, cls):
    """Returns the identity function of the the input."""
    return 'identity', attrs, inputs

def random_uniform(attrs, inputs, cls):
    """Draw random samples from a uniform distribtuion."""
    new_attr = translation_utils._remove_attributes(attrs, ['seed'])
    return 'random_uniform', new_attr, inputs

def random_normal(attrs, inputs, cls):
    """Draw random samples from a Gaussian distribution."""
    new_attr = translation_utils._remove_attributes(attrs, ['seed'])
    new_attr = translation_utils._fix_attribute_names(new_attr, {'mean' : 'loc'})
    return 'random_uniform', new_attr, inputs

# Arithmetic Operations
def add(attrs, inputs, cls):
    """Adding two tensors"""
    new_attr = {}
    if 'broadcast' in attrs and attrs['broadcast'] == 1:
        op_value = translation_utils._fix_bias_shape('broadcast_add', inputs, cls)
        return op_value, new_attr, inputs
    return 'elemwise_add', new_attr, inputs

def subtract(attrs, inputs, cls):
    """Subtracting two tensors"""
    new_attr = {}
    if 'broadcast' in attrs and attrs['broadcast'] == 1:
        return 'broadcast_sub', new_attr, inputs
    return 'elemwise_sub', new_attr, inputs


def multiply(attrs, inputs, cls):
    """Multiply two tensors"""
    new_attr = {}
    if 'broadcast' in attrs and attrs['broadcast'] == 1:
        op_value = translation_utils._fix_bias_shape('broadcast_mul', inputs, cls)
        return op_value, new_attr, inputs
    return 'elemwise_mul', new_attr, inputs

def divide(attrs, inputs, cls):
    """Divide two tensors"""
    new_attr = {}
    if 'broadcast' in attrs and attrs['broadcast'] == 1:
        return 'broadcast_div', new_attr, inputs
    return 'elemwise_div', new_attr, inputs

def absolute(attrs, inputs, cls):
    """Returns element-wise absolute value of the input."""
    return 'abs', attrs, inputs

def negative(attrs, inputs, cls):
    """Negation of every element in a tensor"""
    return 'negative', attrs, inputs

def add_n(attrs, inputs, cls):
    """Elementwise sum of arrays"""
    return 'add_n', attrs, inputs

# Sorting and Searching
def argmax(attrs, inputs, cls):
    """Returns indices of the maximum values along an axis"""
    return 'argmax', attrs, inputs


def argmin(attrs, inputs, cls):
    """Returns indices of the minimum values along an axis."""
    return 'argmin', attrs, inputs

def maximum(attrs, inputs, cls):
    """
    Elementwise maximum of arrays.
    MXNet maximum compares only two symbols at a time.
    ONNX can send more than two to compare.
    Breaking into multiple mxnet ops to compare two symbols at a time
    """
    if len(inputs) > 1:
        mxnet_op = symbol.maximum(inputs[0], inputs[1])
        for op_input in inputs[2:]:
            mxnet_op = symbol.maximum(mxnet_op, op_input)
    else:
        mxnet_op = inputs[0]
    return mxnet_op, attrs, inputs

def minimum(attrs, inputs, cls):
    """Elementwise minimum of arrays."""
    # MXNet minimum compares only two symbols at a time.
    # ONNX can send more than two to compare.
    # Breaking into multiple mxnet ops to compare two symbols at a time
    if len(inputs) > 1:
        mxnet_op = symbol.minimum(inputs[0], inputs[1])
        for op_input in inputs[2:]:
            mxnet_op = symbol.minimum(mxnet_op, op_input)
    else:
        mxnet_op = inputs[0]
    return mxnet_op, attrs, inputs

#Hyperbolic functions
def tanh(attrs, inputs, cls):
    """Returns the hyperbolic tangent of the input array."""
    return 'tanh', attrs, inputs

# Rounding
def ceil(attrs, inputs, cls):
    """ Calculate ceil value for input """
    return 'ceil', attrs, inputs

def floor(attrs, inputs, cls):
    """ Calculate floor value for input """
    return 'floor', attrs, inputs

# Joining and spliting
def concat(attrs, inputs, cls):
    """ Joins input arrays along a given axis. """
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axis': 'dim'})
    return 'concat', new_attrs, inputs


# Basic neural network functions
def sigmoid(attrs, inputs, cls):
    """Computes elementwise sigmoid of the input array"""
    return 'sigmoid', attrs, inputs

def relu(attrs, inputs, cls):
    """Computes rectified linear function."""
    return 'relu', attrs, inputs

def pad(attrs, inputs, cls):
    """ Add padding to input tensor"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'pads'  : 'pad_width',
                                                               'value' : 'constant_value'
                                                              })
    new_attrs['pad_width'] = translation_utils._pad_sequence_fix(new_attrs.get('pad_width'))
    return 'pad', new_attrs, inputs

def matrix_multiplication(attrs, inputs, cls):
    """Performs general matrix multiplication"""
    return 'linalg_gemm2', attrs, inputs

def batch_norm(attrs, inputs, cls):
    """Batch normalization."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'epsilon' : 'eps',
                                                               'is_test':'fix_gamma'})
    new_attrs = translation_utils._remove_attributes(new_attrs,
                                                     ['spatial', 'consumed_inputs'])
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'cudnn_off': 1})

    # in test mode "fix_gamma" should be unset.
    new_attrs['fix_gamma'] = 0 if new_attrs['fix_gamma'] == 1 else 1
    return 'BatchNorm', new_attrs, inputs


def leaky_relu(attrs, inputs, cls):
    """Leaky Relu function"""
    if 'alpha' in attrs:
        new_attrs = translation_utils._fix_attribute_names(attrs, {'alpha' : 'slope'})
    else:
        new_attrs = translation_utils._add_extra_attributes(attrs, {'slope': 0.01})
    return 'LeakyReLU', new_attrs, inputs

def _elu(attrs, inputs, cls):
    """Elu function"""
    if 'alpha' in attrs:
        new_attrs = translation_utils._fix_attribute_names(attrs, {'alpha' : 'slope'})
    else:
        new_attrs = translation_utils._add_extra_attributes(attrs, {'slope': 1.0})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'act_type': 'elu'})
    return 'LeakyReLU', new_attrs, inputs

def _prelu(attrs, inputs, cls):
    """PRelu function"""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type': 'prelu'})
    return 'LeakyReLU', new_attrs, inputs

def softmax(attrs, inputs, cls):
    """Softmax function."""
    if 'axis' not in attrs:
        attrs = translation_utils._add_extra_attributes(attrs, {'axis': 1})
    return 'softmax', attrs, inputs

def conv(attrs, inputs, cls):
    """Compute N-D convolution on (N+2)-D input."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'kernel_shape' : 'kernel',
                                                               'strides' : 'stride',
                                                               'pads': 'pad',
                                                               'dilations': 'dilate',
                                                               'group': 'num_group'})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'num_group' : 1})
    new_attrs = translation_utils._fix_bias('Convolution', new_attrs, len(inputs))

    new_attrs = translation_utils._fix_channels('Convolution', new_attrs, inputs, cls)
    kernel = new_attrs['kernel']
    stride = new_attrs['stride'] if 'stride' in new_attrs else []
    padding = new_attrs['pad'] if 'pad' in new_attrs else []
    dilations = new_attrs['dilate'] if 'dilate' in new_attrs else []
    num_filter = new_attrs['num_filter']
    num_group = new_attrs['num_group']
    no_bias = new_attrs['no_bias'] if 'no_bias' in new_attrs else 0
    bias = None if no_bias is True else inputs[2]

    # Unlike ONNX, MXNet's convolution operator does not support asymmetric padding, so we first
    # use 'Pad' operator, which supports asymmetric padding. Then use the convolution operator.
    pad_width = (0, 0, 0, 0) + translation_utils._pad_sequence_fix(padding, kernel_dim=len(kernel))
    pad_op = symbol.pad(inputs[0], mode='constant', pad_width=pad_width)

    conv_op = symbol.Convolution(pad_op, inputs[1], bias,
                                 kernel=kernel, stride=stride, dilate=dilations,
                                 num_filter=num_filter, num_group=num_group, no_bias=no_bias)

    return conv_op, new_attrs, inputs

def deconv(attrs, inputs, cls):
    """Computes transposed convolution of the input tensor."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'kernel_shape' : 'kernel',
                                                               'strides' : 'stride',
                                                               'pads': 'pad',
                                                               'dilations': 'dilate',
                                                               'group': 'num_group'})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'num_group' : 1})
    new_attrs = translation_utils._fix_bias('Deconvolution', new_attrs, len(inputs))

    new_attrs = translation_utils._fix_channels('Deconvolution', new_attrs, inputs, cls)
    kernel = new_attrs['kernel']
    stride = new_attrs['stride'] if 'stride' in new_attrs else []
    padding = new_attrs['pad'] if 'pad' in new_attrs else []
    dilations = new_attrs['dilate'] if 'dilate' in new_attrs else []
    num_filter = new_attrs['num_filter']
    num_group = new_attrs['num_group']
    no_bias = new_attrs['no_bias'] if 'no_bias' in new_attrs else False
    bias = None if no_bias is True else inputs[2]

    # Unlike ONNX, MXNet's deconvolution operator does not support asymmetric padding, so we first
    # use 'Pad' operator, which supports asymmetric padding. Then use the deconvolution operator.
    pad_width = (0, 0, 0, 0) + translation_utils._pad_sequence_fix(padding, kernel_dim=len(kernel))
    pad_op = symbol.pad(inputs[0], mode='constant', pad_width=pad_width)

    deconv_op = symbol.Deconvolution(pad_op, inputs[1], bias,
                                     kernel=kernel, stride=stride, dilate=dilations,
                                     num_filter=num_filter, num_group=num_group, no_bias=no_bias)

    return deconv_op, new_attrs, inputs

def fully_connected(attrs, inputs, cls):
    """Applies a linear transformation: Y=XWT+b."""
    new_attrs = translation_utils._remove_attributes(attrs, ['axis'])

    new_attrs = translation_utils._fix_bias('FullyConnected', new_attrs, len(inputs))

    new_attrs = translation_utils._fix_channels('FullyConnected', new_attrs, inputs, cls)

    return 'FullyConnected', new_attrs, inputs


def global_maxpooling(attrs, inputs, cls):
    """Performs max pooling on the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'global_pool': True,
                                                                'kernel': (1, 1),
                                                                'pool_type': 'max'})
    return 'Pooling', new_attrs, inputs


def global_avgpooling(attrs, inputs, cls):
    """Performs avg pooling on the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'global_pool': True,
                                                                'kernel': (1, 1),
                                                                'pool_type': 'avg'})
    return 'Pooling', new_attrs, inputs


def linalg_gemm(attrs, inputs, cls):
    """Performs general matrix multiplication and accumulation"""
    trans_a = 0
    trans_b = 0
    alpha = 1
    beta = 1
    if 'transA' in attrs:
        trans_a = attrs['transA']
    if 'transB' in attrs:
        trans_b = attrs['transB']
    if 'alpha' in attrs:
        alpha = attrs['alpha']
    if 'beta' in attrs:
        beta = attrs['beta']
    flatten_a = symbol.flatten(inputs[0])
    matmul_op = symbol.linalg_gemm2(A=flatten_a, B=inputs[1],
                                    transpose_a=trans_a, transpose_b=trans_b,
                                    alpha=alpha)
    gemm_op = symbol.broadcast_add(matmul_op, beta*inputs[2])
    new_attrs = translation_utils._fix_attribute_names(attrs, {'transA': 'transpose_a',
                                                               'transB': 'transpose_b'})
    new_attrs = translation_utils._remove_attributes(new_attrs, ['broadcast'])
    return gemm_op, new_attrs, inputs

def local_response_norm(attrs, inputs, cls):
    """Local Response Normalization."""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'bias': 'knorm',
                                                        'size' : 'nsize'})
    return 'LRN', new_attrs, inputs

def dropout(attrs, inputs, cls):
    """Dropout Regularization."""
    mode = 'training'
    if attrs['is_test'] == 0:
        mode = 'always'
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'ratio': 'p'})
    new_attrs = translation_utils._remove_attributes(new_attrs, ['is_test'])
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'mode': mode})
    return 'Dropout', new_attrs, inputs

# Changing shape and type.
def reshape(attrs, inputs, cls):
    """Reshape the given array by the shape attribute."""
    return 'reshape', attrs, inputs

def cast(attrs, inputs, cls):
    """ Cast input to a given dtype"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'to' : 'dtype'})
    new_attrs['dtype'] = new_attrs['dtype'].lower()
    return 'cast', new_attrs, inputs

def split(attrs, inputs, cls):
    """Splits an array along a particular axis into multiple sub-arrays."""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'split' : 'num_outputs'})
    return 'split', new_attrs, inputs

def _slice(attrs, inputs, cls):
    """Returns a slice of the input tensor along multiple axes."""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'axes' : 'axis',
                                                        'ends' : 'end',
                                                        'starts' : 'begin'})
    # onnx slice provides slicing on multiple axis. Adding multiple slice_axis operator
    # for multiple axes from mxnet
    begin = new_attrs.get('begin')
    end = new_attrs.get('end')
    axes = new_attrs.get('axis', tuple(range(len(begin))))
    slice_op = symbol.slice_axis(inputs[0], axis=axes[0], begin=begin[0], end=end[0])
    if len(axes) > 1:
        for i, axis in enumerate(axes):
            slice_op = symbol.slice_axis(slice_op, axis=axis, begin=begin[i], end=end[i])
    return slice_op, new_attrs, inputs

def transpose(attrs, inputs, cls):
    """Transpose the input array."""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'perm' : 'axes'})
    return 'transpose', new_attrs, inputs

def squeeze(attrs, inputs, cls):
    """Remove single-dimensional entries from the shape of a tensor."""
    # MXNet doesnt have a squeeze operator.
    # Using "split" to perform similar operation.
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'axes' : 'axis'})
    axes = new_attrs.get('axis')
    mxnet_op = symbol.split(inputs[0], axis=axes[0], num_outputs=1, squeeze_axis=1)
    for i in axes[1:]:
        mxnet_op = symbol.split(mxnet_op, axis=i-1, num_outputs=1, squeeze_axis=1)
    return mxnet_op, new_attrs, inputs


def flatten(attrs, inputs, cls):
    """Flattens the input array into a 2-D array by collapsing the higher dimensions."""
    #Mxnet does not have axis support. By default uses axis=1
    if 'axis' in attrs and attrs['axis'] != 1:
        raise RuntimeError("Flatten operator only supports axis=1")
    new_attrs = translation_utils._remove_attributes(attrs, ['axis'])
    return 'Flatten', new_attrs, inputs

#Powers
def reciprocal(attrs, inputs, cls):
    """Returns the reciprocal of the argument, element-wise."""
    return 'reciprocal', attrs, inputs

def squareroot(attrs, inputs, cls):
    """Returns element-wise square-root value of the input."""
    return 'sqrt', attrs, inputs

def power(attrs, inputs, cls):
    """Returns element-wise result of base element raised to powers from exp element."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'exponent':'exp'})
    if 'broadcast' in attrs and attrs['broadcast'] == 1:
        new_attrs = translation_utils._remove_attributes(new_attrs, ['broadcast'])
        return 'broadcast_power', new_attrs, inputs
    return 'pow', new_attrs, inputs

def exponent(attrs, inputs, cls):
    """Elementwise exponent of input array."""
    return 'exp', attrs, inputs

def _log(attrs, inputs, cls):
    """Elementwise log of input array."""
    return 'log', attrs, inputs

# Reduce Functions
def reduce_max(attrs, inputs, cls):
    """Reduce the array along a given axis by maximum value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'max', new_attrs, inputs

def reduce_mean(attrs, inputs, cls):
    """Reduce the array along a given axis by mean value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'mean', new_attrs, inputs

def reduce_min(attrs, inputs, cls):
    """Reduce the array along a given axis by mean value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'min', new_attrs, inputs

def reduce_sum(attrs, inputs, cls):
    """Reduce the array along a given axis by mean value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'sum', new_attrs, inputs

def reduce_prod(attrs, inputs, cls):
    """Reduce the array along a given axis by mean value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'prod', new_attrs, inputs

def avg_pooling(attrs, inputs, cls):
    """ Average pooling"""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'kernel_shape': 'kernel',
                                                        'strides': 'stride',
                                                        'pads': 'pad',
                                                       })
    new_attrs = translation_utils._add_extra_attributes(new_attrs,
                                                        {'pooling_convention': 'valid'
                                                        })
    new_op = translation_utils._fix_pooling('avg', inputs, new_attrs)

    return new_op, new_attrs, inputs


def max_pooling(attrs, inputs, cls):
    """ Average pooling"""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'kernel_shape': 'kernel',
                                                        'strides': 'stride',
                                                        'pads': 'pad',
                                                       })

    new_attrs = translation_utils._add_extra_attributes(new_attrs,
                                                        {'pooling_convention': 'valid'
                                                        })
    new_op = translation_utils._fix_pooling('max', inputs, new_attrs)

    return new_op, new_attrs, inputs
