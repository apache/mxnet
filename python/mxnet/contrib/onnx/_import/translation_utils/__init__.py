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
# pylint: disable=invalid-name,no-self-use,too-many-branches,too-few-public-methods,too-many-arguments
from __future__ import absolute_import as _abs
from ....base import string_types

def _fix_attribute_names(attrs, change_map):
    new_attr = {}
    for k in attrs.keys():
        if k in change_map:
            new_attr[change_map[k]] = attrs[k]
        else:
            new_attr[k] = attrs[k]
    return new_attr
    
def _fix_pooling(op_name, inputs, new_attr):
    """onnx pooling operator supports asymmetrical padding
    Adding pad operator before pooling in mxnet to work with onnx"""
    pool_type = 'avg' if op_name == 'AveragePool' else 'max'
    stride = new_attr.get('strides')
    kernel = new_attr.get('kernel_shape')
    padding = new_attr.get('pads')
    pad_width = (0, 0, 0, 0) + _pad_sequence_fix(padding, len(kernel))
    new_pad_op = symbol.pad(inputs[0], mode='constant', pad_width=pad_width)
    new_pooling_op = symbol.Pooling(new_pad_op, pool_type=pool_type,
                                stride=stride, kernel=kernel)
    return new_pooling_op

def _fix_slice(inputs, new_attr):
    """onnx slice provides slicing on multiple axis. Adding multiple slice_axis operator
    for multiple axes from mxnet"""
    begin = new_attr.get('begin')
    end = new_attr.get('end')
    axes = new_attr.get('axis', tuple(range(len(begin))))
    slice_op = symbol.slice_axis(inputs[0], axis=axes[0], begin=begin[0], end=end[0])
    if len(axes) > 1:
        for i, axis in enumerate(axes):
            slice_op = symbol.slice_axis(slice_op, axis=axis, begin=begin[i], end=end[i])
    return slice_op

def _fix_squeeze(inputs, new_attr):
    """
    MXNet doesnt have a squeeze operator.
    Using "split" to perform similar operation.
    "split" can be slower compared to "reshape".
     This can have performance impact.
     TODO: Remove this implementation once mxnet adds the support.
    """
    axes = new_attr.get('axis')
    op = symbol.split(inputs[0], axis=axes[0], num_outputs=1, squeeze_axis=1)
    for i in axes[1:]:
        op = symbol.split(op, axis=i-1, num_outputs=1, squeeze_axis=1)
    return op

def _fix_gemm(op_name, inputs, old_attr):
    """Using FullyConnected operator in place of linalg_gemm to perform same operation"""
    op = getAttr(symbol, op_name, None)
    alpha = float(old_attr.get('alpha', 1.0))
    beta = float(old_attr.get('beta', 1.0))
    transA = int(old_attr.get('transA', 0))
    transB = int(old_attr.get('transB', 0))
    if transA:
        inputs[0] = symbol.transpose(inputs[0], axes=(1, 0))
    if not transB:
        inputs[1] = symbol.transpose(inputs[1], axes=(1, 0))
    new_inputs = [alpha*inputs[0], inputs[1], beta*inputs[2]]
    new_attr = {'num_hidden' : self._params[inputs[2].name].shape[0]}
    return op, new_inputs, new_attr

def _fix_outputs(op, outputs):
    """A workaround to handle dropout or similar operator that have more than one out
    in ONNX.
    """
    if op == 'Dropout':
        assert len(outputs) == 2, "ONNX have two outputs for dropout layer."
        outputs = outputs[:-1]
    return outputs

def _fix_bias(op, attrs, num_inputs):
    """A workaround for 'use_bias' attribute since onnx don't provide this attribute,
    we have to check the number of inputs to decide it."""
    if op not in [symbol.Convolution, symbol.Deconvolution, symbol.FullyConnected]:
        return attrs
    if num_inputs == 3:
        attrs['no_bias'] = False
    elif num_inputs == 2:
        attrs['no_bias'] = True
    else:
        raise ValueError("Unexpected number of inputs for: {}".format(op))
    return attrs


def _fix_bias_shape(op_name, inputs, attrs):
    """A workaround to reshape bias term to (1, num_channel)."""
    if (op_name == 'Add' or op_name == 'Mul') and (int(len(self._params)) > 0) and \
            ('broadcast' in attrs and attrs['broadcast'] == 1):
        assert len(list(inputs)) == 2
        bias_name = self._renames.get(inputs[1], inputs[1])
        bias = self._params[bias_name]
        assert len(bias.shape) == 1
        # reshape to (1, n)
        bias = nd.array(bias.asnumpy().reshape((1, -1, 1, 1)))
        # broadcast_add expects shape with sym.variable
        self._nodes[bias_name] = symbol.Variable(name=bias_name, shape=bias.shape)
        self._params[bias_name] = bias


def _fix_channels(op, attrs, inputs):
    """A workaround for getting 'channels' or 'units' since onnx don't provide
    these attributes. We check the shape of weights provided to get the number.
    """
    if op not in [symbol.Convolution, symbol.Deconvolution, symbol.FullyConnected]:
        return attrs
    weight_name = self._renames[inputs[1]]
    if not weight_name in self._params:
        raise ValueError("Unable to get channels/units attr from onnx graph.")
    else:
        wshape = self._params[weight_name].shape
        assert len(wshape) >= 2, "Weights shape is invalid: {}".format(wshape)
        channels = wshape[0]
        if op in [symbol.FullyConnected]:
            attrs['num_hidden'] = channels
        else:
            attrs['num_filter'] = channels
    return attrs
