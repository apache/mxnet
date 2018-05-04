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
"""Utilities used for translating operators from Onnx to Mxnet."""
# pylint: disable=protected-access
from __future__ import absolute_import as _abs
from .... import symbol


def _fix_attribute_names(attrs, change_map):
    """
    Change attribute names as per values in change_map dictionary.
    Parameters
    ----------
    :param attrs : dict Dict of operator attributes
    :param change_map : dict Dict of onnx attribute name to mxnet attribute names.

    Returns
    -------
    :return new_attr : dict Converted dict of operator attributes.
    """
    new_attr = {}
    for k in attrs.keys():
        if k in change_map:
            new_attr[change_map[k]] = attrs[k]
        else:
            new_attr[k] = attrs[k]
    return new_attr

def _remove_attributes(attrs, remove_list):
    """
    Removes attributes in the remove list from the input attribute dict
    :param attrs : Dict of operator attributes
    :param remove_list : list of attributes to be removed

    :return new_attr : Dict of operator attributes without the listed attributes.
    """
    new_attrs = {}
    for attr in attrs.keys():
        if attr not in remove_list:
            new_attrs[attr] = attrs[attr]
    return new_attrs

def _add_extra_attributes(attrs, extra_attr_map):
    """
    :param attrs:  Current Attribute list
    :param extraAttrMap:  Additional attributes to be added
    :return: new_attr
    """
    for attr in extra_attr_map:
        if attr not in attrs:
            attrs[attr] = extra_attr_map[attr]
    return attrs


def _pad_sequence_fix(attr, kernel_dim=None):
    """Changing onnx's pads sequence to match with mxnet's pad_width
    mxnet: (x1_begin, x1_end, ... , xn_begin, xn_end)
    onnx: (x1_begin, x2_begin, ... , xn_end, xn_end)"""
    new_attr = ()
    if len(attr) % 2 == 0:
        for index in range(int(len(attr) / 2)):
            new_attr = new_attr + attr[index::int(len(attr) / 2)]
        # Making sure pad values  are in the attr for all axes.
        if kernel_dim is not None:
            while len(new_attr) < kernel_dim*2:
                new_attr = new_attr + (0, 0)

    return new_attr


def _fix_pooling(pool_type, inputs, new_attr):
    """onnx pooling operator supports asymmetrical padding
    Adding pad operator before pooling in mxnet to work with onnx"""
    stride = new_attr.get('stride')
    kernel = new_attr.get('kernel')
    padding = new_attr.get('pad')

    # Adding default stride.
    if stride is None:
        stride = (1,) * len(kernel)

    # Add padding attr if not provided.
    if padding is None:
        padding = (0,) * len(kernel) * 2

    # Mxnet Pad operator supports only 4D/5D tensors.
    # For 1D case, these are the steps:
    #    Step 1. Add extra dummy dimension to make it 4D. Adding to  axis = 2
    #    Step 2. Apply padding to this changed tensor
    #    Step 3. Remove the extra dimension added in step 1.
    if len(kernel) == 1:
        dummy_axis = 2
        # setting 0 padding to the new dim to be added.
        padding = (0, padding[0], 0, padding[1])
        pad_width = (0, 0, 0, 0) + _pad_sequence_fix(padding, kernel_dim=2)

        # Step 1.
        curr_sym = symbol.expand_dims(inputs[0], axis=dummy_axis)

        # Step 2. Common for all tensor sizes
        new_pad_op = symbol.pad(curr_sym, mode='edge', pad_width=pad_width)

        # Step 3: Removing extra dim added.
        new_pad_op = symbol.split(new_pad_op, axis=dummy_axis, num_outputs=1, squeeze_axis=1)
    else:
        # For 2D/3D cases:
        # Apply padding
        pad_width = (0, 0, 0, 0) + _pad_sequence_fix(padding, kernel_dim=len(kernel))
        curr_sym = inputs[0]

        if pool_type == 'max':
            # For max pool : mode = 'edge', we should replicate the
            # edge values to pad, so that we only include  input data values
            # for calculating 'max'
            new_pad_op = symbol.pad(curr_sym, mode='edge', pad_width=pad_width)
        else:
            # For avg pool, we should add 'zeros' for padding  so mode='constant'
            new_pad_op = symbol.pad(curr_sym, mode='constant', pad_width=pad_width)

    # Apply pooling without pads.
    new_pooling_op = symbol.Pooling(new_pad_op, pool_type=pool_type, stride=stride, kernel=kernel)
    return new_pooling_op

def _fix_bias(op_name, attrs, num_inputs):
    """A workaround for 'use_bias' attribute since onnx don't provide this attribute,
    we have to check the number of inputs to decide it."""
    if num_inputs == 3:
        attrs['no_bias'] = False
    elif num_inputs == 2:
        attrs['no_bias'] = True
    else:
        raise ValueError("Unexpected number of inputs for: {}".format(op_name))
    return attrs

def _fix_bias_shape(op_name, inputs, cls):
    """A workaround to reshape bias term to (1, num_channel)."""
    if int(len(cls._params)) > 0:
        assert len(list(inputs)) == 2

        op_sym = symbol.reshape(inputs[1], shape=(1, -1, 1, 1))
        if op_name == 'broadcast_add':
            op_sym = symbol.broadcast_add(op_sym, inputs[0])
        elif op_name == 'broadcast_mul':
            op_sym = symbol.broadcast_mul(op_sym, inputs[0])
    else:
        op_sym = op_name
    return op_sym


def _fix_channels(op_name, attrs, inputs, cls):
    """A workaround for getting 'channels' or 'units' since onnx don't provide
    these attributes. We check the shape of weights provided to get the number.
    """
    weight_name = inputs[1].name
    if not weight_name in cls._params:
        raise ValueError("Unable to get channels/units attr from onnx graph.")
    else:
        wshape = cls._params[weight_name].shape
        assert len(wshape) >= 2, "Weights shape is invalid: {}".format(wshape)

        if op_name == 'FullyConnected':
            attrs['num_hidden'] = wshape[0]
        else:
            if op_name == 'Convolution':
                # Weight shape for Conv and FC: (M x C x kH x kW) : M is number of
                # feature maps/hidden  and C is number of channels
                attrs['num_filter'] = wshape[0]
            elif op_name == 'Deconvolution':
                # Weight shape for DeConv : (C x M x kH x kW) : M is number of
                # feature maps/filters and C is number of channels
                attrs['num_filter'] = wshape[1]
    return attrs


def _fix_gemm(op_name, inputs, old_attr, cls):
    """Using FullyConnected operator in place of linalg_gemm to perform same operation"""
    op_sym = getattr(symbol, op_name, None)
    alpha = float(old_attr.get('alpha', 1.0))
    beta = float(old_attr.get('beta', 1.0))
    trans_a = int(old_attr.get('transA', 0))
    trans_b = int(old_attr.get('transB', 0))
    if trans_a:
        inputs[0] = symbol.transpose(inputs[0], axes=(1, 0))
    if not trans_b:
        inputs[1] = symbol.transpose(inputs[1], axes=(1, 0))
    new_inputs = [alpha*inputs[0], inputs[1], beta*inputs[2]]
    new_attr = {'num_hidden' : cls._params[inputs[2].name].shape[0]}
    return op_sym, new_attr, new_inputs
