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
# pylint: disable=
from __future__ import absolute_import as _abs
from .... import symbol


def _fix_attribute_names(attrs, change_map):
    """
    Change attribute names as per values in change_map dictionary.
    Parameters
    ----------
    attrs : dict
        Dict of operator attributes
    change_map : dict
        Dict of onnx attribute name to mxnet attribute names.

    Returns
    -------
    new_attr : dict
        Converted dict of operator attributes.
    """
    new_attr = {}
    for k in attrs.keys():
        if k in change_map:
            new_attr[change_map[k]] = attrs[k]
        else:
            new_attr[k] = change_map[k]

    return new_attr

def _add_extra_attributes(attrs, extraAttrMap):
    """
    :param attrs:  Current Attribute list
    :param extraAttrMap:  Additional attributes to be added
    :return: new_attr
    """

    for attr in extraAttrMap:
        attrs[attr] = extraAttrMap[attr]

    return attrs


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


def _fix_pooling(op_name, inputs, new_attr):
    """onnx pooling operator supports asymmetrical padding
    Adding pad operator before pooling in mxnet to work with onnx"""
    pool_type = 'avg' if op_name == 'AveragePool' else 'max'
    stride = new_attr.get('stride')
    kernel = new_attr.get('kernel')
    padding = new_attr.get('pad')
    pad_width = (0, 0, 0, 0) + _pad_sequence_fix(padding, len(kernel))
    new_pad_op = symbol.pad(inputs[0], mode='constant', pad_width=pad_width)
    new_pooling_op = symbol.Pooling(new_pad_op, pool_type=pool_type,
                                    stride=stride, kernel=kernel)
    return new_pooling_op
