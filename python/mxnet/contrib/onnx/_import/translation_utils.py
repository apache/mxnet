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
