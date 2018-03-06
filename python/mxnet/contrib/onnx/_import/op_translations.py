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


# Arithmetic Operations
def add(op_name, attrs, inputs):
    """Adding two tensors"""
    new_attr = {}
    if 'broadcast' in attrs and attrs['broadcast'] == 1:
        return 'broadcast_add', new_attr, inputs
    return 'elemwise_add', new_attr, inputs


def absolute(op_name, attrs, inputs):
    return 'abs', attrs, inputs


def negative(op_name, attrs, inputs):
    """Negation of every element in a tensor"""
    return 'negative', attrs, inputs


# Sorting and Searching
def argmax(op_name, attrs, inputs):
    return 'argmax', attrs, inputs


def argmin(op_name, attrs, inputs):
    return 'argmin', attrs, inputs


# Rounding
def ceil(op_name, attrs, inputs):
    """ Calculate ceil value for input """
    return 'ceil', attrs, inputs


# Joining and spliting
def concat(op_name, attrs, inputs):
    """ Joins input arrays along a given axis. """
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axis': 'dim'})
    return 'concat', new_attrs, inputs


# Basic neural network functions
def sigmoid(op_name, attrs, inputs):
    """Computes elementwise sigmoid of the input array"""
    return 'sigmoid', attrs, inputs


def pad(op_name, attrs, inputs):
    """ Add padding to input tensor"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'pads'  : 'pad_width',
                                                               'value' : 'constant_value'
                                                              })
    new_attrs['pad_width'] = translation_utils._pad_sequence_fix(new_attrs.get('pad_width'))
    return 'pad', new_attrs, inputs


# Changing shape and type.
def reshape(op_name, attrs, inputs):
    """Reshape the given array by the shape attribute."""
    return 'reshape', attrs, inputs

def  cast(op_name, attrs, inputs):
    """ Cast input to a given dtype"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'to': 'dtype'})
    return 'cast', new_attrs, inputs

# Reduce Functions
def reduce_max(op_name, attrs, inputs):
    """Reduce the array along a given axis by maximum value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'max', new_attrs, inputs


def reduce_mean(op_name, attrs, inputs):
    """Reduce the array along a given axis by mean value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'mean', new_attrs, inputs


def avg_pooling(op_name, attrs, inputs):
    """ Average pooling"""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'kernel_shape': 'kernel',
                                                        'strides': 'stride',
                                                        'pads': 'pad',
                                                       })
    new_attrs = translation_utils._add_extra_attributes(new_attrs,
                                                        {'pool_type': 'avg',
                                                         'pooling_convention': 'valid'
                                                        })
    op = translation_utils._fix_pooling(op_name, inputs, new_attrs)
    return op, new_attrs, inputs
