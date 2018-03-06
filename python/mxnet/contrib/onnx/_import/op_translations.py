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
def _add(op_name, attrs, inputs):
    """Adding two tensors"""
    new_attr = {}
    if 'broadcast' in attrs and attrs['broadcast'] == 1:
        return 'broadcast_add', new_attr, inputs
    return 'elemwise_add', new_attr, inputs

def negative(op_name, attrs, inputs):
    """Negation of every element in a tensor"""
    return "negative", attrs, inputs

#Basic neural network functions
def sigmoid(op_name, attrs, inputs):
    """Computes elementwise sigmoid of the input array"""
    return "sigmoid", attrs, inputs
    
#Changing shape and type.
def reshape(op_name, attrs, inputs):
    """Reshape the given array by the shape attribute."""
    return "reshape", attrs, inputs

#Reduce Functions
def reduce_max(op_name, attrs, inputs):
    """Reduce the array along a given axis by maximum value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'max', new_attrs, inputs

def reduce_mean(op_name, attrs, inputs):
    """Reduce the array along a given axis by mean value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'mean', new_attrs, inputs