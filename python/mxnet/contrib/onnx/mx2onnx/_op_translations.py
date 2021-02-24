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
#
# Based on
#  https://github.com/NVIDIA/mxnet_to_onnx/blob/master/mx2onnx_converter/
# mx2onnx_converter_functions.py
#  Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
#  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# coding: utf-8
# pylint: disable=too-many-locals,no-else-return,too-many-lines
# pylint: disable=anomalous-backslash-in-string,eval-used
# pylint: disable=too-many-function-args
"""
Conversion Functions for common layers.
Add new functions here with a decorator.
"""

import re
import logging
import numpy as np
from .export_onnx import MXNetGraph as mx_op
try:
    import onnx
except ImportError:
    onnx = None


def parse_helper(attrs, attrs_name, alt_value=None):
    """Helper function to parse operator attributes in required format."""
    tuple_re = re.compile(r'\([0-9L|,| ]+\)')
    if not attrs:
        return alt_value
    attrs_str = None if attrs.get(attrs_name) is None else str(attrs.get(attrs_name))
    if attrs_str is None:
        return alt_value
    attrs_match = tuple_re.search(attrs_str)
    if attrs_match is not None:
        if attrs_match.span() == (0, len(attrs_str)):
            dims = eval(attrs_str)
            return dims
        else:
            raise AttributeError("Malformed %s dimensions: %s" % (attrs_name, str(attrs_str)))
    return alt_value

def transform_padding(pad_width):
    """Helper function to convert padding format for pad operator.
    """
    num_pad_values = len(pad_width)
    onnx_pad_width = [0]*num_pad_values

    start_index = 0
    # num_pad_values will always be multiple of 2
    end_index = int(num_pad_values/2)
    for idx in range(0, num_pad_values):
        if idx % 2 == 0:
            onnx_pad_width[start_index] = pad_width[idx]
            start_index += 1
        else:
            onnx_pad_width[end_index] = pad_width[idx]
            end_index += 1

    return onnx_pad_width


def convert_string_to_list(string_val):
    """Helper function to convert string to list.
     Used to convert shape attribute string to list format.
    """
    result_list = []

    list_string = string_val.split(',')
    for val in list_string:
        val = str(val.strip())
        val = val.replace("(", "")
        val = val.replace(")", "")
        val = val.replace("L", "")
        val = val.replace("[", "")
        val = val.replace("]", "")
        if val == "None":
            result_list.append(None)
        elif val != "":
            result_list.append(int(val))

    return result_list

def get_boolean_attribute_value(attrs, attr_name):
    """ Helper function to convert a string version
    of Boolean attributes to integer for ONNX.
    Takes attribute dictionary and attr_name as
    parameters.
    """
    return 1 if attrs.get(attr_name, 0) in ["True", "1"] else 0

def get_inputs(node, kwargs):
    """Helper function to get inputs"""
    name = node["name"]
    outputs_lookup = kwargs["outputs_lookup"]
    inputs = node["inputs"]
    attrs = node.get("attrs", {})

    input_nodes = []
    for ip in inputs:
        input_node_name = outputs_lookup[ip[0]][ip[1]]
        input_nodes.append(input_node_name)

    return name, input_nodes, attrs

def create_basic_op_node(op_name, node, kwargs):
    """Helper function to create a basic operator
    node that doesn't contain op specific attrs"""
    name, input_nodes, _ = get_inputs(node, kwargs)

    node = onnx.helper.make_node(
        op_name,
        input_nodes,
        [name],
        name=name
    )
    return [node]

def create_const_scalar_node(input_name, value, kwargs):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    from onnx.helper import make_tensor
    initializer = kwargs["initializer"]
    input_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype]
    tensor_node = make_tensor(input_name, input_type, (), ([value]))
    initializer.append(tensor_node)

def create_const_node(input_name, value, kwargs):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    from onnx.helper import make_tensor
    initializer = kwargs["initializer"]
    input_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype]
    input_shape = value.shape
    tensor_node = make_tensor(input_name, input_type, input_shape, value)
    initializer.append(tensor_node)

def create_tensor(tensor_list, tensor_name, initializer, dtype='int64'):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    tensor_np = np.array(tensor_list, dtype=dtype)
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[tensor_np.dtype]
    dims = np.shape(tensor_np)
    if dtype == np.float16:
        tensor_np = tensor_np.view(dtype=np.uint16)
    initializer.append(
        onnx.helper.make_tensor(
            name=tensor_name,
            data_type=data_type,
            dims=dims,
            vals=tensor_np.flatten().tolist(),
            raw=False
        )
    )

@mx_op.register("null")
def convert_weights_and_inputs(node, **kwargs):
    """Helper function to convert weights and inputs.
    """
    name, _, _ = get_inputs(node, kwargs)

    if kwargs["is_input"] is False:
        weights = kwargs["weights"]
        initializer = kwargs["initializer"]
        np_arr = weights[name]
        data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np_arr.dtype]
        dims = np.shape(np_arr)

        tensor_node = onnx.helper.make_tensor_value_info(name, data_type, dims)

        initializer.append(
            onnx.helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=np_arr.flatten().tolist(),
                raw=False
            )
        )

        return [tensor_node]
    else:
        tval_node = onnx.helper.make_tensor_value_info(name, kwargs["in_type"], kwargs["in_shape"])
        return [tval_node]


@mx_op.register('Convolution')
def convert_convolution(node, **kwargs):
    """Map MXNet's convolution operator attributes to onnx's Conv operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    kernel = convert_string_to_list(attrs.get('kernel', '()'))
    stride = convert_string_to_list(attrs.get('stride', '()'))
    dilate = convert_string_to_list(attrs.get('dilate', '()'))
    pad = convert_string_to_list(attrs.get('pad', '()'))
    num_group = int(attrs.get('num_group', 1))
    no_bias = attrs.get('no_bias', 'False')
    layout = attrs.get('layout', 'NCHW')

    if layout not in ['NCHW', 'NCDHW']:
        raise NotImplementedError('Convolution currently does not support layout not in '
                                  '[\'NCHW\', \'NCDHW\']')

    if no_bias == 'True':
        assert len(input_nodes) == 2, 'Convolution takes 2 input if no_bias==True'
    else:
        assert len(input_nodes) == 3, 'Convolution takes 3 input if no_bias==False'

    kwargs_ = {}
    if kernel:
        kwargs_['kernel_shape'] = tuple(kernel)
    if pad:
        kwargs_['pads'] = tuple(pad) + tuple(pad)
    if stride:
        kwargs_['strides'] = stride
    if dilate:
        kwargs_['dilations'] = dilate

    nodes = [
        make_node('Conv', input_nodes, [name], group=num_group, **kwargs_)
    ]

    return nodes


@mx_op.register("Deconvolution")
def convert_deconvolution(node, **kwargs):
    """Map MXNet's deconvolution operator attributes to onnx's ConvTranspose operator
    and return the created node.
    """
    name, inputs, attrs = get_inputs(node, kwargs)

    kernel_dims = list(parse_helper(attrs, "kernel"))
    stride_dims = list(parse_helper(attrs, "stride", [1, 1]))
    pad_dims = list(parse_helper(attrs, "pad", [0, 0]))
    num_group = int(attrs.get("num_group", 1))
    dilations = list(parse_helper(attrs, "dilate", [1, 1]))
    adj_dims = list(parse_helper(attrs, "adj", [0, 0]))

    pad_dims = pad_dims + pad_dims

    deconv_node = onnx.helper.make_node(
        "ConvTranspose",
        inputs=inputs,
        outputs=[name],
        kernel_shape=kernel_dims,
        strides=stride_dims,
        dilations=dilations,
        output_padding=adj_dims,
        pads=pad_dims,
        group=num_group,
        name=name
    )

    return [deconv_node]


@mx_op.register("Crop")
def convert_crop(node, **kwargs):
    """Map MXNet's crop operator attributes to onnx's Crop operator
    and return the created node.
    """
    name, inputs, attrs = get_inputs(node, kwargs)
    num_inputs = len(inputs)

    y, x = list(parse_helper(attrs, "offset", [0, 0]))
    h, w = list(parse_helper(attrs, "h_w", [0, 0]))
    if num_inputs > 1:
        h, w = kwargs["out_shape"][-2:]
    border = [x, y, x + w, y + h]

    crop_node = onnx.helper.make_node(
        "Crop",
        inputs=[inputs[0]],
        outputs=[name],
        border=border,
        scale=[1, 1],
        name=name
    )

    logging.warning(
        "Using an experimental ONNX operator: Crop. " \
        "Its definition can change.")

    return [crop_node]


@mx_op.register("FullyConnected")
def convert_fully_connected(node, **kwargs):
    """Map MXNet's FullyConnected operator attributes to onnx's Gemm operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    input_type = kwargs['in_type']
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]
    flatten = get_boolean_attribute_value(attrs, 'flatten')
    no_bias = get_boolean_attribute_value(attrs, 'no_bias')
    num_hidden = int(attrs.get('num_hidden'))

    nodes = []
    if flatten:
        nodes += [
            make_node('Flatten', [input_nodes[0]], [name+'_data_flattened'])
        ]
    else:
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+'_orig_shape']),
            make_node('Shape', [name+'_orig_shape'], [name+'_dim']),
            make_node('Flatten', [input_nodes[0]], [name+'_data_flattened'], axis=-1),
        ]

    in_nodes = [name+'_data_flattened', input_nodes[1]]

    if no_bias:
        create_const_scalar_node(name+'_bias', np.int32(0).astype(dtype), kwargs)
        in_nodes.append(name+'_bias')
    else:
        in_nodes.append(input_nodes[2])

    if flatten:
        nodes += [
            make_node('Gemm', in_nodes, [name], alpha=1.0, beta=1.0, transA=0, transB=1, name=name)
        ]
    else:
        create_tensor([0], name+'_0', kwargs['initializer'])
        create_tensor([1], name+'_1', kwargs['initializer'])
        create_tensor([num_hidden], name+'_num_hidden', kwargs['initializer'])
        nodes += [
            make_node('Gemm', in_nodes, [name+'_gemm'], alpha=1.0, beta=1.0, transA=0, transB=1),
            make_node('Sub', [name+'_dim', name+'_1'], [name+'dim_minus_1']),
            make_node('Slice', [name+'_orig_shape', name+'_0', name+'dim_minus_1'],
                      [name+'_shape_sliced']),
            make_node('Concat', [name+'_shape_sliced', name+'_num_hidden'],
                      [name+'_shape_new'], axis=0),
            make_node('Reshape', [name+'_gemm', name+'_shape_new'], [name], name=name)
        ]

    return nodes


@mx_op.register("BatchNorm")
def convert_batchnorm(node, **kwargs):
    """Map MXNet's BatchNorm operator attributes to onnx's BatchNormalization operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    momentum = float(attrs.get("momentum", 0.9))
    eps = float(attrs.get("eps", 0.001))
    axis = int(attrs.get("axis", 1))

    if axis != 1:
        raise NotImplementedError("batchnorm axis != 1 is currently not supported.")

    bn_node = onnx.helper.make_node(
        "BatchNormalization",
        input_nodes,
        [name],
        name=name,
        epsilon=eps,
        momentum=momentum
        # MXNet computes mean and variance per channel for batchnorm.
        # Default for onnx is across all spatial features. Relying on default
        # ONNX behavior of spatial=1 for ONNX opset 8 and below. As the spatial
        # attribute is deprecated in opset 9 and above, not explicitly encoding it.
    )
    return [bn_node]


@mx_op.register("tanh")
def convert_tanh(node, **kwargs):
    """Map MXNet's tanh operator attributes to onnx's Tanh operator
    and return the created node.
    """
    return create_basic_op_node('Tanh', node, kwargs)

@mx_op.register("cos")
def convert_cos(node, **kwargs):
    """Map MXNet's cos operator attributes to onnx's Cos operator
    and return the created node.
    """
    return create_basic_op_node('Cos', node, kwargs)

@mx_op.register("sin")
def convert_sin(node, **kwargs):
    """Map MXNet's sin operator attributes to onnx's Sin operator
    and return the created node.
    """
    return create_basic_op_node('Sin', node, kwargs)

@mx_op.register("tan")
def convert_tan(node, **kwargs):
    """Map MXNet's tan operator attributes to onnx's tan operator
    and return the created node.
    """
    return create_basic_op_node('Tan', node, kwargs)

@mx_op.register("arccos")
def convert_acos(node, **kwargs):
    """Map MXNet's acos operator attributes to onnx's acos operator
    and return the created node.
    """
    return create_basic_op_node('Acos', node, kwargs)

@mx_op.register("arcsin")
def convert_asin(node, **kwargs):
    """Map MXNet's asin operator attributes to onnx's asin operator
    and return the created node.
    """
    return create_basic_op_node('Asin', node, kwargs)

@mx_op.register("arctan")
def convert_atan(node, **kwargs):
    """Map MXNet's atan operator attributes to onnx's atan operator
    and return the created node.
    """
    return create_basic_op_node('Atan', node, kwargs)

#Basic neural network functions
@mx_op.register("sigmoid")
def convert_sigmoid(node, **kwargs):
    """Map MXNet's sigmoid operator attributes to onnx's Sigmoid operator
    and return the created node.
    """
    return create_basic_op_node('Sigmoid', node, kwargs)

@mx_op.register("relu")
def convert_relu(node, **kwargs):
    """Map MXNet's relu operator attributes to onnx's Relu operator
    and return the created node.
    """
    return create_basic_op_node('Relu', node, kwargs)

@mx_op.register("Activation")
def convert_activation(node, **kwargs):
    """Map MXNet's Activation operator attributes to onnx's Tanh/Relu operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    act_type = attrs["act_type"]

    # Creating a dictionary here, but if this titlecase pattern
    # mxnet_name.title()
    act_types = {
        "tanh": "Tanh",
        "relu": "Relu",
        "sigmoid": "Sigmoid",
        "softrelu": "Softplus",
        "softsign": "Softsign"
    }

    act_name = act_types.get(act_type)
    if act_name:
        node = onnx.helper.make_node(
            act_name,
            input_nodes,
            [name],
            name=name
        )
    else:
        raise AttributeError(
            "Activation %s not implemented or recognized in the converter" % act_type
        )

    return [node]


@mx_op.register("Pad")
def convert_pad(node, **kwargs):
    """Map MXNet's pad operator attributes to onnx's Pad operator
    and return the created node.
    """
    from onnx.helper import make_node
    opset_version = kwargs["opset_version"]
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mxnet_pad_width = convert_string_to_list(attrs.get("pad_width"))
    onnx_pad_width = transform_padding(mxnet_pad_width)

    pad_mode = attrs.get("mode")
    pad_value = np.float32(attrs.get("constant_value", 0.0))

    if opset_version >= 11:
        # starting with opset 11, pads and constant_value are inputs instead of attributes
        create_const_node(name+"_pads", np.array(onnx_pad_width, dtype='int64'), kwargs)
        nodes = []
        if pad_mode == "constant":
            create_const_scalar_node(name+"_const", pad_value, kwargs)
            nodes += [
                make_node("Pad", [input_nodes[0], name+"_pads", name+"_const"], [name], mode=pad_mode, name=name)
            ]
        else:
            nodes += [
                make_node("Pad", [input_nodes[0], name+"_pads"], [name], mode=pad_mode, name=name)
            ]
        return nodes
    else:
        if pad_mode == "constant":
            node = onnx.helper.make_node(
                'Pad',
                inputs=input_nodes,
                outputs=[name],
                mode='constant',
                value=pad_value,
                pads=onnx_pad_width,
                name=name
            )
        else:
            node = onnx.helper.make_node(
                'Pad',
                inputs=input_nodes,
                outputs=[name],
                mode=pad_mode,
                pads=onnx_pad_width,
                name=name
            )
        return [node]


def create_helper_trans_node(node_name, input_node):
    """create extra transpose node for dot operator"""
    trans_node = onnx.helper.make_node(
        'Transpose',
        inputs=[input_node],
        outputs=[node_name],
        name=node_name
    )
    return trans_node


@mx_op.register("dot")
def convert_dot(node, **kwargs):
    """Map MXNet's dot operator attributes to onnx's
    MatMul and Transpose operators based on the values set for
    transpose_a, transpose_b attributes."""
    name, input_nodes, attrs = get_inputs(node, kwargs)

    trans_a = get_boolean_attribute_value(attrs, "transpose_a")
    trans_b = get_boolean_attribute_value(attrs, "transpose_b")

    nodes = []
    input_nodes = []
    if trans_a:
        nodes.append(create_helper_trans_node(name+"_a", input_nodes[0]))
        input_nodes.append(name+"_a")
    else:
        input_nodes.append(input_nodes[0])

    if trans_b:
        nodes.append(create_helper_trans_node(name+"_b", input_nodes[1]))
        input_nodes.append(name+"_b")
    else:
        input_nodes.append(input_nodes[1])

    nodes.appennd(onnx.helper.make_node('MatMul', input_nodes, [name], name=name))
    return nodes


@mx_op.register("_linalg_gemm2")
def convert_linalg_gemm2(node, **kwargs):
    """Map MXNet's _linalg_gemm2 operator attributes to onnx's
    MatMul and Transpose operators based on the values set for
    transpose_a, transpose_b attributes.
    Return multiple nodes created.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    # Getting the attributes and assigning default values.
    alpha = float(attrs.get("alpha", 1.0))
    trans_a = get_boolean_attribute_value(attrs, "transpose_a")
    trans_b = get_boolean_attribute_value(attrs, "transpose_b")

    op_name = "transpose" + str(kwargs["idx"])

    if alpha == 1.0 and trans_a == 0 and trans_b == 0:
        matmul_node = onnx.helper.make_node(
            'MatMul',
            inputs=input_nodes,
            outputs=[name],
            name=name
        )
        return [matmul_node]
    elif trans_a == 1 and trans_b == 0:
        op_name = "transpose" + str(kwargs["idx"])
        node_name = op_name+"_a"
        trans_a_node = onnx.helper.make_node(
            'Transpose',
            inputs=[input_nodes[0]],
            outputs=[op_name+"_a"],
            name=node_name
        )

        matmul_node = onnx.helper.make_node(
            'MatMul',
            inputs=[node_name, input_nodes[1]],
            outputs=[name],
            name=name
        )
        return [trans_a_node, matmul_node]

    elif trans_a == 0 and trans_b == 1:
        node_name = op_name + "_b"
        trans_b_node = onnx.helper.make_node(
            'Transpose',
            inputs=[input_nodes[1]],
            outputs=[op_name+"_b"],
            name=node_name
        )

        matmul_node = onnx.helper.make_node(
            'MatMul',
            inputs=[input_nodes[0], node_name],
            outputs=[name],
            name=name
        )

        return [trans_b_node, matmul_node]
    else:
        node_name_a = op_name+"_a"
        trans_a_node = onnx.helper.make_node(
            'Transpose',
            inputs=[input_nodes[0]],
            outputs=[op_name+"_a"],
            name=node_name_a
        )

        node_name_b = op_name + "_b"
        trans_b_node = onnx.helper.make_node(
            'Transpose',
            inputs=[input_nodes[1]],
            outputs=[op_name+"_b"],
            name=node_name_b
        )

        matmul_node = onnx.helper.make_node(
            'MatMul',
            inputs=input_nodes,
            outputs=[name],
            name=name
        )

        return [trans_a_node, trans_b_node, matmul_node]


@mx_op.register('Pooling')
def convert_pooling(node, **kwargs):
    """Map MXNet's Pooling operator attributes to onnx's
    MaxPool/AveragePool/GlobalMaxPool/GlobalAveragePool operators
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    kernel = convert_string_to_list(attrs.get('kernel', '()'))
    pool_type = attrs.get('pool_type', 'max')
    global_pool = attrs.get('global_pool', 'False')
    _ = attrs.get('cudnn_off', 'False')
    pooling_convention = attrs.get('pooling_convention', 'valid')
    stride = convert_string_to_list(attrs.get('stride', '()'))
    pad = convert_string_to_list(attrs.get('pad', '()'))
    p_value = int(attrs.get('p_value', '0'))
    count_include_pad = attrs.get('count_include_pad', 'True')
    layout = attrs.get('layout', 'NCHW')

    if pooling_convention == 'same':
        raise NotImplementedError('Pooling currently does not support '
                                  'pooling_convention==\'same\'')
    if pool_type == 'sum':
        raise NotImplementedError('Pooling currently does not support pool_type==\'sum\'')
    if pool_type == 'lp' and global_pool == 'False' and pooling_convention != 'valid':
        raise NotImplementedError('Pooling currently does not support '
                                  'pooling_convention!=\'valid\' when pool_type==\'lp\' and global_pool==False')

    if layout not in ['NCHW', 'NCDHW']:
        raise NotImplementedError('Pooling currently does not support layout not in '
                                  '[\'NCHW\', \'NCDHW\']')

    kwargs_ = {}
    if kernel:
        kwargs_['kernel_shape'] = tuple(kernel)
    if pad:
        kwargs_['pads'] = tuple(pad) + tuple(pad)
    if stride:
        kwargs_['strides'] = stride

    ceil_mode = 1 if pooling_convention == 'full' else 0
    count_include_pad = 1 if count_include_pad == 'True' else 0

    nodes = []
    if pool_type == 'avg' and global_pool == 'False':
        nodes += [
            make_node('AveragePool', [input_nodes[0]], [name], ceil_mode=ceil_mode,
                      count_include_pad=count_include_pad, **kwargs_)
        ]
    elif pool_type == 'max' and global_pool == 'False':
        nodes += [
            make_node('MaxPool', [input_nodes[0]], [name], ceil_mode=ceil_mode, **kwargs_)
        ]
    elif pool_type == 'lp' and global_pool == 'False':
        nodes += [
            make_node('LpPool', [input_nodes[0]], [name], p=p_value, **kwargs_)
        ]
    elif pool_type == 'avg' and global_pool == 'True':
        nodes += [
            make_node('GlobalAveragePool', [input_nodes[0]], [name])
        ]
    elif pool_type == 'max' and global_pool == 'True':
        nodes += [
            make_node('GlobalMaxPool', [input_nodes[0]], [name])
        ]
    elif pool_type == 'lp' and global_pool == 'True':
        nodes += [
            make_node('GlobalLpPool', [input_nodes[0]], [name], p=p_value)
        ]
    else:
        raise NotImplementedError('Unknown parameter values in Pooling')

    return nodes


@mx_op.register("exp")
def convert_exp(node, **kwargs):
    """Map MXNet's exp operator attributes to onnx's Exp operator
    and return the created node.
    """
    return create_basic_op_node('Exp', node, kwargs)

@mx_op.register("_copy")
def convert_copy(node, **kwargs):
    """Map MXNet's _copy operator attributes to onnx's Identity operator
    and return the created node.
    """
    return create_basic_op_node('Identity', node, kwargs)

@mx_op.register("identity")
def convert_identity(node, **kwargs):
    """Map MXNet's identity operator attributes to onnx's Identity operator
    and return the created node.
    """
    return create_basic_op_node('Identity', node, kwargs)

@mx_op.register("InstanceNorm")
def convert_instancenorm(node, **kwargs):
    """Map MXNet's InstanceNorm operator attributes to onnx's InstanceNormalization operator
    based on the input node's attributes and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    eps = float(attrs.get("eps", 0.001))

    node = onnx.helper.make_node(
        'InstanceNormalization',
        inputs=input_nodes,
        outputs=[name],
        name=name,
        epsilon=eps)

    return [node]

@mx_op.register("LeakyReLU")
def convert_leakyrelu(node, **kwargs):
    """Map MXNet's LeakyReLU operator attributes to onnx's Elu/LeakyRelu/PRelu operators
    based on the input node's attributes and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    act_type = attrs.get("act_type", "leaky")
    alpha = float(attrs.get("slope", 0.25))

    act_name = {"elu": "Elu", "leaky": "LeakyRelu", "prelu": "PRelu",
                "selu": "Selu"}

    if act_type in ("prelu", "selu"):
        node = onnx.helper.make_node(
            act_name[act_type],
            inputs=input_nodes,
            outputs=[name],
            name=name)
    elif act_type in ('gelu'):
        sqrt2 = np.float32(1.4142135623730951)
        create_const_scalar_node(name+"_sqrt2", sqrt2, kwargs)
        create_const_scalar_node(name+"_one", np.float32(1.0), kwargs)
        create_const_scalar_node(name+"_half", np.float32(0.5), kwargs)
        nodes = [
            make_node("Div", [input_nodes[0], name+"_sqrt2"], [name+"_div0_out"]),
            make_node("Erf", [name+"_div0_out"], [name+"_erf0_out"]),
            make_node("Add", [name+"_erf0_out", name+"_one"], [name+"_add0_out"]),
            make_node("Mul", [input_nodes[0], name+"_add0_out"], [name+"_mul0_out"]),
            make_node("Mul", [name+"_mul0_out", name+"_half"], [name], name=name)
        ]
        return nodes
    else:
        node = onnx.helper.make_node(
            act_name[act_type],
            inputs=input_nodes,
            outputs=[name],
            name=name,
            alpha=alpha)

    return [node]


@mx_op.register("softmax")
def convert_softmax(node, **kwargs):
    """Map MXNet's softmax operator attributes to onnx's Softmax operator
    and return the created node.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("axis", -1))
    temperature = str(attrs.get("temperature", 'None'))
    if temperature == 'None':
        temperature = 1.
    else:
        temperature = float(temperature)

    use_length = str(attrs.get("use_length", 'None'))
    input_type = kwargs["in_type"]
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]
    data = input_nodes[0]

    # use op set 11 ONNX Softmax
    if axis == -1 and temperature == 1.:
        nodes = []
        if use_length == "True":
            # magic number, this is fp16 min
            create_tensor([-65500.0], name+"_mask_val", kwargs["initializer"], dtype=dtype)
            create_tensor([1], name+"_1", kwargs["initializer"])
            create_const_scalar_node(name+"_0_s", np.int64(0), kwargs)
            create_const_scalar_node(name+"_1_s", np.int64(1), kwargs)
            nodes += [
                make_node("Shape", [data], [name+"_shape"]),
                make_node("Shape", [name+"_shape"], [name+"_dim"]),
                make_node("Sub", [name+"_dim", name+"_1"], [name+"_dim_m1"]),
                make_node("Slice", [name+"_shape", name+"_dim_m1", name+"_dim"],
                          [name+"_dim_last_"]),
                make_node("Squeeze", [name+"_dim_last_"], [name+"_dim_last"]),
                make_node("Range", [name+"_0_s", name+"_dim_last", name+"_1_s"], [name+"_range"]),
                make_node("Cast", [input_nodes[1]], [name+"_len"], to=int(TensorProto.INT64)),
                make_node("Unsqueeze", [name+"_len"], [name+"_len_unsqueezed"], axes=(-1,)),
                make_node("Less", [name+"_range", name+"_len_unsqueezed"], [name+"_less"]),
                make_node("Where", [name+'_less', data, name+"_mask_val"], [name+"_data_masked"])
            ]
            data = name+"_data_masked"

        nodes += [
            make_node("Softmax", [data], [name], axis=-1)
        ]

        return nodes

    create_tensor([temperature], name+"_tmp", kwargs["initializer"], dtype=dtype)
    nodes = [
        make_node("Div", [data, name+"_tmp"], [name+'_data']),
        make_node("Exp", [name+'_data'], [name+"_exp_out"]),
        make_node("ReduceSum", [name+"_exp_out"], [name+"_rsum_out"], axes=[axis], keepdims=1)
    ]
    if len(input_nodes) == 1:
        nodes += [
            make_node("Div", [name+"_exp_out", name+"_rsum_out"], [name], name=name)
        ]
        return nodes
    elif use_length == "True":
        length = input_nodes[1]

        create_tensor([axis], name+"_axis", kwargs["initializer"])
        create_tensor([0], name+"_0", kwargs["initializer"])
        create_tensor([1], name+"_1", kwargs["initializer"])
        create_const_scalar_node(name+'_-1_s', np.int64(-1), kwargs)
        create_const_scalar_node(name+'_0_s', np.int64(0), kwargs)
        create_const_scalar_node(name+'_1_s', np.int64(1), kwargs)
        nodes += [
            # cast data type
            make_node("Cast", [length], [name+"_length"], to=int(TensorProto.INT64)),
            make_node("Cast", [name+"_0"], [name+"_0_itype"], to=input_type),
            make_node("Cast", [name+"_1"], [name+"_1_itype"], to=input_type),
            # softmax output
            make_node("Div", [name+"_exp_out", name+"_rsum_out"], [name+"_div1_out"]),
            # update axis
            make_node("Shape", [data], [name+"_shape0_out"]),
            make_node("Shape", [name+"_shape0_out"], [name+"_in_dim"]),
            make_node("Add", [name+"_in_dim", name+"_axis"], [name+"_dim+axis"]),
            make_node("Less", [name+"_axis", name+"_0_s"], [name+"_less0_out"]),
            make_node("Where", [name+"_less0_out", name+"_dim+axis", name+"_axis"], [name+"_final_axis"]),
            # data mask
            make_node("Add", [name+"_final_axis", name+"_1_s"], [name+"_final_axis+1"]),
            make_node("Slice", [name+"_shape0_out", name+"_final_axis", name+"_final_axis+1"], [name+"_axis_dim"]),
            make_node("Squeeze", [name+"_axis_dim"], [name+"_axis_dim_s"]),
            make_node("Range", [name+"_0_s", name+"_axis_dim_s", name+"_1_s"], [name+"_range0_out"]),
            # one hot for axis
            make_node("Squeeze", [name+"_in_dim"], [name+"_in_dim_s"]),
            make_node("Range", [name+"_0_s", name+"_in_dim_s", name+"_1_s"], [name+"_range1_out"]),
            make_node("Equal", [name+"_range1_out", name+"_final_axis"], [name+"_equal_out"]),
            make_node("Cast", [name+"_equal_out"], [name+"_one_hot"], to=int(TensorProto.INT64)),
            # reshape data mask for less
            make_node("Sub", [name+"_axis_dim_s", name+"_1_s"], [name+"_sub0_out"]),
            make_node("Mul", [name+"_one_hot", name+"_sub0_out"], [name+"_mul0_out"]),
            make_node("Add", [name+"_mul0_out", name+"_1_s"], [name+"_add0_out"]),
            make_node('Reshape', [name+"_range0_out", name+"_add0_out"], [name+"_reshape0_out"]),
            # reshape length for less
            make_node("Mul", [name+"_one_hot", name+"_-1_s"], [name+"_mul1_out"]),
            make_node("Add", [name+"_mul1_out", name+"_1_s"], [name+"_add1_out"]),
            make_node("Sub", [name+"_shape0_out", name+"_1_s"], [name+"_sub1_out"]),
            make_node("Mul", [name+"_add1_out", name+"_sub1_out"], [name+"_mul2_out"]),
            make_node("Add", [name+"_mul2_out", name+"_1_s"], [name+"_add2_out"]),
            make_node('Reshape', [name+"_length", name+"_add2_out"], [name+"_reshape1_out"]),
            # mask output
            make_node("Less", [name+"_reshape0_out", name+"_reshape1_out"], [name+"_less_out"]),
            make_node("Cast", [name+"_less_out"], [name+"_mask"], to=input_type),
            make_node("Mul", [name+"_div1_out", name+"_mask"], [name+"_mul3_out"]),
            make_node("ReduceSum", [name+"_mul3_out"], [name+"_rsum1_out"], axes=[axis], keepdims=1),
            make_node("Equal", [name+"_rsum1_out", name+"_0_itype"], [name+"_equal1_out"]),
            make_node("Where", [name+"_equal1_out", name+"_1_itype", name+"_rsum1_out"], [name+"_where_out"]),
            make_node("Div", [name+"_mul3_out", name+"_where_out"], [name], name=name)
        ]
        return nodes

    else:
        raise NotImplementedError("use_length must be true when both data and length are paased in.")

# There's also mx.sym.softmax(), which doesn't do cross-entropy loss,
# just softmax for inference - hence the name convert_softmax_output.
@mx_op.register("SoftmaxOutput")
def convert_softmax_output(node, **kwargs):
    """Map MXNet's SoftmaxOutput operator attributes to onnx's Softmax operator
    and return the created node.
    """
    name = node["name"]

    input1 = kwargs["outputs_lookup"][node["inputs"][0][0]][node["inputs"][0][1]]

    softmax_node = onnx.helper.make_node(
        "Softmax",
        [input1],
        [name],
        axis=1,
        name=name
    )

    return [softmax_node]

@mx_op.register("LogisticRegressionOutput")
def convert_logistic_regression_output(node, **kwargs):
    """Map MXNet's SoftmaxOutput operator attributes to onnx's Softmax operator
    and return the created node.
    """
    name = node["name"]
    input1 = kwargs["outputs_lookup"][node["inputs"][0][0]][node["inputs"][0][1]]

    sigmoid_node = onnx.helper.make_node(
        "Sigmoid",
        [input1],
        [name],
        name=name
    )
    return [sigmoid_node]

@mx_op.register("BlockGrad")
def convert_blockgrad(node, **kwargs):
    """ Skip operator  """
    return create_basic_op_node('Identity', node, kwargs)

@mx_op.register("MakeLoss")
def convert_makeloss(node, **kwargs):
    """ Skip operator  """
    return create_basic_op_node('Identity', node, kwargs)

@mx_op.register("Concat")
def convert_concat(node, **kwargs):
    """Map MXNet's Concat operator attributes to onnx's Concat operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("dim", 1))
    concat_node = onnx.helper.make_node(
        "Concat",
        input_nodes,
        [name],
        axis=axis,
        name=name
    )
    return [concat_node]


@mx_op.register("transpose")
def convert_transpose(node, **kwargs):
    """Map MXNet's transpose operator attributes to onnx's Transpose operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axes = attrs.get("axes", ())
    if axes:
        axes = tuple(map(int, re.findall(r'\d+', axes)))

        transpose_node = onnx.helper.make_node(
            "Transpose",
            input_nodes,
            [name],
            perm=axes,
            name=name
        )
    else:
        transpose_node = onnx.helper.make_node(
            "Transpose",
            input_nodes,
            [name],
            name=name
        )

    return [transpose_node]


@mx_op.register("LRN")
def convert_lrn(node, **kwargs):
    """Map MXNet's LRN operator attributes to onnx's LRN operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    alpha = float(attrs.get("alpha", 0.0001))
    beta = float(attrs.get("beta", 0.75))
    bias = float(attrs.get("knorm", 1.0))
    size = int(attrs.get("nsize"))

    lrn_node = onnx.helper.make_node(
        "LRN",
        inputs=input_nodes,
        outputs=[name],
        name=name,
        alpha=alpha,
        beta=beta,
        bias=bias,
        size=size
    )

    return [lrn_node]


@mx_op.register("L2Normalization")
def convert_l2normalization(node, **kwargs):
    """Map MXNet's L2Normalization operator attributes to onnx's LpNormalization operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mode = attrs.get("mode", "instance")

    if mode != "channel":
        raise AttributeError("L2Normalization: ONNX currently supports channel mode only")

    l2norm_node = onnx.helper.make_node(
        "LpNormalization",
        input_nodes,
        [name],
        axis=1,  # channel only
        name=name
    )
    return [l2norm_node]


@mx_op.register("Dropout")
def convert_dropout(node, **kwargs):
    """Map MXNet's Dropout operator attributes to onnx's Dropout operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    _ = float(attrs.get("p", 0.5))
    _ = convert_string_to_list(attrs.get("axes", "None"))
    mode = attrs.get('mode', 'training')

    if mode != 'training':
        raise NotImplementedError("Dropout does not currently support mode!=\'training\'")

    nodes = [
        make_node('Identity', [input_nodes[0]], [name])
    ]

    return nodes


@mx_op.register("Flatten")
def convert_flatten(node, **kwargs):
    """Map MXNet's Flatten operator attributes to onnx's Flatten operator
    and return the created node.
    """
    return create_basic_op_node('Flatten', node, kwargs)

@mx_op.register("clip")
def convert_clip(node, **kwargs):
    """Map MXNet's Clip operator attributes to onnx's Clip operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    opset_version = kwargs["opset_version"]

    a_min = float(attrs.get('a_min', -np.inf))
    a_max = float(attrs.get('a_max', np.inf))

    if opset_version >= 11:
        # opset >= 11 requires min/max to be inputs
        create_const_scalar_node(name+"_min", np.float32(a_min), kwargs)
        create_const_scalar_node(name+"_max", np.float32(a_max), kwargs)
        nodes = [
            make_node("Clip", [input_nodes[0], name+"_min", name+"_max"], [name], name=name)
        ]
    else:
        nodes = [
            make_node("Clip", input_nodes, [name], name=name, min=a_min, max=a_max)
        ]
    return nodes


def scalar_op_helper(node, op_name, **kwargs):
    """Helper function for scalar arithmetic operations"""
    name, input_nodes, attrs = get_inputs(node, kwargs)
    from onnx import numpy_helper
    input_type = kwargs["in_type"]
    scalar_value = np.array([attrs.get("scalar", 1)],
                            dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type])

    initializer = kwargs["initializer"]
    flag = True
    # If the input value is in initializer, just multiply with scalar input
    # and create a new initializer
    for i in initializer:
        if i.name == input_nodes[0]:
            if op_name == 'Mul':
                new_initializer = numpy_helper.to_array(i) * scalar_value[0]
            elif op_name == 'Sub':
                if name.startswith("_rminusscalar"):
                    new_initializer = scalar_value[0] - numpy_helper.to_array(i)
                else:
                    new_initializer = numpy_helper.to_array(i) - scalar_value[0]
            elif op_name == 'Add':
                new_initializer = numpy_helper.to_array(i) + scalar_value[0]
            elif op_name == 'Div':
                if name.startswith("_rdivscalar"):
                    new_initializer = scalar_value[0] / numpy_helper.to_array(i)
                else:
                    new_initializer = numpy_helper.to_array(i) / scalar_value[0]
            elif op_name == 'Pow':
                new_initializer = numpy_helper.to_array(i) ** scalar_value[0]
            flag = False
            break

    # else create a new tensor of the scalar value, add it in initializer
    if flag is True:
        dims = np.shape(scalar_value)

        scalar_op_name = "scalar_op" + str(kwargs["idx"])
        tensor_node = onnx.helper.make_tensor_value_info(scalar_op_name, input_type, dims)

        initializer.append(
            onnx.helper.make_tensor(
                name=scalar_op_name,
                data_type=input_type,
                dims=dims,
                vals=scalar_value,
                raw=False,
            )
        )

        mul_node = onnx.helper.make_node(
            op_name,
            [input_nodes[0], scalar_op_name],
            [name],
            name=name
        )

        return [tensor_node, mul_node]
    else:
        data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[new_initializer.dtype]
        dims = np.shape(new_initializer)

        new_a_node = input_nodes[0] + str(kwargs["idx"])
        tensor_node = onnx.helper.make_tensor_value_info(new_a_node, data_type, dims)

        initializer.append(
            onnx.helper.make_tensor(
                name=new_a_node,
                data_type=data_type,
                dims=dims,
                vals=new_initializer,
                raw=False,
            )
        )
        return [tensor_node]

# Convert scalar value into node and pass it as input to mul_node
@mx_op.register("_mul_scalar")
def convert_mul_scalar(node, **kwargs):
    """Map MXNet's _mul_scalar operator attributes to onnx's Mul operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Mul', **kwargs)


# Convert scalar value into node and pass it as input to mul_node
@mx_op.register("_minus_scalar")
def convert_minus_scalar(node, **kwargs):
    """Map MXNet's _minus_scalar operator attributes to onnx's Minus operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Sub', **kwargs)

@mx_op.register("_rminus_scalar")
def convert_rminus_scalar(node, **kwargs):
    """Map MXNet's _rminus_scalar operator attributes to onnx's Sub operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Sub', **kwargs)

# Convert scalar value into node and pass it as input to mul_node
@mx_op.register("_plus_scalar")
def convert_add_scalar(node, **kwargs):
    """Map MXNet's _plus_scalar operator attributes to onnx's Add operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Add', **kwargs)

# Convert scalar value into node and pass it as input to mul_node
@mx_op.register("_div_scalar")
def convert_div_scalar(node, **kwargs):
    """Map MXNet's _div_scalar operator attributes to onnx's Div operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Div', **kwargs)

@mx_op.register("_rdiv_scalar")
def convert_rdiv_scalar(node, **kwargs):
    """Map MXNet's _rdiv_scalar operator attributes to onnx's Div operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Div', **kwargs)

@mx_op.register("_power_scalar")
def convert_pow_scalar(node, **kwargs):
    """Map MXNet's _pow_scalar operator attributes to onnx's Pow operator.
    Creates a new node for the input scalar value, adds it to the initializer
    and return multiple created nodes.
    """
    return scalar_op_helper(node, 'Pow', **kwargs)

# Sorting and Searching
@mx_op.register("argmax")
def convert_argmax(node, **kwargs):
    """Map MXNet's argmax operator attributes to onnx's ArgMax operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("axis"))
    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    node = onnx.helper.make_node(
        'ArgMax',
        inputs=input_nodes,
        axis=axis,
        keepdims=keepdims,
        outputs=[name],
        name=name
    )
    return [node]

@mx_op.register("argmin")
def convert_argmin(node, **kwargs):
    """Map MXNet's argmin operator attributes to onnx's ArgMin operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("axis"))
    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    node = onnx.helper.make_node(
        'ArgMin',
        inputs=input_nodes,
        axis=axis,
        keepdims=keepdims,
        outputs=[name],
        name=name
    )
    return [node]

@mx_op.register("_maximum")
def convert_maximum(node, **kwargs):
    """Map MXNet's _maximum operator attributes to onnx's Max operator
    and return the created node.
    """
    return create_basic_op_node('Max', node, kwargs)


@mx_op.register("_minimum")
def convert_minimum(node, **kwargs):
    """Map MXNet's _minimum operator attributes to onnx's Min operator
    and return the created node.
    """
    return create_basic_op_node('Min', node, kwargs)

@mx_op.register("min")
def convert_min(node, **kwargs):
    """Map MXNet's min operator attributes to onnx's ReduceMin operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = attrs.get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        node = onnx.helper.make_node(
            'ReduceMin',
            inputs=input_nodes,
            outputs=[name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )

        return [node]
    else:
        node = onnx.helper.make_node(
            'ReduceMin',
            inputs=input_nodes,
            outputs=[name],
            keepdims=keepdims,
            name=name
        )

        return [node]


@mx_op.register("max")
def convert_max(node, **kwargs):
    """Map MXNet's max operator attributes to onnx's ReduceMax operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = attrs.get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        node = onnx.helper.make_node(
            'ReduceMax',
            inputs=input_nodes,
            outputs=[name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )

        return [node]
    else:
        node = onnx.helper.make_node(
            'ReduceMax',
            inputs=input_nodes,
            outputs=[name],
            keepdims=keepdims,
            name=name
        )

        return [node]


@mx_op.register("mean")
def convert_mean(node, **kwargs):
    """Map MXNet's mean operator attributes to onnx's ReduceMean operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = attrs.get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        node = onnx.helper.make_node(
            'ReduceMean',
            inputs=input_nodes,
            outputs=[name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )

        return [node]
    else:
        node = onnx.helper.make_node(
            'ReduceMean',
            inputs=input_nodes,
            outputs=[name],
            keepdims=keepdims,
            name=name
        )

        return [node]


@mx_op.register("prod")
def convert_prod(node, **kwargs):
    """Map MXNet's prod operator attributes to onnx's ReduceProd operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = attrs.get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        node = onnx.helper.make_node(
            'ReduceProd',
            inputs=input_nodes,
            outputs=[name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )

        return [node]
    else:
        node = onnx.helper.make_node(
            'ReduceProd',
            inputs=input_nodes,
            outputs=[name],
            keepdims=keepdims,
            name=name
        )

        return [node]


# Arithmetic Operations
@mx_op.register("elemwise_add")
def convert_elementwise_add(node, **kwargs):
    """Map MXNet's elemwise_add operator attributes to onnx's Add operator
    and return the created node.
    """
    return create_basic_op_node('Add', node, kwargs)


@mx_op.register("broadcast_add")
def covert_broadcast_add(node, **kwargs):
    """Map MXNet's broadcast_add operator attributes to onnx's Add operator
    and return the created node.
    """
    return create_basic_op_node('Add', node, kwargs)


@mx_op.register("elemwise_sub")
def convert_elementwise_sub(node, **kwargs):
    """Map MXNet's elemwise_sub operator attributes to onnx's Sub operator
    and return the created node.
    """
    return create_basic_op_node('Sub', node, kwargs)

@mx_op.register("broadcast_sub")
def covert_broadcast_sub(node, **kwargs):
    """Map MXNet's broadcast_sub operator attributes to onnx's Sub operator
    and return the created node.
    """
    return create_basic_op_node('Sub', node, kwargs)

@mx_op.register("elemwise_mul")
def convert_elemwise_mul(node, **kwargs):
    """Map MXNet's elemwise_mul operator attributes to onnx's Mul operator
    and return the created node.
    """
    return create_basic_op_node('Mul', node, kwargs)

@mx_op.register("broadcast_mul")
def convert_broadcast_mul(node, **kwargs):
    """Map MXNet's broadcast_mul operator attributes to onnx's Mul operator
    and return the created node.
    """
    return create_basic_op_node('Mul', node, kwargs)

@mx_op.register("broadcast_minimum")
def convert_broadcast_min(node, **kwargs):
    """Map MXNet's broadcast_minimum operator attributes to onnx's Min operator
    and return the created node.
    """
    return create_basic_op_node('Min', node, kwargs)

@mx_op.register("elemwise_div")
def convert_elemwise_div(node, **kwargs):
    """Map MXNet's elemwise_div operator attributes to onnx's Div operator
    and return the created node.
    """
    return create_basic_op_node('Div', node, kwargs)

@mx_op.register("broadcast_div")
def convert_broadcast_div(node, **kwargs):
    """Map MXNet's broadcast_div operator attributes to onnx's Div operator
    and return the created node.
    """
    return create_basic_op_node('Div', node, kwargs)

@mx_op.register("negative")
def convert_negative(node, **kwargs):
    """Map MXNet's negative operator attributes to onnx's Neg operator
    and return the created node.
    """
    return create_basic_op_node('Neg', node, kwargs)

@mx_op.register("abs")
def convert_abs(node, **kwargs):
    """Map MXNet's abs operator attributes to onnx's Abs operator
    and return the created node.
    """
    return create_basic_op_node('Abs', node, kwargs)

@mx_op.register("add_n")
def convert_addn(node, **kwargs):
    """Map MXNet's add_n operator attributes to onnx's Sum operator
    and return the created node.
    """
    return create_basic_op_node('Sum', node, kwargs)

 # Rounding
@mx_op.register("ceil")
def convert_ceil(node, **kwargs):
    """Map MXNet's ceil operator attributes to onnx's Ceil operator
    and return the created node.
    """
    return create_basic_op_node('Ceil', node, kwargs)

@mx_op.register("floor")
def convert_floor(node, **kwargs):
    """Map MXNet's floor operator attributes to onnx's Floor operator
    and return the created node.
    """
    return create_basic_op_node('Floor', node, kwargs)


# Changing shape and type.
@mx_op.register("Reshape")
def convert_reshape(node, **kwargs):
    """Map MXNet's Reshape operator attributes to onnx's Reshape operator.
    Converts output shape attribute to output shape tensor
    and return multiple created nodes.
    """
    from onnx.helper import make_node

    name, input_nodes, attrs = get_inputs(node, kwargs)

    reverse = attrs.get('reverse', 'False')
    targ_shape = convert_string_to_list(attrs["shape"])

    # In general -2, -3, -4 in the target shape are not supoorted, but there are
    # a few special cases that we can convert to supported scenarios

    # If -2 and -3 are not used and there is no 0 to the right of -4, then we can just remove -4
    if -4 in targ_shape and -3 not in targ_shape and -2 not in targ_shape and reverse != 'True':
        if 0 not in targ_shape:
            targ_shape = [i for i in targ_shape if i != -4]
        else:
            # index of first -4
            ind_4 = targ_shape.index(-4)
            # index of last 0
            ind0 = len(targ_shape) - 1 - targ_shape[::-1].index(0)
            if ind_4 > ind0:
                targ_shape = [i for i in targ_shape if i != -4]

    if targ_shape == [-3, 0] and reverse != 'True':
        targ_shape = [-1, 0]
        reverse = 'True'

    if targ_shape == [0, 0, -3, -3] and reverse != 'True':
        nodes = [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('Split', [name+'_shape'], [name+'_dim0', name+'_dim1', name+'_dim2',
                                                 name+'_dim3', name+'_dim4', name+'_dim5'],
                      axis=0),
            make_node('Mul', [name+'_dim2', name+'_dim3'], [name+'_mul_1']),
            make_node('Mul', [name+'_dim4', name+'_dim5'], [name+'_mul_2']),
            make_node('Concat', [name+'_dim0', name+'_dim1', name+'_mul_1', name+'_mul_2'],
                      [name+'_shape_new'], axis=0),
            make_node('Reshape', [input_nodes[0], name+'_shape_new'], [name], name=name)
        ]
        return nodes

    if targ_shape == [0, -4, -1, 4, 0, 0] and reverse != 'True':
        create_tensor([4], name+'_4', kwargs['initializer'])
        nodes = [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('Split', [name+'_shape'], [name+'_dim0', name+'_dim1', name+'_dim2',
                                                 name+'_dim3'], axis=0),
            make_node('Div', [name+'_dim1', name+'_4'], [name+'_div']),
            make_node('Concat', [name+'_dim0', name+'_div', name+'_4', name+'_dim2', name+'_dim3'],
                      [name+'_shape_new'], axis=0),
            make_node('Reshape', [input_nodes[0], name+'_shape_new'], [name], name=name)
        ]
        return nodes

    if targ_shape == [0, 0, -4, 2, 2, 0, 0] and reverse != 'True':
        create_tensor([2], name+'_2', kwargs['initializer'])
        nodes = [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('Split', [name+'_shape'], [name+'_dim0', name+'_dim1', name+'_dim2',
                                                 name+'_dim3', name+'_dim4'], axis=0),
            make_node('Concat', [name+'_dim0', name+'_dim1', name+'_2', name+'_2',
                                 name+'_dim3', name+'_dim4'], [name+'_shape_new'], axis=0),
            make_node('Reshape', [input_nodes[0], name+'_shape_new'], [name], name=name)
        ]
        return nodes

    if targ_shape == [-4, 1, -1, 0, 0, 0] and reverse != 'True':
        create_tensor([1], name+'_1', kwargs['initializer'])
        create_tensor([-1], name+'_m1', kwargs['initializer'])
        nodes = [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('Split', [name+'_shape'], [name+'_dim0', name+'_dim1', name+'_dim2',
                                                 name+'_dim3'], axis=0),
            make_node('Concat', [name+'_1', name+'_m1', name+'_dim1', name+'_dim2', name+'_dim3'],
                      [name+'_shape_new'], axis=0),
            make_node('Reshape', [input_nodes[0], name+'_shape_new'], [name], name=name)
        ]
        return nodes

    if targ_shape == [-4, 1, 1000, 0, 0] and reverse != 'True':
        create_tensor([1], name+'_1', kwargs['initializer'])
        create_tensor([1000], name+'_1000', kwargs['initializer'])
        nodes = [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('Split', [name+'_shape'], [name+'_dim0', name+'_dim1', name+'_dim2'], axis=0),
            make_node('Concat', [name+'_1', name+'_1000', name+'_dim1', name+'_dim2'],
                      [name+'_shape_new'], axis=0),
            make_node('Reshape', [input_nodes[0], name+'_shape_new'], [name], name=name)
        ]
        return nodes

    not_supported_shape = [-2, -3, -4]
    for val in targ_shape:
        if val in not_supported_shape:
            raise AttributeError("Reshape: Shape value not supported in ONNX", val)

    create_tensor(targ_shape, name+'_targ_shape', kwargs['initializer'])

    nodes = []
    if reverse == 'False':
        nodes += [
            make_node('Reshape', [input_nodes[0], name+'_targ_shape'], [name], name=name)
            ]
    else:
        create_tensor([0], name+'_0', kwargs['initializer'])
        create_tensor([1], name+'_1', kwargs['initializer'])
        nodes += [
            make_node('Shape', [name+'_targ_shape'], [name+'_targ_dim']),
            make_node('Shape', [input_nodes[0]], [name+'_orig_shape']),
            make_node('Shape', [name+'_orig_shape'], [name+'_orig_dim']),
            make_node('Sub', [name+'_targ_dim', name+'_orig_dim'], [name+'_dim_diff']),
            make_node('Abs', [name+'_dim_diff'], [name+'_pad_len']),
            make_node('Less', [name+'_targ_dim', name+'_orig_dim'], [name+'_targ_less_orig']),
            make_node('Less', [name+'_orig_dim', name+'_targ_dim'], [name+'_orig_less_targ']),
            make_node('Where', [name+'_targ_less_orig', name+'_pad_len', name+'_0'],
                      [name+'_targ_pad_len']),
            make_node('Where', [name+'_orig_less_targ', name+'_pad_len', name+'_0'],
                      [name+'_orig_pad_len']),
            make_node('Concat', [name+'_targ_pad_len', name+'_0'], [name+'_targ_pads'], axis=0),
            make_node('Concat', [name+'_orig_pad_len', name+'_0'], [name+'_orig_pads'], axis=0),
            make_node('Pad', [name+'_targ_shape', name+'_targ_pads', name+'_1'],
                      [name+'_targ_shape_padded'], mode='constant'),
            make_node('Pad', [name+'_orig_shape', name+'_orig_pads', name+'_1'],
                      [name+'_orig_shape_padded'], mode='constant'),
            make_node('Equal', [name+'_targ_shape_padded', name+'_0'],
                      [name+'_targ_shape_0_mask']),
            make_node('Where', [name+'_targ_shape_0_mask', name+'_orig_shape_padded',
                                name+'_targ_shape_padded'], [name+'_targ_shape_new']),
            make_node('Shape', [name+'_targ_shape_new'], [name+'_targ_new_dim']),
            make_node('Slice', [name+'_targ_shape_new', name+'_targ_pad_len',
                                name+'_targ_new_dim'], [name+'_targ_shape_final']),
            make_node('Reshape', [input_nodes[0], name+'_targ_shape_final'], [name], name=name)
            ]

    return nodes

@mx_op.register("Cast")
def convert_cast(node, **kwargs):
    """Map MXNet's Cast operator attributes to onnx's Cast operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    dtype = attrs.get('dtype')
    to_dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
    nodes = [
        onnx.helper.make_node("Cast", input_nodes, [name], to=to_dtype, name=name)
    ]
    return nodes


@mx_op.register("slice_axis")
def convert_slice_axis(node, **kwargs):
    """Map MXNet's slice_axis operator attributes to onnx's Slice operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("axis"))
    begin = int(attrs.get("begin"))
    end = attrs.get("end", None)

    nodes = []
    create_tensor([axis], name+'_axis', kwargs["initializer"])
    create_tensor([begin], name+'_begin', kwargs["initializer"])
    if not end or end == 'None':
        # ONNX doesn't support None for ends. Since ends=None depicts
        # length of dimension, passing dimension in this case.
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+"_data_shape"])
        ]
        # corner case when end = None and axis = -1
        if axis == -1:
            create_tensor([-1], name+'_-1', kwargs["initializer"])
            nodes += [
                make_node('Shape', [name+'_data_shape'], [name+'_data_dim']),
                make_node('Add', [name+'_data_dim', name+'_-1'], [name+'_axis_max']),
                make_node('Slice', [name+'_data_shape', name+'_axis_max', name+'_data_dim'], [name+'_end']),
            ]
        else:
            create_tensor([axis+1], name+"_axis_plus_1", kwargs["initializer"])
            nodes += [
                make_node('Slice', [name+'_data_shape', name+'_axis', name+'_axis_plus_1'],
                          [name+"_end"])
            ]
    else:
        create_tensor([int(end)], name+'_end', kwargs["initializer"])

    nodes += [
        make_node('Slice', [input_nodes[0], name+'_begin', name+'_end', name+'_axis'],
                  [name], name=name)
        ]

    return nodes


@mx_op.register('SliceChannel')
def convert_slice_channel(node, **kwargs):
    """Map MXNet's SliceChannel operator attributes to onnx's Squeeze or Split
    operator based on squeeze_axis attribute
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    num_outputs = int(attrs.get('num_outputs'))
    axis = int(attrs.get('axis', 1))
    squeeze_axis = attrs.get('squeeze_axis', 'False')

    create_tensor([axis], name+'_axis', kwargs['initializer'])
    create_tensor([axis+1], name+'axis_p1', kwargs['initializer'])

    nodes = []
    if squeeze_axis in ['True', '1']:
        nodes += [
            make_node('Split', [input_nodes[0]], [name+str(i)+'_' for i in range(num_outputs)],
                      axis=axis)
        ]
        for i in range(num_outputs):
            nodes += [
                make_node('Squeeze', [name+str(i)+'_'], [name+str(i)], axes=[axis])
            ]
    else:
        nodes += [
            make_node('Split', [input_nodes[0]], [name+str(i) for i in range(num_outputs)],
                      axis=axis)
        ]

    return nodes

@mx_op.register("expand_dims")
def convert_expand_dims(node, **kwargs):
    """Map MXNet's expand_dims operator attributes to onnx's Unsqueeze operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("axis"))

    node = onnx.helper.make_node(
        "Unsqueeze",
        input_nodes,
        [name],
        axes=[axis],
        name=name,
    )
    return [node]

@mx_op.register("squeeze")
def convert_squeeze(node, **kwargs):
    """Map MXNet's squeeze operator attributes to onnx's squeeze operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = attrs.get("axis", None)
    if not axis:
        node = onnx.helper.make_node(
            "Squeeze",
            input_nodes,
            [name],
            name=name
        )
    else:
        axis = convert_string_to_list(axis)

        node = onnx.helper.make_node(
            "Squeeze",
            input_nodes,
            [name],
            axes=axis,
            name=name,
        )
    return [node]


@mx_op.register("log")
def convert_log(node, **kwargs):
    """Map MXNet's log operator attributes to onnx's Log operator
    and return the created node.
    """
    return create_basic_op_node('Log', node, kwargs)

@mx_op.register("reciprocal")
def convert_reciprocal(node, **kwargs):
    """Map MXNet's reciprocal operator attributes to onnx's Reciprocal operator
    and return the created node.
    """
    return create_basic_op_node('Reciprocal', node, kwargs)

@mx_op.register("_power")
def convert_power(node, **kwargs):
    """Map MXNet's _power operator attributes to onnx's Pow operator
    and return the created node.
    """
    return create_basic_op_node('Pow', node, kwargs)

@mx_op.register("broadcast_power")
def convert_broadcast_power(node, **kwargs):
    """Map MXNet's _power operator attributes to onnx's Pow operator
    and return the created node.
    """
    return create_basic_op_node('Pow', node, kwargs)

@mx_op.register("sqrt")
def convert_sqrt(node, **kwargs):
    """Map MXNet's sqrt operator attributes to onnx's Sqrt operator
    and return the created node.
    """
    return create_basic_op_node('Sqrt', node, kwargs)

@mx_op.register("depth_to_space")
def convert_depthtospace(node, **kwargs):
    """Map MXNet's depth_to_space operator attributes to onnx's
    DepthToSpace operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    blksize = int(attrs.get("block_size", 0))

    node = onnx.helper.make_node(
        "DepthToSpace",
        input_nodes,
        [name],
        blocksize=blksize,
        name=name,
    )
    return [node]

@mx_op.register("space_to_depth")
def convert_spacetodepth(node, **kwargs):
    """Map MXNet's space_to_depth operator attributes to onnx's
    SpaceToDepth operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    blksize = int(attrs.get("block_size", 0))

    node = onnx.helper.make_node(
        "SpaceToDepth",
        input_nodes,
        [name],
        blocksize=blksize,
        name=name,
    )
    return [node]

@mx_op.register("square")
def convert_square(node, **kwargs):
    """Map MXNet's square operator attributes to onnx's Pow operator
    and return the created node.
    """
    name, input_nodes, _ = get_inputs(node, kwargs)

    initializer = kwargs["initializer"]
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')]

    power2_name = "square_tensor" + str(kwargs["idx"])
    tensor_node = onnx.helper.make_tensor_value_info(power2_name, data_type, (1,))
    initializer.append(
        onnx.helper.make_tensor(
            name=power2_name,
            data_type=data_type,
            dims=(1,),
            vals=[2],
            raw=False,
        )
    )

    input_nodes.append(power2_name)

    node = onnx.helper.make_node(
        "Pow",
        input_nodes,
        [name],
        name=name
    )
    return [tensor_node, node]

@mx_op.register("sum")
def convert_sum(node, **kwargs):
    """Map MXNet's sum operator attributes to onnx's ReduceSum operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = attrs.get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes:
        node = onnx.helper.make_node(
            'ReduceSum',
            inputs=input_nodes,
            outputs=[name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )
    else:
        node = onnx.helper.make_node(
            'ReduceSum',
            inputs=input_nodes,
            outputs=[name],
            keepdims=keepdims,
            name=name
        )
    return [node]


@mx_op.register("shape_array")
def convert_shape(node, **kwargs):
    """Map MXNet's shape_array operator attributes to onnx's Shape operator
    and return the created node.
    """
    return create_basic_op_node('Shape', node, kwargs)


@mx_op.register("hard_sigmoid")
def convert_hardsigmoid(node, **kwargs):
    """Map MXNet's hard_sigmoid operator attributes to onnx's HardSigmoid operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    # Converting to float32
    alpha = float(attrs.get("alpha", 0.2))
    beta = float(attrs.get("beta", 0.5))

    node = onnx.helper.make_node(
        'HardSigmoid',
        input_nodes,
        [name],
        alpha=alpha,
        beta=beta,
        name=name
    )
    return [node]

@mx_op.register("broadcast_lesser")
def convert_broadcast_lesser(node, **kwargs):
    """Map MXNet's broadcast_lesser operator attributes to onnx's Less operator
    and return the created node.
    """
    return create_basic_op_node('Less', node, kwargs)

@mx_op.register("broadcast_greater")
def convert_broadcast_greater(node, **kwargs):
    """Map MXNet's broadcast_greater operator attributes to onnx's Greater operator
    and return the created node.
    """
    return create_basic_op_node('Greater', node, kwargs)

@mx_op.register("broadcast_equal")
def convert_broadcast_equal(node, **kwargs):
    """Map MXNet's broadcast_equal operator attributes to onnx's Equal operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, _ = get_inputs(node, kwargs)
    input_type = kwargs['in_type']

    nodes = [
        make_node("Equal", input_nodes, [name+"_equal"]),
        make_node("Cast", [name+"_equal"], [name], name=name, to=int(input_type))
    ]
    return nodes


@mx_op.register("broadcast_logical_and")
def convert_broadcast_logical_and(node, **kwargs):
    """Map MXNet's broadcast logical and operator attributes to onnx's Add operator
    and return the created node.
    """
    return create_basic_op_node('And', node, kwargs)


@mx_op.register("broadcast_logical_or")
def convert_broadcast_logical_or(node, **kwargs):
    """Map MXNet's broadcast logical or operator attributes to onnx's Or operator
    and return the created node.
    """
    return create_basic_op_node('Or', node, kwargs)


@mx_op.register("broadcast_logical_xor")
def convert_broadcast_logical_xor(node, **kwargs):
    """Map MXNet's broadcast logical xor operator attributes to onnx's Xor operator
    and return the created node.
    """
    return create_basic_op_node('Xor', node, kwargs)


@mx_op.register("logical_not")
def convert_logical_not(node, **kwargs):
    """Map MXNet's logical not operator attributes to onnx's Not operator
    and return the created node.
    """
    return create_basic_op_node('Not', node, kwargs)


@mx_op.register("size_array")
def convert_size(node, **kwargs):
    """Map MXNet's size_array operator attributes to onnx's Size operator
    and return the created node.
    """
    return create_basic_op_node('Size', node, kwargs)


@mx_op.register("log_softmax")
def convert_logsoftmax(node, **kwargs):
    """Map MXNet's log_softmax operator attributes to onnx's LogSoftMax operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    # Converting to int
    axis = int(attrs.get("axis", -1))
    temp = attrs.get("temperature", 'None')
    if temp != 'None':
        raise AttributeError("LogSoftMax: ONNX supports only temperature=None")

    node = onnx.helper.make_node(
        'LogSoftmax',
        input_nodes,
        [name],
        axis=axis,
        name=name
    )
    return [node]

@mx_op.register("norm")
def convert_norm(node, **kwargs):
    """Map MXNet's norm operator attributes to onnx's ReduceL1 and ReduceL2 operators
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = attrs.get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")
    ord = int(attrs.get("ord", 2))

    onnx_op_name = "ReduceL1" if ord == 1 else "ReduceL2"

    if axes:
        reduce_node = onnx.helper.make_node(
            onnx_op_name,
            input_nodes,
            [name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )
        return [reduce_node]
    else:
        reduce_node = onnx.helper.make_node(
            onnx_op_name,
            input_nodes,
            [name],
            keepdims=keepdims,
            name=name
        )
        return [reduce_node]

@mx_op.register("_sample_multinomial")
def convert_multinomial(node, **kwargs):
    """Map MXNet's multinomial operator attributes to onnx's
    Multinomial operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(attrs.get("dtype", 'int32'))]
    sample_size = convert_string_to_list(attrs.get("shape", '1'))
    if len(sample_size) < 2:
        sample_size = sample_size[-1]
    else:
        raise AttributeError("ONNX currently supports integer sample_size only")
    node = onnx.helper.make_node(
        "Multinomial",
        input_nodes,
        [name],
        dtype=dtype,
        sample_size=sample_size,
        name=name,
    )
    return [node]


@mx_op.register("_random_uniform")
def convert_random_uniform(node, **kwargs):
    """Map MXNet's random_uniform operator attributes to onnx's RandomUniform
    operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    # Converting to float32
    low = float(attrs.get("low", 0))
    high = float(attrs.get("high", 1.0))
    shape = convert_string_to_list(attrs.get('shape', '[]'))
    dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(attrs.get('dtype', 'float32'))]

    node = onnx.helper.make_node(
        'RandomUniform',
        input_nodes,
        [name],
        low=low,
        high=high,
        dtype=dtype,
        shape=shape,
        name=name
    )
    return [node]


@mx_op.register("_random_normal")
def convert_random_normal(node, **kwargs):
    """Map MXNet's random_normal operator attributes to onnx's RandomNormal
    operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    # Converting to float32
    mean = float(attrs.get("loc", 0))
    scale = float(attrs.get("scale", 1.0))
    shape = convert_string_to_list(attrs.get('shape', '[]'))
    dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(attrs.get('dtype', 'float32'))]

    node = onnx.helper.make_node(
        'RandomNormal',
        input_nodes,
        [name],
        mean=mean,
        scale=scale,
        dtype=dtype,
        shape=shape,
        name=name
    )
    return [node]


@mx_op.register("ROIPooling")
def convert_roipooling(node, **kwargs):
    """Map MXNet's ROIPooling operator attributes to onnx's MaxRoiPool
    operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    pooled_shape = convert_string_to_list(attrs.get('pooled_size'))
    scale = float(attrs.get("spatial_scale"))

    node = onnx.helper.make_node(
        'MaxRoiPool',
        input_nodes,
        [name],
        pooled_shape=pooled_shape,
        spatial_scale=scale,
        name=name
    )
    return [node]


@mx_op.register("tile")
def convert_tile(node, **kwargs):
    """Map MXNet's Tile operator attributes to onnx's Tile
    operator and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    data = input_nodes[0]
    reps = convert_string_to_list(attrs["reps"])

    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor(reps, name+'_reps', kwargs['initializer'], dtype='int64')
    create_tensor([len(reps)], name+'_reps_len', kwargs['initializer'])

    nodes = [
        make_node('Shape', [data], [name+'_data_shape']),
        make_node('Shape', [name+'_data_shape'], [name+'_data_dim']),
        make_node('Max', [name+'_data_dim', name+'_reps_len'], [name+'_max']),
        make_node('Sub', [name+'_max', name+'_data_dim'], [name+'_data_diff']),
        make_node('Concat', [name+'_data_diff', name+'_0'], [name+'_concat0_out'], axis=0),
        make_node('Pad', [name+'_data_shape', name+'_concat0_out', name+'_1'], [name+'_data_shape_pad']),
        make_node('Reshape', [data, name+'_data_shape_pad'], [name+'_data']),
        make_node('Sub', [name+'_max', name+'_reps_len'], [name+'_reps_diff']),
        make_node('Concat', [name+'_reps_diff', name+'_0'], [name+'_concat1_out'], axis=0),
        make_node('Pad', [name+'_reps', name+'_concat1_out', name+'_1'], [name+'_reps_pad']),
        make_node('Tile', [name+'_data', name+'_reps_pad'], [name], name=name),
    ]

    return nodes


@mx_op.register("broadcast_to")
def convert_broadcast_to(node, **kwargs):
    """Map MXNet's broadcast_to operator attributes to onnx's Expand
    operator and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    shape_list = convert_string_to_list(attrs["shape"])

    initializer = kwargs["initializer"]
    output_shape_np = np.array(shape_list, dtype='int64')
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[output_shape_np.dtype]
    dims = np.shape(output_shape_np)

    output_shape_name = "expand_attr_tensor" + str(kwargs["idx"])
    tensor_node = onnx.helper.make_tensor_value_info(output_shape_name, data_type, dims)

    initializer.append(
        onnx.helper.make_tensor(
            name=output_shape_name,
            data_type=data_type,
            dims=dims,
            vals=shape_list,
            raw=False,
        )
    )

    input_nodes.append(output_shape_name)
    expand_node = onnx.helper.make_node(
        "Expand",
        input_nodes,
        [name],
        name=name
    )

    return [tensor_node, expand_node]


@mx_op.register('topk')
def convert_topk(node, **kwargs):
    """Map MXNet's topk operator attributes to onnx's TopK operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')

    axis = int(attrs.get('axis', '-1'))
    k = int(attrs.get('k', '1'))
    ret_type = attrs.get('ret_typ', 'indices')
    is_ascend = attrs.get('is_ascend', 'False')
    is_ascend = is_ascend in ['1', 'True']
    dtype = attrs.get('dtype', 'float32')

    if ret_type == 'mask':
        raise NotImplementedError('topk does not currently support ret_type==\'mask\'')

    create_tensor([k], name+'_k', kwargs['initializer'])

    nodes = []

    if ret_type == 'both':
        if dtype == 'int64':
            nodes += [
                make_node('TopK', [input_nodes[0], name+'_k'], [name+'0', name+'1'], axis=axis,
                          largest=(not is_ascend), sorted=1),
            ]
        else:
            nodes += [
                make_node('TopK', [input_nodes[0], name+'_k'], [name+'0', name+'_1_i'], axis=axis,
                          largest=(not is_ascend), sorted=1),
                make_node('Cast', [name+'_1_i'], [name+'1'],
                          to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)])
            ]
    elif ret_type == 'value':
        nodes += [
            make_node('TopK', [input_nodes[0], name+'_k'], [name+'0', name+'_'], axis=axis,
                      largest=(not is_ascend), sorted=1),
        ]
    else:
        if dtype == 'int64':
            nodes += [
                make_node('TopK', [input_nodes[0], name+'_k'], [name+'_', name], axis=axis,
                          largest=(not is_ascend), sorted=1),
            ]
        else:
            nodes += [
                make_node('TopK', [input_nodes[0], name+'_k'], [name+'__', name+'_tmp'], axis=axis,
                          largest=(not is_ascend), sorted=1),
                make_node('Cast', [name+'_tmp'], [name],
                          to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)])
            ]

    return nodes


@mx_op.register("take")
def convert_take(node, **kwargs):
    """Map MXNet's Take operator attributes to onnx's Gather operator.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', 0))
    mode = str(attrs.get('mode', 'clip'))

    data = input_nodes[0]
    indices = input_nodes[1]

    nodes = [
        make_node('Cast', [indices], [name+'_indices'], to=int(TensorProto.INT64)),
    ]

    if mode == 'raise':
        nodes += [
            make_node('Gather', [data, name+'_indices'], [name], axis=axis, name=name)
        ]

        return nodes

    create_tensor([-1], name+'_-1', kwargs["initializer"])
    nodes += [
        make_node('Shape', [data], [name+'_data_shape']),
    ]

    # corner case
    if axis == -1:
        nodes += [
            make_node('Shape', [name+'_data_shape'], [name+'_data_dim']),
            make_node('Add', [name+'_data_dim', name+'_-1'], [name+'_axis_max']),
            make_node('Slice', [name+'_data_shape', name+'_axis_max', name+'_data_dim'], [name+'_slice0_out']),
        ]

    else:
        create_tensor([axis], name+'_axis', kwargs["initializer"])
        create_tensor([axis+1], name+'_axis+1', kwargs["initializer"])
        nodes += [
            make_node('Slice', [name+'_data_shape', name+'_axis', name+'_axis+1'], [name+'_slice0_out']),
        ]

    if mode == 'clip':
        create_tensor([0], name+'_0', kwargs["initializer"])
        nodes += [
            make_node('Add', [name+'_slice0_out', name+'_-1'], [name+'_max']),
            make_node('Greater', [name+'_indices', name+'_max'], [name+'_max_mask']),
            make_node('Where', [name+'_max_mask', name+'_max', name+'_indices'], [name+'_where0_out']),
            make_node('Less', [name+'_indices', name+'_0'], [name+'_min_mask']),
            make_node('Where', [name+'_min_mask', name+'_0', name+'_where0_out'], [name+'_where1_out']),
            make_node('Gather', [data, name+'_where1_out'], [name], axis=axis, name=name)
        ]

    elif mode == 'wrap':
        nodes += [
            make_node('Mod', [name+'_indices', name+'_slice0_out'], [name+'_mod0_out']),
            make_node('Gather', [data, name+'_mod0_out'], [name], axis=axis, name=name)
        ]

    else:
        raise NotImplementedError("mode must be clip, wrap or raise.")

    return nodes


@mx_op.register("LayerNorm")
def convert_layer_norm(node, **kwargs):
    """Map MXNet's LayerNorm operator attributes to onnx operators.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axes = int(attrs.get('axis', -1))
    eps = attrs.get('eps', 9.99999975e-06)

    input_type = int(kwargs['in_type'])
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]

    create_tensor([axes], name+"_axes", kwargs["initializer"])
    create_tensor([axes+1], name+"_axes+1", kwargs["initializer"])
    create_const_scalar_node(name+'_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name+'_1_s', np.int64(1), kwargs)
    create_const_scalar_node(name+"_2_s", np.int64(2).astype(dtype), kwargs)
    create_const_scalar_node(name+"_eps", np.float32(eps), kwargs)

    nodes = [
        make_node("ReduceMean", [input_nodes[0]], [name+"_rm0_out"], axes=[axes]),
        make_node("Sub", [input_nodes[0], name+"_rm0_out"], [name+"_sub0_out"]),
        make_node("Pow", [name+"_sub0_out", name+"_2_s"], [name+"_pow0_out"]),
        make_node("ReduceMean", [name+"_pow0_out"], [name+"_rm1_out"], axes=[axes]),
        make_node("Add", [name+"_rm1_out", name+"_eps"], [name+"_add0_out"]),
        make_node("Sqrt", [name+"_add0_out"], [name+"_sqrt0_out"]),
        make_node("Div", [name+"_sub0_out", name+"_sqrt0_out"], [name+"_div0_out"]),
    ]

    if axes == -1:
        nodes += [
            make_node("Mul", [name+"_div0_out", input_nodes[1]], [name+"_mul0_out"]),
            make_node("Add", [name+"_mul0_out", input_nodes[2]], [name], name=name)
        ]
    else:
        nodes += [
            make_node("Shape", [input_nodes[0]], [name+"_shape0_out"]),
            make_node("Shape", [name+"_shape0_out"], [name+"_in_dim"]),
            make_node("Squeeze", [name+"_in_dim"], [name+"_in_dim_s"]),
            make_node("Range", [name+"_0_s", name+"_in_dim_s", name+"_1_s"], [name+"_range"]),
            make_node("Equal", [name+"_range", name+"_axes"], [name+"_equal"]),
            make_node("Cast", [name+"_equal"], [name+"_one_hot"], to=int(TensorProto.INT64)),
            make_node("Slice", [name+"_shape0_out", name+"_axes", name+"_axes+1"], [name+"_slice_out"]),
            make_node("Squeeze", [name+"_slice_out"], [name+"_slice_out_s"]),
            make_node("Sub", [name+"_slice_out_s", name+"_1_s"], [name+"_sub1_out"]),
            make_node("Mul", [name+"_one_hot", name+"_sub1_out"], [name+"_mul0_out"]),
            make_node("Add", [name+"_mul0_out", name+"_1_s"], [name+"_add1_out"]),
            make_node('Reshape', [input_nodes[1], name+"_add1_out"], [name+"gamma_exp"]),
            make_node('Reshape', [input_nodes[2], name+"_add1_out"], [name+"beta_exp"]),
            make_node('Expand', [name+"gamma_exp", name+"_shape0_out"], [name+"gamma_exp1"]),
            make_node('Expand', [name+"beta_exp", name+"_shape0_out"], [name+"beta_exp1"]),
            make_node("Mul", [name+"_div0_out", name+"gamma_exp1"], [name+"_mul1_out"]),
            make_node("Add", [name+"_mul1_out", name+"beta_exp1"], [name], name=name)
        ]

    return nodes


@mx_op.register("_contrib_interleaved_matmul_selfatt_qk")
def convert_matmul_selfatt_qk(node, **kwargs):
    """Map MXNet's _contrib_interleaved_matmul_selfatt_qk operator
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    heads = int(attrs.get('heads'))

    # a, b, c, d, e are seq_len, batch_size, num_heads, 3, head_dim respectively
    create_tensor([0], name+"_0", kwargs["initializer"])
    create_tensor([1], name+"_1", kwargs["initializer"])
    create_tensor([1], name+"_1_f", kwargs["initializer"], dtype='float32')
    create_tensor([2], name+"_2", kwargs["initializer"])
    create_tensor([3], name+"_3", kwargs["initializer"])
    create_tensor([heads], name+"_c", kwargs["initializer"])
    create_tensor([3], name+"_d", kwargs["initializer"])
    nodes = [
        make_node('Shape', [input_nodes[0]], [name+"_data_shape"]),
        make_node('Slice', [name+'_data_shape', name+'_0', name+'_1'], [name+"_a"]),
        make_node('Slice', [name+'_data_shape', name+'_1', name+'_2'], [name+"_b"]),
        make_node('Slice', [name+'_data_shape', name+'_2', name+'_3'], [name+"_cde"]),
        make_node('Div', [name+'_cde', name+'_c'], [name+'_de']),
        make_node('Div', [name+'_de', name+'_d'], [name+'_e']),
        make_node('Cast', [name+'_e'], [name+'_e_f'], to=int(TensorProto.FLOAT)),
        make_node('Sqrt', [name+'_e_f'], [name+'_sqrt_e']),
        make_node('Div', [name+'_1_f', name+'_sqrt_e'], [name+'_1_over_sqrt_e']),
        make_node('Mul', [name+'_b', name+'_c'], [name+'_bc']),

        make_node("Concat", [name+'_a', name+'_b', name+'_c', name+'_d', name+'_e'], \
            [name+'_shape0'], axis=0),
        make_node("Concat", [name+'_0', name+'_0', name+'_0', name+'_0', name+'_0'], \
            [name+'_slice_start0'], axis=0),
        make_node("Concat", [name+'_a', name+'_b', name+'_c', name+'_1', name+'_e'], \
            [name+'_slice_end0'], axis=0),
        make_node("Concat", [name+'_a', name+'_b', name+'_c', name+'_e'], \
            [name+'_shape1'], axis=0),
        make_node("Concat", [name+'_bc', name+'_a', name+'_e'], \
            [name+'_shape2'], axis=0),
        make_node("Concat", [name+'_0', name+'_0', name+'_0', name+'_1', name+'_0'], \
            [name+'_slice_start1'], axis=0),
        make_node("Concat", [name+'_a', name+'_b', name+'_c', name+'_2', name+'_e'], \
            [name+'_slice_end1'], axis=0),

        make_node('Reshape', [input_nodes[0], name+'_shape0'], [name+'_reshape0_out']),
        make_node('Slice', [name+'_reshape0_out', name+'_slice_start0', name+'_slice_end0'], \
            [name+'_slice0_out']),
        make_node('Reshape', [name+'_slice0_out', name+'_shape1'], [name+'_reshape1_out']),
        make_node('Transpose', [name+'_reshape1_out'], [name+'_transpose0_out'], \
            perm=(1, 2, 0, 3)),
        make_node('Reshape', [name+'_transpose0_out', name+'_shape2'], [name+'_reshape2_out']),
        make_node('Mul', [name+'_reshape2_out', name+'_1_over_sqrt_e'], [name+'_mul0_out']),
        make_node('Slice', [name+'_reshape0_out', name+'_slice_start1', name+'_slice_end1'], \
            [name+'_slice1_out']),
        make_node('Reshape', [name+'_slice1_out', name+'_shape1'], [name+'_reshape3_out']),
        make_node('Transpose', [name+'_reshape3_out'], [name+'_transpose1_out'], \
            perm=(1, 2, 0, 3)),
        make_node('Reshape', [name+'_transpose1_out', name+'_shape2'], [name+'_reshape4_out']),
        make_node('Transpose', [name+'_reshape4_out'], [name+'_transpose2_out'], \
            perm=(0, 2, 1)),
        make_node('MatMul', [name+'_mul0_out', name+'_transpose2_out'], [name], name=name)
    ]

    return nodes

@mx_op.register("_contrib_interleaved_matmul_selfatt_valatt")
def convert_contrib_interleaved_matmul_selfatt_valatt(node, **kwargs):
    """Map MXNet's _contrib_interleaved_matmul_selfatt_valatt operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    qkv = input_nodes[0]
    att = input_nodes[1]
    num_heads = int(attrs.get('heads'))

    create_tensor([num_heads], name+"_const_num_heads", kwargs["initializer"])
    create_tensor([0], name+"_const_0", kwargs["initializer"])
    create_tensor([1], name+"_const_1", kwargs["initializer"])
    create_tensor([2], name+"_const_2", kwargs["initializer"])
    create_tensor([3], name+"_const_3", kwargs["initializer"])
    create_tensor([4], name+"_const_4", kwargs["initializer"])
    create_tensor([5], name+"_const_5", kwargs["initializer"])
    create_tensor([0, 0, num_heads, 3, -1], name+"_reshape0_shape", kwargs["initializer"])
    create_tensor([0, 0, 0, 2, 0], name+"_slice_start", kwargs["initializer"])
    create_tensor([0, 0, 0, -1], name+"_reshape1_shape", kwargs["initializer"])
    create_tensor([0, 0, -1], name+"_reshape4_shape", kwargs["initializer"])

    nodes = [
        make_node("Shape", [qkv], [name+"_shape_qkv"]),
        make_node("Slice", [name+"_shape_qkv", name+"_const_0", name+"_const_1"], [name+"_qkv_d0"]),
        make_node("Slice", [name+"_shape_qkv", name+"_const_1", name+"_const_2"], [name+"_qkv_d1"]),
        make_node("Slice", [name+"_shape_qkv", name+"_const_2", name+"_const_3"], [name+"_qkv_d2"]),
        make_node('Mul', [name+"_qkv_d1", name+'_const_num_heads'], [name+'_mul_out']),
        make_node("Reshape", [qkv, name+"_reshape0_shape"], [name+"_reshape0_output"]),
        make_node("Shape", [name+"_reshape0_output"], [name+"_shape_reshape0"]),
        make_node("Slice", [name+"_shape_reshape0", name+"_const_4", name+"_const_5"], [name+"_d4"]),
        make_node("Concat", [name+"_mul_out", name+"_qkv_d0", name+"_d4"], [name+"_reshape2_shape"], axis=0),
        make_node("Concat", [name+"_qkv_d1", name+"_const_num_heads", name+"_qkv_d0", name+"_d4"], \
            [name+"_reshape3_shape"], axis=0),
        make_node("Concat", [name+"_qkv_d0", name+"_qkv_d1", name+"_qkv_d2", name+"_const_3", name+"_d4"], \
            [name+"_slice_end"], axis=0),
        make_node("Slice", [name+"_reshape0_output", name+"_slice_start", name+"_slice_end"], [name+"_slice_output"]),
        make_node("Reshape", [name+"_slice_output", name+"_reshape1_shape"], [name+"_reshape1_output"]),
        make_node("Transpose", [name+"_reshape1_output"], [name+"_transpose0_output"], perm=[1, 2, 0, 3]),
        make_node("Reshape", [name+"_transpose0_output", name+"_reshape2_shape"], [name+"_reshape2_output"]),
        make_node("MatMul", [att, name+"_reshape2_output"], [name+"_matmul_output"]),
        make_node("Reshape", [name+"_matmul_output", name+"_reshape3_shape"], [name+"_reshape3_output"]),
        make_node("Transpose", [name+"_reshape3_output"], [name+"_transpose2_output"], perm=[2, 0, 1, 3]),
        make_node("Reshape", [name+"_transpose2_output", name+"_reshape4_shape"], [name], name=name)
    ]
    return nodes


@mx_op.register("broadcast_axis")
def convert_broadcast_axis(node, **kwargs):
    """Map MXNet's broadcast_axis
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = convert_string_to_list(attrs.get('axis', '()'))
    size = convert_string_to_list(attrs.get('size', '()'))
    assert len(axis) == len(size)

    shape_name = name+'_shape_0'

    create_tensor([0], name+'_0', kwargs["initializer"])
    create_tensor([1], name+'_1', kwargs["initializer"])
    create_const_scalar_node(name+'_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name+'_1_s', np.int64(1), kwargs)

    nodes = [
        make_node('Shape', [input_nodes[0]], [shape_name]),
        make_node('Shape', [shape_name], [name+'_in_dim']),
        make_node('Squeeze', [name+'_in_dim'], [name+'_in_dim_s']),
        make_node('Range', [name+'_0_s', name+'_in_dim_s', name+'_1_s'], [name+'_range']),
    ]

    for i, axis in enumerate(axis):
        if axis not in (0, 1):
            create_tensor([axis], name+'_'+str(axis), kwargs["initializer"])
        create_tensor([size[i]-1], name+'_size_'+str(i), kwargs["initializer"])
        nodes += [
            make_node('Equal', [name+'_range', name+'_'+str(axis)], [name+'_equal_'+str(i)]),
            make_node('Cast', [name+'_equal_'+str(i)], [name+'_cast_'+str(i)], to=int(TensorProto.INT64)),
            make_node('Mul', [name+'_size_'+str(i), name+'_cast_'+str(i)], [name+'_mul_'+str(i)]),
            make_node('Add', [name+'_mul_'+str(i), name+'_1'], [name+'_add_'+str(i)]),
            make_node('Mul', [name+'_add_'+str(i), shape_name], [name+'_shape_'+str(i+1)])
        ]
        shape_name = name+'_shape_'+str(i+1)

    nodes += [
        make_node('Expand', [input_nodes[0], shape_name], [name], name=name)
    ]

    return nodes


@mx_op.register("SequenceMask")
def convert_sequencemask(node, **kwargs):
    """Map MXNet's SequenceMask operator
    """
    from onnx.helper import make_node
    from onnx import TensorProto

    name, input_nodes, attrs = get_inputs(node, kwargs)

    use_sequence_length = attrs.get('use_sequence_length', 'False')
    mask_val = float(attrs.get('value', '0'))
    axis = int(attrs.get('axis', '0'))

    if(use_sequence_length == 'False'):
        return [make_node('Identity', [input_nodes[0]], [name], name=name)]

    create_tensor([0], name+'_0', kwargs["initializer"])
    create_tensor([1], name+'_1', kwargs["initializer"])
    create_tensor([2], name+'_2', kwargs["initializer"])
    create_const_scalar_node(name+'_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name+'_1_s', np.int64(1), kwargs)
    create_const_scalar_node(name+'_2_s', np.int64(2), kwargs)
    create_tensor([mask_val], name+'_mask_val', kwargs["initializer"], dtype='float32')

    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_in_shape']),
        make_node('Slice', [name+'_in_shape', name+'_0', name+'_1'], [name+'_slice_0']),
        make_node('Slice', [name+'_in_shape', name+'_1', name+'_2'], [name+'_slice_1']),
        make_node('Concat', [name+'_slice_0', name+'_1'], [name+'_shape_0'], axis=0),
        make_node('Shape', [name+'_in_shape'], [name+'_in_dim']),
        make_node('Squeeze', [name+'_in_dim'], [name+'_in_dim_s']),
        make_node('Range', [name+'_0_s', name+'_in_dim_s', name+'_1_s'], [name+'_range_0']),
        make_node('Less', [name+'_range_0', name+'_2'], [name+'_less_0']),
        make_node('Where', [name+'_less_0', name+'_in_shape', name+'_1'], [name+'_shape_1'])
    ]

    if(axis == 0):
        nodes += [
            make_node('Squeeze', [name+'_slice_0'], [name+'_max_len']),
            make_node('Range', [name+'_0_s', name+'_max_len', name+'_1_s'], [name+'_range_1']),
            make_node('Reshape', [name+'_range_1', name+'_shape_0'], [name+"_reshape_0"]),
            make_node('Cast', [input_nodes[1]], [name+'_cast'], to=int(TensorProto.INT64)),
            make_node('Less', [name+'_reshape_0', name+'_cast'], [name+'_less_1']),
            make_node('Reshape', [name+'_less_1', name+'_shape_1'], [name+"_reshape_1"]),
            make_node('Where', [name+'_reshape_1', input_nodes[0], name+'_mask_val'], [name], name=name),
        ]
    else:
        nodes += [
            make_node('Squeeze', [name+'_slice_1'], [name+'_max_len']),
            make_node('Range', [name+'_0_s', name+'_max_len', name+'_1_s'], [name+'_range_1']),
            make_node('Reshape', [input_nodes[1], name+'_shape_0'], [name+"_reshape_0"]),
            make_node('Cast', [name+"_reshape_0"], [name+'_cast'], to=int(TensorProto.INT64)),
            make_node('Less', [name+'_range_1', name+'_cast'], [name+'_less_1']),
            make_node('Reshape', [name+'_less_1', name+'_shape_1'], [name+"_reshape_1"]),
            make_node('Where', [name+'_reshape_1', input_nodes[0], name+'_mask_val'], [name], name=name),
        ]
    return nodes


@mx_op.register("Embedding")
def convert_embedding(node, **kwargs):
    """Map MXNet's Embedding operator attributes to onnx's
    Gather operator."""
    from onnx.helper import make_node
    from onnx import TensorProto

    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', 0))

    nodes = [
        make_node('Cast', [input_nodes[0]], [name+'_indices_casted'], to=int(TensorProto.INT64)),
        make_node('Gather', [input_nodes[1], name+'_indices_casted'], [name], axis=axis, name=name)
    ]

    return nodes


@mx_op.register("stack")
def convert_stack(node, **kwargs):
    """Map MXNet's stack operator to onnx operators.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', 0))
    idx = 0
    nodes = []
    for input_node in input_nodes:
        nodes.append(onnx.helper.make_node(
            "Unsqueeze",
            inputs=[input_node],
            outputs=[name+"_unsqueeze"+str(idx)],
            axes=[axis]
        ))
        idx += 1

    nodes.append(onnx.helper.make_node(
        "Concat",
        inputs=[name+"_unsqueeze"+str(i) for i in range(len(nodes))],
        outputs=[name],
        name=name,
        axis=axis
    ))
    return nodes


@mx_op.register("slice")
def convert_slice(node, **kwargs):
    """Map MXNet's slice operator to onnx Slice operator."""
    from onnx.helper import make_node

    name, input_nodes, attrs = get_inputs(node, kwargs)

    starts = convert_string_to_list(attrs.get('begin'))
    ends = convert_string_to_list(attrs.get('end'))
    steps = convert_string_to_list(attrs.get('step', '[]'))

    assert len(starts) == len(ends)
    if len(steps) == 0 or (len(steps) == 1 and steps[0] is None):
        steps = [1 for x in starts]
    else:
        assert len(steps) == len(starts)
    steps = [1 if x is None else x for x in steps]
    for i, s in enumerate(steps):
        if s < 0:
            raise NotImplementedError('slice operator does not support negative steps yet')
        if starts[i] is None:
            starts[i] = 0
        if ends[i] is None:
            ends[i] = 2**63-1

    create_const_scalar_node(name+'_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name+'_1_s', np.int64(1), kwargs)
    create_const_scalar_node(name+'_len_s', np.int64(len(starts)), kwargs)
    create_tensor(starts, name+'_starts', kwargs['initializer'])
    create_tensor(ends, name+'_ends', kwargs['initializer'])
    create_tensor(steps, name+'_steps', kwargs['initializer'])

    nodes = [
        make_node('Range', [name+'_0_s', name+'_len_s', name+'_1_s'], [name+'_axes']),
        make_node("Slice", [input_nodes[0], name+'_starts', name+'_ends', name+'_axes',
                            name+'_steps'], [name], name=name)
    ]

    return nodes


@mx_op.register("zeros_like")
def convert_zeros_like(node, **kwargs):
    """Map MXNet's zeros_like operator attributes to onnx's ConstantOfShape operator.
    """
    from onnx.helper import make_node, make_tensor
    name, input_nodes, attrs = get_inputs(node, kwargs)
    dtype = attrs.get('dtype')
    if dtype is not None:
        data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
    else:
        data_type = kwargs['in_type']

    # create tensor with shape of input
    tensor_value = make_tensor(name+"_zero", data_type, [1], [0])
    nodes = [
        make_node("Shape", [input_nodes[0]], [name+"_shape"]),
        make_node("ConstantOfShape", [name+"_shape"], [name], name=name, value=tensor_value)
    ]
    return nodes


@mx_op.register("ones_like")
def convert_ones_like(node, **kwargs):
    """Map MXNet's ones_like operator attributes to onnx's ConstantOfShape operator.
    """
    from onnx.helper import make_node, make_tensor
    name, input_nodes, attrs = get_inputs(node, kwargs)
    dtype = attrs.get('dtype')
    if dtype is not None:
        data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]
    else:
        data_type = kwargs['in_type']
    # create tensor with shape of input
    tensor_value = make_tensor(name+"_one", data_type, [1], [1])
    nodes = [
        make_node("Shape", [input_nodes[0]], [name+"_shape"]),
        make_node("ConstantOfShape", [name+"_shape"], [name], name=name, value=tensor_value)
    ]
    return nodes


@mx_op.register("_contrib_arange_like")
def convert_arange_like(node, **kwargs):
    """Map MXNet's arange_like operator attributes to onnx's Range and Reshape operators.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError("ONNX opset 11 or greater is required to export this operator")

    input_type = kwargs['in_type']
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]
    axis = attrs.get('axis', 'None')
    start = attrs.get('start', 0.)
    step = attrs.get('step', 1.)
    repeat = int(attrs.get('repeat', 1))
    if repeat != 1:
        raise NotImplementedError("arange_like operator with repeat != 1 not yet implemented.")

    create_const_scalar_node(name+"_start", np.dtype(dtype).type(start), kwargs)
    create_const_scalar_node(name+"_step", np.dtype(dtype).type(step), kwargs)
    create_const_scalar_node(name+"_half_step", np.dtype(dtype).type(float(step)*0.5), kwargs)

    nodes = []
    if axis == 'None':
        # output will be same shape as input
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+"_shape0_out"]),
            make_node("ReduceProd", [name+"_shape0_out"], [name+"_redprod0_out"]),
            make_node('Squeeze', [name+'_redprod0_out'], [name+'_reshape0_out']),
            make_node("Cast", [name+"_reshape0_out"], [name+"_cast0_out"], to=input_type),
            make_node("Mul", [name+"_cast0_out", name+"_step"], [name+"_mul0_out"]),
            make_node("Add", [name+"_mul0_out", name+"_start"], [name+"_add1_out"]),
            make_node("Sub", [name+"_add1_out", name+"_half_step"], [name+"_sub0_out"]),
            make_node("Range", [name+"_start", name+"_sub0_out", name+"_step"], [name+"_range0_out"]),
            make_node("Reshape", [name+"_range0_out", name+"_shape0_out"], [name], name=name)
        ]
    else:
        # determine shape of axis
        create_tensor([int(axis)], name+"_axis_start", kwargs["initializer"], dtype='int64')
        create_tensor([int(axis)+1], name+"_axis_end", kwargs["initializer"], dtype='int64')
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+"_shape0_out"]),
            make_node('Slice', [name+"_shape0_out", name+"_axis_start", name+"_axis_end"], [name+"_slice0_out"]),
            make_node("ReduceProd", [name+"_slice0_out"], [name+"_reprod0_out"]),
            make_node('Squeeze', [name+'_reprod0_out'], [name+'_reshape0_out']),
            make_node("Cast", [name+"_reshape0_out"], [name+"_cast0_out"], to=input_type),
            make_node("Mul", [name+"_cast0_out", name+"_step"], [name+"_mul0_out"]),
            make_node("Add", [name+"_mul0_out", name+"_start"], [name+"_add1_out"]),
            make_node("Sub", [name+"_add1_out", name+"_half_step"], [name+"_sub0_out"]),
            make_node("Range", [name+"_start", name+"_sub0_out", name+"_step"], [name], name=name)
        ]

    return nodes


@mx_op.register("_contrib_BilinearResize2D")
def convert_contrib_BilinearResize2D(node, **kwargs):
    """Map MXNet's contrib_BilinearResize2D operator attributes to onnx.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError("ONNX opset 11 or greater is required to export this operator")

    height = int(attrs.get('height', 0))
    width = int(attrs.get('width', 0))

    scale_height = float(attrs.get('scale_height', 0))
    scale_width = float(attrs.get('scale_width', 0))

    if height * width == 0 and scale_height * scale_width == 0:
        raise AttributeError('height, width or scale_height, scale_width cannot be 0')

    mode = attrs.get('mode', 'size')
    if mode != 'size':
        raise NotImplementedError('contrib_BilinearResize2D with mode other than "size" is \
                                   not supported')

    create_tensor([], name+'_roi', kwargs['initializer'], dtype='float32')

    nodes = []
    if scale_height == 0:
        create_tensor([0], name+'_0', kwargs['initializer'])
        create_tensor([2], name+'_2', kwargs['initializer'])
        create_tensor([height, width], name+'_h_w', kwargs['initializer'], dtype='int64')
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('Slice', [name+'_shape', name+'_0', name+'_2'], [name+'_shape_01']),
            make_node('Concat', [name+'_shape_01', name+'_h_w'], [name+'_new_shape'], axis=0),
            make_node('Cast', [name+'_shape'], [name+'_shape_f'], to=int(TensorProto.FLOAT)),
            make_node('Cast', [name+'_new_shape'], [name+'_new_shape_f'],
                      to=int(TensorProto.FLOAT)),
            make_node('Div', [name+'_new_shape_f', name+'_shape_f'], [name+'_scales']),
            make_node('Resize', [input_nodes[0], name+'_roi', name+'_scales'], [name],
                      mode='linear', coordinate_transformation_mode='align_corners', name=name)
        ]
    else:
        create_tensor([1, 1, scale_height, scale_width], name+'_scales', kwargs['initializer'],
                      dtype='float32')
        nodes += [
            make_node('Resize', [input_nodes[0], name+'_roi', name+'_scales'], [name],
                      mode='linear', coordinate_transformation_mode='align_corners', name=name)
        ]

    return nodes


@mx_op.register("_arange")
def convert_arange(node, **kwargs):
    """Map MXNet's arange operator attributes to onnx's Range operator.
    """
    from onnx.helper import make_node
    name, _, attrs = get_inputs(node, kwargs)

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError("ONNX opset 11 or greater is required to export this operator")

    start = attrs.get('start', 0.)
    stop = attrs.get('stop')
    step = attrs.get('step', 1.)
    dtype = attrs.get('dtype', 'float32')
    repeat = int(attrs.get('repeat', 1))

    if stop == 'None':
        stop = start
        start = 0

    if repeat != 1:
        raise NotImplementedError("arange operator with repeat != 1 not yet implemented.")

    create_const_scalar_node(name+"_start", np.dtype(dtype).type(start), kwargs)
    create_const_scalar_node(name+"_stop", np.dtype(dtype).type(stop), kwargs)
    create_const_scalar_node(name+"_step", np.dtype(dtype).type(step), kwargs)

    nodes = [
        make_node("Range", [name+"_start", name+"_stop", name+"_step"], [name], name=name)
    ]

    return nodes


@mx_op.register("reverse")
def convert_reverse(node, **kwargs):
    """Map MXNet's reverse operator attributes to ONNX
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get('axis', 0))

    # Transpose takes perm as a parameter, so we must 'pad' the input to a known dim (10 here)
    perm = [i for i in range(10)]
    perm[0], perm[axis] = axis, 0

    create_tensor([10], name+'_10', kwargs['initializer'])
    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor([-1], name+'_m1', kwargs['initializer'])
    create_tensor([axis], name+'_axis', kwargs['initializer'])
    create_tensor([axis+1], name+'_axis_p1', kwargs['initializer'])
    create_const_scalar_node(name+'_m1_s', np.int64(-1), kwargs)

    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_shape']),
        make_node('Shape', [name+'_shape'], [name+'_dim']),
        make_node('Sub', [name+'_10', name+'_dim'], [name+'_sub']),
        make_node('Concat', [name+'_0', name+'_sub'], [name+'_concat'], axis=0),
        make_node('Pad', [name+'_shape', name+'_concat', name+'_1'], [name+'_shape_10_dim']),
        make_node('Reshape', [input_nodes[0], name+'_shape_10_dim'], [name+'_data_10_dim']),
        make_node('Transpose', [name+'_data_10_dim'], [name+'_data_t'], perm=perm),
        make_node('Slice', [name+'_shape', name+'_axis', name+'_axis_p1'], [name+'_axis_len']),
        make_node('Sub', [name+'_axis_len', name+'_1'], [name+'_axis_len_m1']),
        make_node('Squeeze', [name+'_axis_len_m1'], [name+'_axis_len_m1_s']),
        make_node('Range', [name+'_axis_len_m1_s', name+'_m1_s', name+'_m1_s'], [name+'_indices']),
        make_node('Gather', [name+'_data_t', name+'_indices'], [name+'_gather']),
        make_node('Transpose', [name+'_gather'], [name+'_data_reversed'], perm=perm),
        make_node('Reshape', [name+'_data_reversed', name+'_shape'], [name], name=name)
    ]

    return nodes


@mx_op.register('repeat')
def convert_repeat(node, **kwargs):
    """Map MXNet's repeat operator attributes to onnx's Tile operator.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')

    repeats = int(attrs.get('repeats', 1))
    axis = attrs.get('axis', 'None')

    if repeats <= 0:
        raise NotImplementedError('repeat operator does not support parameter repeats==0')

    nodes = []
    if axis == 'None':
        create_tensor([repeats], name+'_rep', kwargs['initializer'])
        create_tensor([1, repeats], name+'_repeats', kwargs['initializer'])
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('ReduceProd', [name+'_shape'], [name+'_size']),
            make_node('Reshape', [input_nodes[0], name+'_size'], [name+'_flat']),
            make_node('Unsqueeze', [name+'_flat'], [name+'_unsqueeze'], axes=[-1]),
            make_node('Tile', [name+'_unsqueeze', name+'_repeats'], [name+'_tile']),
            make_node('Mul', [name+'_size', name+'_rep'], [name+'_new_size']),
            make_node('Reshape', [name+'_tile', name+'_new_size'], [name], name=name)
        ]
    else:
        axis = int(axis)
        repeats -= 1
        create_tensor([repeats], name+'_repeats', kwargs['initializer'])
        create_tensor([1], name+'_1', kwargs['initializer'])
        create_tensor([0], name+'_0', kwargs['initializer'])
        create_tensor([axis], name+'_axis', kwargs['initializer'])
        create_const_scalar_node(name+"_0_s", np.int64(0), kwargs)
        create_const_scalar_node(name+"_1_s", np.int64(1), kwargs)
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('Shape', [name+'_shape'], [name+'_dim']),
            make_node('Squeeze', [name+'_dim'], [name+'_dim_s']),
            make_node('Range', [name+'_0_s', name+'_dim_s', name+'_1_s'], [name+'_range'])
        ]
        if axis < 0:
            nodes += [
                make_node('Add', [name+'_axis', name+'_dim'], [name+'_true_axis']),
                make_node('Equal', [name+'_range', name+'_true_axis'], [name+'_one_hot'])
                ]
        else:
            nodes += [
                make_node('Equal', [name+'_range', name+'_axis'], [name+'_one_hot'])
                ]
        nodes += [
            make_node('Cast', [name+'_one_hot'], [name+'_one_hot_int'], to=int(TensorProto.INT64)),
            make_node('Mul', [name+'_repeats', name+'_one_hot_int'], [name+'_mul']),
            make_node('Add', [name+'_mul', name+'_1'], [name+'_add']),
            make_node('Concat', [name+'_1', name+'_add'], [name+'_repeats_tensor'], axis=0)
            ]
        if axis == -1:
            nodes += [
                make_node('Concat', [name+'_shape', name+'_1'], [name+'_unsqueeze_shape'], axis=0),
                make_node('Reshape', [input_nodes[0], name+'_unsqueeze_shape'],
                          [name+'_unsqueeze'])
                ]
        else:
            nodes += [
                make_node('Unsqueeze', [input_nodes[0]], [name+'_unsqueeze'], axes=[axis+1])
                ]
        nodes += [
            make_node('Tile', [name+'_unsqueeze', name+'_repeats_tensor'], [name+'_tile']),
            make_node('Mul', [name+'_shape', name+'_add'], [name+'_new_shape']),
            make_node('Reshape', [name+'_tile', name+'_new_shape'], [name], name=name)
            ]

    return nodes


@mx_op.register('_contrib_box_nms')
def convert_contrib_box_nms(node, **kwargs):
    """Map MXNet's _contrib_box_nms operator to ONNX
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')

    input_type = kwargs['in_type']
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]

    overlap_thresh = float(attrs.get('overlap_thresh', '0.5'))
    valid_thresh = float(attrs.get('valid_thresh', '0'))
    topk = int(attrs.get('topk', '-1'))
    coord_start = int(attrs.get('coord_start', '2'))
    score_index = int(attrs.get('score_index', '1'))
    id_index = int(attrs.get('id_index', '-1'))
    force_suppress = attrs.get('force_suppress', 'True')
    background_id = int(attrs.get('background_id', '-1'))
    in_format = attrs.get('in_format', 'corner')
    out_format = attrs.get('out_format', 'corner')

    center_point_box = 0 if in_format == 'corner' else 1

    if in_format != out_format:
        raise NotImplementedError('box_nms does not currently support in_fomat != out_format')

    if background_id != -1:
        raise NotImplementedError('box_nms does not currently support background_id != -1')

    if id_index != -1 or force_suppress == 'False':
        logging.warning('box_nms: id_idex != -1 or/and force_suppress == False detected. '
                        'However, due to ONNX limitations, boxes of different categories will NOT '
                        'be exempted from suppression. This might lead to different behavior than '
                        'native MXNet')

    create_tensor([coord_start], name+'_cs', kwargs['initializer'])
    create_tensor([coord_start+4], name+'_cs_p4', kwargs['initializer'])
    create_tensor([score_index], name+'_si', kwargs['initializer'])
    create_tensor([score_index+1], name+'_si_p1', kwargs['initializer'])
    create_tensor([topk], name+'_topk', kwargs['initializer'])
    create_tensor([overlap_thresh], name+'_ot', kwargs['initializer'], dtype=np.float32)
    create_tensor([valid_thresh], name+'_vt', kwargs['initializer'], dtype=np.float32)
    create_tensor([-1], name+'_m1', kwargs['initializer'])
    create_tensor([-1], name+'_m1_f', kwargs['initializer'], dtype=dtype)
    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor([2], name+'_2', kwargs['initializer'])
    create_tensor([3], name+'_3', kwargs['initializer'])
    create_tensor([0, 1, -1], name+'_scores_shape', kwargs['initializer'])
    create_tensor([0, 0, 1, 0], name+'_pad', kwargs['initializer'])
    create_tensor([0, -1], name+'_bat_spat_helper', kwargs['initializer'])
    create_const_scalar_node(name+"_0_s", np.int64(0), kwargs)
    create_const_scalar_node(name+"_1_s", np.int64(1), kwargs)

    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_shape']),
        make_node('Shape', [name+'_shape'], [name+'_dim']),
        make_node('Sub', [name+'_dim', name+'_2'], [name+'_dim_m2']),
        make_node('Slice', [name+'_shape', name+'_dim_m2', name+'_dim'], [name+'_shape_last2']),
        make_node('Concat', [name+'_m1', name+'_shape_last2'], [name+'_shape_3d'], axis=0),
        make_node('Reshape', [input_nodes[0], name+'_shape_3d'], [name+'_data_3d']),
        make_node('Slice', [name+'_data_3d', name+'_cs', name+'_cs_p4', name+'_m1'],
                  [name+'_boxes']),
        make_node('Slice', [name+'_data_3d', name+'_si', name+'_si_p1', name+'_m1'],
                  [name+'_scores_raw']),
        make_node('Reshape', [name+'_scores_raw', name+'_scores_shape'], [name+'_scores']),
        make_node('Shape', [name+'_scores'], [name+'_scores_shape_actual']),
        make_node('NonMaxSuppression',
                  [name+'_boxes', name+'_scores', name+'_topk', name+'_ot', name+'_vt'],
                  [name+'_nms'], center_point_box=center_point_box),
        make_node('Slice', [name+'_nms', name+'_0', name+'_3', name+'_m1', name+'_2'],
                  [name+'_nms_sliced']),
        make_node('GatherND', [name+'_data_3d', name+'_nms_sliced'], [name+'_candidates']),
        make_node('Pad', [name+'_candidates', name+'_pad', name+'_m1_f'], [name+'_cand_padded']),
        make_node('Shape', [name+'_nms'], [name+'_nms_shape']),
        make_node('Slice', [name+'_nms_shape', name+'_0', name+'_1'], [name+'_cand_cnt']),
        make_node('Squeeze', [name+'_cand_cnt'], [name+'_cc_s']),
        make_node('Range', [name+'_0_s', name+'_cc_s', name+'_1_s'], [name+'_cand_indices']),
        make_node('Slice', [name+'_scores_shape_actual', name+'_0', name+'_3', name+'_m1',
                            name+'_2'], [name+'_shape_bat_spat']),
        make_node('Slice', [name+'_shape_bat_spat', name+'_1', name+'_2'], [name+'_spat_dim']),
        make_node('Expand', [name+'_cand_cnt', name+'_shape_bat_spat'], [name+'_base_indices']),
        make_node('ScatterND', [name+'_base_indices', name+'_nms_sliced', name+'_cand_indices'],
                  [name+'_indices']),
        make_node('TopK', [name+'_indices', name+'_spat_dim'], [name+'_indices_sorted', name+'__'],
                  largest=0, axis=-1, sorted=1),
        make_node('Gather', [name+'_cand_padded', name+'_indices_sorted'], [name+'_gather']),
        make_node('Reshape', [name+'_gather', name+'_shape'], [name+'0'])
    ]

    return nodes


@mx_op.register("_greater_scalar")
def convert_greater_scalar(node, **kwargs):
    """Map MXNet's greater_scalar operator attributes to onnx's Greater
    operator and return the created node.
    """
    from onnx.helper import make_node, make_tensor
    name, input_nodes, attrs = get_inputs(node, kwargs)

    scalar = float(attrs.get('scalar'))
    input_type = kwargs['in_type']
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]

    if str(dtype).startswith('int'):
        scalar = int(scalar)
    else:
        if dtype == 'float16':
            # when using float16, we must convert it to np.uint16 view first
            scalar = np.float16(scalar).view(np.uint16)

    tensor_value = make_tensor(name+"_scalar", input_type, [1], [scalar])
    nodes = [
        make_node("Constant", [], [name+"_rhs"], value=tensor_value),
        make_node("Greater", [input_nodes[0], name+"_rhs"], [name+"_gt"]),
        make_node("Cast", [name+"_gt"], [name], to=input_type, name=name)
    ]
    return nodes


@mx_op.register("_lesser_scalar")
def convert_lesser_scalar(node, **kwargs):
    """Map MXNet's lesser_scalar operator attributes to onnx's Less
    operator and return the created node.
    """
    from onnx.helper import make_node, make_tensor
    name, input_nodes, attrs = get_inputs(node, kwargs)

    scalar = float(attrs.get('scalar'))
    input_type = kwargs['in_type']
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]

    if str(dtype).startswith('int'):
        scalar = int(scalar)
    else:
        if dtype == 'float16':
            # when using float16, we must convert it to np.uint16 view first
            scalar = np.float16(scalar).view(np.uint16)

    tensor_value = make_tensor(name+"_scalar", input_type, [1], [scalar])
    nodes = [
        make_node("Constant", [], [name+"_rhs"], value=tensor_value),
        make_node("Less", [input_nodes[0], name+"_rhs"], [name+"_lt"]),
        make_node("Cast", [name+"_lt"], [name], to=input_type, name=name)
    ]
    return nodes


@mx_op.register("_equal_scalar")
def convert_equal_scalar(node, **kwargs):
    """Map MXNet's equal_scalar operator attributes to onnx.
    """
    from onnx.helper import make_node, make_tensor
    name, input_nodes, attrs = get_inputs(node, kwargs)

    scalar = float(attrs.get('scalar'))
    input_type = kwargs['in_type']
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]

    if str(dtype).startswith('int'):
        scalar = int(scalar)
    else:
        if dtype == 'float16':
            # when using float16, we must convert it to np.uint16 view first
            scalar = np.float16(scalar).view(np.uint16)

    tensor_value = make_tensor(name+"_scalar", input_type, [1], [scalar])
    nodes = [
        make_node("Constant", [], [name+"_rhs"], value=tensor_value),
        make_node("Equal", [input_nodes[0], name+"_rhs"], [name+"_eq"]),
        make_node("Cast", [name+"_eq"], [name], to=input_type, name=name)
    ]
    return nodes


@mx_op.register("where")
def convert_where(node, **kwargs):
    """Map MXNet's where operator attributes to onnx's Where
    operator and return the created node.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, _ = get_inputs(node, kwargs)
    nodes = [
        make_node("Cast", [input_nodes[0]], [name+"_bool"], to=int(TensorProto.BOOL)),
        make_node("Where", [name+"_bool", input_nodes[1], input_nodes[2]], [name], name=name)
    ]
    return nodes


@mx_op.register('_maximum_scalar')
def convert_maximum_scalar(node, **kwargs):
    """Map MXNet's _maximum_scalar
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    input_type = int(kwargs['in_type'])
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]

    scalar = None
    if 'float' in str(dtype):
        scalar = float(attrs.get('scalar', '0'))
    else:
        scalar = int(attrs.get('scalar', '0'))

    create_tensor([scalar], name+'_scalar', kwargs['initializer'], dtype=dtype)
    nodes = [
        make_node('Max', [input_nodes[0], name+'_scalar'], [name], name=name)
    ]

    return nodes

@mx_op.register('_minimum_scalar')
def convert_minimum_scalar(node, **kwargs):
    """Map MXNet's _minimum_scalar
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    input_type = int(kwargs['in_type'])
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]

    scalar = None
    if 'float' in str(dtype):
        scalar = float(attrs.get('scalar', '0'))
    else:
        scalar = int(attrs.get('scalar', '0'))

    create_tensor([scalar], name+'_scalar', kwargs['initializer'], dtype=dtype)
    nodes = [
        make_node('Min', [input_nodes[0], name+'_scalar'], [name], name=name)
    ]

    return nodes

@mx_op.register("_contrib_box_decode")
def convert_contrib_box_decode(node, **kwargs):
    """Map MXNet's _contrib_box_decode operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    data = input_nodes[0]
    anchors = input_nodes[1]
    input_type = kwargs['in_type']
    fmt = attrs.get('format', 'center')
    std0 = float(attrs.get('std0', '1.'))
    std1 = float(attrs.get('std1', '1.'))
    std2 = float(attrs.get('std2', '1.'))
    std3 = float(attrs.get('std3', '1.'))
    clip = float(attrs.get('clip', '-1.'))

    if fmt not in ['center', 'corner']:
        raise NotImplementedError("format must be either corner or center.")

    create_tensor([0], name+'_0', kwargs["initializer"])
    create_tensor([2], name+'_2', kwargs["initializer"])
    create_tensor([4], name+'_4', kwargs["initializer"])
    create_tensor([2], name+'_2f', kwargs["initializer"], dtype='float32')
    create_tensor([clip], name+'_clip', kwargs["initializer"], dtype='float32')
    create_tensor([std0, std1, std2, std3], name+'_std_1d', kwargs["initializer"], dtype='float32')
    create_tensor([1, 4], name+'_std_shape', kwargs["initializer"])

    nodes = [
        make_node("Cast", [data], [name+'_data'], to=int(onnx.TensorProto.FLOAT)),
        make_node("Cast", [anchors], [name+'_anchors'], to=int(onnx.TensorProto.FLOAT)),
        make_node('Reshape', [name+'_std_1d', name+'_std_shape'], [name+'_std']),
        make_node("Mul", [name+'_data', name+'_std'], [name+'_mul0_out']),
        make_node('Slice', [name+'_mul0_out', name+'_0', name+'_2', name+'_2'], [name+'_data_xy']),
        make_node('Slice', [name+'_mul0_out', name+'_2', name+'_4', name+'_2'], [name+'_data_wh']),
    ]

    if fmt == 'corner':
        nodes += [
            make_node('Slice', [name+'_anchors', name+'_0', name+'_2', name+'_2'], [name+'_slice0_out']),
            make_node('Slice', [name+'_anchors', name+'_2', name+'_4', name+'_2'], [name+'_slice1_out']),
            make_node('Sub', [name+'_slice1_out', name+'_slice0_out'], [name+'_anchor_wh']),
            make_node('Div', [name+'_anchor_wh', name+'_2f'], [name+'_div0_out']),
            make_node("Add", [name+'_slice0_out', name+'_div0_out'], [name+'_anchor_xy']),
        ]
    else:
        nodes += [
            make_node('Slice', [name+'_anchors', name+'_0', name+'_2', name+'_2'], [name+'_anchor_xy']),
            make_node('Slice', [name+'_anchors', name+'_2', name+'_4', name+'_2'], [name+'_anchor_wh']),
        ]

    nodes += [
        make_node("Mul", [name+'_data_xy', name+'_anchor_wh'], [name+'_mul1_out']),
        make_node("Add", [name+'_mul1_out', name+'_anchor_xy'], [name+'_add0_out']),
    ]

    if clip > 0.:
        nodes += [
            make_node("Less", [name+"_data_wh", name+"_clip"], [name+"_less0_out"]),
            make_node('Where', [name+'_less0_out', name+'_data_wh', name+'_clip'], [name+'_where0_out']),
            make_node("Exp", [name+'_where0_out'], [name+'_exp0_out']),
        ]
    else:
        nodes += [
            make_node("Exp", [name+'_data_wh'], [name+'_exp0_out']),
        ]

    nodes += [
        make_node("Mul", [name+'_exp0_out', name+'_anchor_wh'], [name+'_mul2_out']),
        make_node('Div', [name+'_mul2_out', name+'_2f'], [name+'_div1_out']),
        make_node('Sub', [name+'_add0_out', name+'_div1_out'], [name+'_sub0_out']),
        make_node('Add', [name+'_add0_out', name+'_div1_out'], [name+'_add1_out']),
        make_node('Concat', [name+'_sub0_out', name+'_add1_out'], [name+'concat0_out'], axis=2),
        make_node("Cast", [name+'concat0_out'], [name], to=input_type, name=name)
    ]

    return nodes

@mx_op.register("_contrib_AdaptiveAvgPooling2D")
def convert_contrib_AdaptiveAvgPooling2D(node, **kwargs):
    """Map MXNet's _contrib_BilinearResize2D operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    output_size = attrs.get('output_size', '1')
    output_size = convert_string_to_list(output_size)

    if len(output_size) <= 2:
        if output_size[0] != 1 or (len(output_size) == 2 and output_size[1] != 1):
            raise NotImplementedError("_contrib_AdaptiveAvgPooling2D operator with output_size != 1 \
                                not yet implemented.")
    nodes = [
        make_node("GlobalAveragePool", [input_nodes[0]], [name], name=name)
    ]

    return nodes


@mx_op.register('broadcast_mod')
def convert_broadcast_mod(node, **kwargs):
    """Map MXNet's broadcast_mod operator
    """
    from onnx.helper import make_node
    name, input_nodes, _ = get_inputs(node, kwargs)

    # The behavior of MXNet mod is a mixture of np.mod and np.fmod
    # note: the behavior when divison by 0 is supposed to be platform dependent
    #       but here we set the result to 0 to be consistent with MXNet
    nodes = [
        make_node('Sub', [input_nodes[1], input_nodes[1]], [name+'_zero']),
        make_node('Mod', [input_nodes[0], input_nodes[1]], [name+'_mod'], fmod=1),
        make_node('Less', [input_nodes[0], name+'_zero'], [name+'_mask_0']),
        make_node('Less', [input_nodes[1], name+'_zero'], [name+'_mask_1']),
        make_node('Equal', [name+'_mod', name+'_zero'], [name+'_mask_2_']),
        make_node('Not', [name+'_mask_2_'], [name+'_mask_2']),
        make_node('Xor', [name+'_mask_0', name+'_mask_1'], [name+'_mask_']),
        make_node('And', [name+'_mask_', name+'_mask_2'], [name+'_mask']),
        make_node('Where', [name+'_mask', input_nodes[1], name+'_zero'], [name+'_adjustment']),
        make_node('Add', [name+'_mod', name+'_adjustment'], [name+'_adjusted']),
        make_node('Equal', [input_nodes[1], name+'_zero'], [name+'_mask_div_0']),
        make_node('Where', [name+'_mask_div_0', name+'_zero', name+'_adjusted'], [name], name=name)
        ]

    return nodes


@mx_op.register("reshape_like")
def convert_reshape_like(node, **kwargs):
    """Map MXNet's reshape_like operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    lhs = input_nodes[0]
    rhs = input_nodes[1]

    lhs_begin = str(attrs.get('lhs_begin', '0'))
    rhs_begin = str(attrs.get('rhs_begin', '0'))
    lhs_end = str(attrs.get('lhs_end', 'None'))
    rhs_end = str(attrs.get('rhs_end', 'None'))

    if lhs_begin == 'None' or rhs_begin == 'None':
        raise NotImplementedError("lhs_begin and rhs_begin should not be None.")

    lhs_begin = int(lhs_begin)
    rhs_begin = int(rhs_begin)

    # basic case
    if lhs_begin == 0 and lhs_end == 'None' and rhs_begin == 0 and rhs_end == 'None':
        nodes = [
            make_node('Shape', [rhs], [name+'_shape_rhs']),
            make_node('Reshape', [lhs, name+'_shape_rhs'], [name], name=name)
        ]
        return nodes

    create_tensor([0], name+'_0', kwargs["initializer"])
    nodes = [
        make_node('Shape', [lhs], [name+'_lhs_shape']),
        make_node('Shape', [name+'_lhs_shape'], [name+'_lhs_dim']),
        make_node('Shape', [rhs], [name+'_rhs_shape']),
        make_node('Shape', [name+'_rhs_shape'], [name+'_rhs_dim']),
    ]

    if lhs_begin >= 0:
        create_tensor([lhs_begin], name+'_lhs_begin', kwargs["initializer"])
    else:
        create_tensor([lhs_begin], name+'_lhs_begin_neg', kwargs["initializer"])
        nodes += [
            make_node('Add', [name+'_lhs_dim', name+'_lhs_begin_neg'], [name+'_lhs_begin']),
        ]

    if rhs_begin >= 0:
        create_tensor([rhs_begin], name+'_rhs_begin', kwargs["initializer"])
    else:
        create_tensor([rhs_begin], name+'_rhs_begin_neg', kwargs["initializer"])
        nodes += [
            make_node('Add', [name+'_rhs_dim', name+'_rhs_begin_neg'], [name+'_rhs_begin']),
        ]

    if lhs_end == 'None':
        nodes += [
            make_node('Add', [name+'_lhs_dim', name+'_0'], [name+'_lhs_end']),
        ]
    else:
        lhs_end = int(lhs_end)
        if lhs_end >= 0:
            create_tensor([lhs_end], name+'_lhs_end', kwargs["initializer"])
        else:
            create_tensor([lhs_end], name+'_lhs_end_neg', kwargs["initializer"])
            nodes += [
                make_node('Add', [name+'_lhs_dim', name+'_lhs_end_neg'], [name+'_lhs_end']),
            ]

    if rhs_end == 'None':
        nodes += [
            make_node('Add', [name+'_rhs_dim', name+'_0'], [name+'_rhs_end']),
        ]
    else:
        rhs_end = int(rhs_end)
        if rhs_end >= 0:
            create_tensor([rhs_end], name+'_rhs_end', kwargs["initializer"])
        else:
            create_tensor([rhs_end], name+'_rhs_end_neg', kwargs["initializer"])
            nodes += [
                make_node('Add', [name+'_rhs_dim', name+'_rhs_end_neg'], [name+'_rhs_end']),
            ]

    nodes += [
        make_node('Slice', [name+'_lhs_shape', name+'_0', name+'_lhs_begin'], [name+'_slice0_out']),
        make_node('Slice', [name+'_rhs_shape', name+'_rhs_begin', name+'_rhs_end'], [name+'_slice1_out']),
        make_node('Concat', [name+'_slice0_out', name+'_slice1_out'], [name+'_concat0_out'], axis=0),
        make_node('Slice', [name+'_lhs_shape', name+'_lhs_end', name+'_lhs_dim'], [name+'_slice2_out']),
        make_node('Concat', [name+'_concat0_out', name+'_slice2_out'], [name+'_concat1_out'], axis=0),
        make_node('Reshape', [lhs, name+'_concat1_out'], [name], name=name)
    ]

    return nodes


@mx_op.register("gather_nd")
def convert_gather_nd(node, **kwargs):
    """Map MXNet's gather_ND operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, _ = get_inputs(node, kwargs)

    data = input_nodes[0]
    indices = input_nodes[1]

    # Onnx Transpose operator takes perm as a parameter, so we need to 'pad'
    # the input to a known dim (10 here)
    perm = [9] + [i for i in range(1, 9)] + [0]

    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor([10], name+'_10', kwargs['initializer'])
    nodes = [
        # Generate 10-d filter
        make_node('Shape', [indices], [name+'_indices_shape']),
        make_node('Shape', [name+'_indices_shape'], [name+'_indices_dim']),
        make_node('Sub', [name+'_10', name+'_indices_dim'], [name+'_sub0_out']),
        make_node('Concat', [name+'_0', name+'_sub0_out'], [name+'_concat0_out'], axis=0),
        make_node('Pad', [name+'_indices_shape', name+'_concat0_out', name+'_1'], [name+'_shape_10_dim']),
        make_node('Reshape', [indices, name+'_shape_10_dim'], [name+'_indices_10_dim']),
        make_node('Transpose', [name+'_indices_10_dim'], [name+'_transpose0_output'], perm=perm),
        # Reshape filter to acutall dim for GatherND computation
        make_node('Slice', [name+'_indices_shape', name+'_0', name+'_1'],
                  [name+'_slice0_out']),
        make_node('Slice', [name+'_indices_shape', name+'_1', name+'_indices_dim'],
                  [name+'_slice1_out']),
        make_node('Concat', [name+'_slice1_out', name+'_slice0_out'], [name+'_concat1_out'], axis=0),
        make_node('Reshape', [name+'_transpose0_output', name+'_concat1_out'], [name+'_reshape0_out']),
        # Cast data type for indicies
        make_node('Cast', [name+'_reshape0_out'], [name+'_cast0_out'], to=int(onnx.TensorProto.INT64)),
        make_node('GatherND', [data, name+'_cast0_out'], [name], name=name)
    ]

    return nodes


@mx_op.register('UpSampling')
def convert_upsampling(node, **kwargs):
    """Map MXNet's UpSampling operator to onnx.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    scale = int(attrs.get('scale', '1'))
    sample_type = attrs.get('sample_type')
    num_args = int(attrs.get('num_args', '1'))

    if num_args > 1:
        raise NotImplementedError('Upsampling conversion does not currently support num_args > 1')

    if sample_type != 'nearest':
        raise NotImplementedError('Upsampling conversion does not currently support \
                                   sample_type != nearest')

    create_tensor([], name+'_roi', kwargs['initializer'], dtype='float32')
    create_tensor([1, 1, scale, scale], name+'_scales', kwargs['initializer'],
                  dtype='float32')
    nodes = [
        make_node('Resize', [input_nodes[0], name+'_roi', name+'_scales'], [name], mode='nearest',
                  coordinate_transformation_mode='half_pixel')
    ]

    return nodes


@mx_op.register('SwapAxis')
def convert_swapaxis(node, **kwargs):
    """Map MXNet's SwapAxis operator
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    dim1 = int(attrs.get('dim1', '0'))
    dim2 = int(attrs.get('dim2', '0'))

    if dim1 < 0 or dim2 < 0:
        raise NotImplementedError('SwapAxis conversion does not support dim1 < 0\
                                   or dim2 < 0')

    indices = [[dim1], [dim2]]
    vals = [dim2, dim1]
    perm = [i for i in range(10)]
    perm[dim1], perm[dim2] = dim2, dim1

    create_tensor(indices, name+'_ind', kwargs['initializer'])
    create_tensor(indices[::-1], name+'_ind_rev', kwargs['initializer'])
    create_tensor(vals, name+'_vals', kwargs['initializer'])
    create_tensor(perm, name+'_perm', kwargs['initializer'])
    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor([10], name+'_10', kwargs['initializer'])

    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_shape']),
        make_node('Shape', [name+'_shape'], [name+'_dim']),
        make_node('Sub', [name+'_10', name+'_dim'], [name+'_sub']),
        make_node('ScatterND', [name+'_perm', name+'_ind', name+'_vals'],
                  [name+'_perm_new']),
        make_node('GatherND', [name+'_shape', name+'_ind'], [name+'_gather']),
        make_node('ScatterND', [name+'_shape', name+'_ind_rev', name+'_gather'],
                  [name+'_shape_new']),
        make_node('Concat', [name+'_0', name+'_sub'], [name+'_pad'], axis=0),
        make_node('Pad', [name+'_shape', name+'_pad', name+'_1'], [name+'_shape_padded']),
        make_node('Reshape', [input_nodes[0], name+'_shape_padded'], [name+'_data_padded']),
        make_node('Transpose', [name+'_data_padded'], [name+'_trans'], perm=perm),
        make_node('Reshape', [name+'_trans', name+'_shape_new'], [name])
    ]

    return nodes


@mx_op.register('slice_like')
def convert_slice_like(node, **kwargs):
    """Map MXNet's slice_like operator to onnx Slice operator."""
    from onnx.helper import make_node, make_tensor
    from onnx import TensorProto

    name, input_nodes, attrs = get_inputs(node, kwargs)

    axes = convert_string_to_list(attrs.get('axes', 'None'))
    zero = make_tensor(name+'_zero', TensorProto.INT64, [1], [0])

    nodes = []
    if axes == [None]:
        nodes += [
            make_node('Shape', [input_nodes[1]], [name+'_shape_1']),
            make_node('Shape', [name+'_shape_1'], [name+'_dim_1']),
            make_node('ConstantOfShape', [name+'_dim_1'], [name+'_starts'], value=zero),
            make_node('Slice', [input_nodes[0], name+'_starts', name+'_shape_1'], [name])
        ]
    else:
        axes = [[i] for i in axes]
        create_tensor([0], name+'_0', kwargs['initializer'])
        create_tensor(axes, name+'_axes_', kwargs['initializer'])
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+'_shape_0']),
            make_node('Shape', [input_nodes[1]], [name+'_shape_1']),
            make_node('Shape', [name+'_shape_0'], [name+'_dim_0']),
            make_node('Less', [name+'_axes_', name+'_0'], [name+'_less']),
            make_node('Cast', [name+'_less'], [name+'_mask'], to=int(TensorProto.INT64)),
            make_node('Mul', [name+'_mask', name+'_dim_0'], [name+'_mul']),
            make_node('Add', [name+'_axes_', name+'_mul'], [name+'_axes']),
            make_node('ConstantOfShape', [name+'_dim_0'], [name+'_starts'], value=zero),
            make_node('GatherND', [name+'_shape_1', name+'_axes'], [name+'_gather']),
            make_node('ScatterND', [name+'_shape_0', name+'_axes', name+'_gather'],
                      [name+'_ends']),
            make_node('Slice', [input_nodes[0], name+'_starts', name+'_ends'], [name])
            ]

    return nodes


@mx_op.register("broadcast_like")
def convert_broadcast_like(node, **kwargs):
    """Map MXNet's broadcast_like operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    lhs = input_nodes[0]
    rhs = input_nodes[1]
    lhs_axes = convert_string_to_list(str(attrs.get('lhs_axes', 'None')))
    rhs_axes = convert_string_to_list(str(attrs.get('rhs_axes', 'None')))

    if lhs_axes[0] is None or rhs_axes[0] is None:
        nodes = [
            make_node('Shape', [rhs], [name+'_rhs_shape']),
            make_node('Expand', [lhs, name+'_rhs_shape'], [name], name=name)
        ]
        return nodes

    lhs_axes = [[i] for i in lhs_axes]
    rhs_axes = [[i] for i in rhs_axes]

    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor(lhs_axes, name+'_lhs_axes', kwargs['initializer'])
    create_tensor(rhs_axes, name+'_rhs_axes', kwargs['initializer'])

    nodes = [
        make_node('Shape', [lhs], [name+'_lhs_shape']),
        make_node('Shape', [rhs], [name+'_rhs_shape']),
        make_node('Shape', [name+'_lhs_shape'], [name+'_lhs_dim']),
        make_node('Less', [name+'_lhs_axes', name+'_0'], [name+'_less']),
        make_node('Cast', [name+'_less'], [name+'_mask'], to=int(onnx.TensorProto.INT64)),
        make_node('Mul', [name+'_mask', name+'_lhs_dim'], [name+'_mul']),
        make_node('Add', [name+'_lhs_axes', name+'_mul'], [name+'_lhs_axes_positive']),
        make_node('GatherND', [name+'_rhs_shape', name+'_rhs_axes'], [name+'_gather']),
        make_node('ScatterND', [name+'_lhs_shape', name+'_lhs_axes_positive', name+'_gather'],
                  [name+'_scatter']),
        make_node('Expand', [lhs, name+'_scatter'], [name], name=name)
    ]

    return nodes


@mx_op.register('_contrib_ROIAlign')
def convert_contrib_roialign(node, **kwargs):
    """Map MXNet's _contrib_ROIAlign
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    pooled_size = convert_string_to_list(str(attrs.get('pooled_size')))
    spatial_scale = float(attrs.get('spatial_scale'))
    sample_ratio = int(attrs.get('sample_ratio', '0'))
    position_sensitive = attrs.get('position_sensitive', 'False')
    aligned = attrs.get('aligned', 'False')

    if position_sensitive != 'False':
        raise NotImplementedError('_contrib_ROIAlign does not currently support \
                                   position_sensitive!=False')
    if aligned != 'False':
        raise NotImplementedError('_contrib_ROIAlign does not currently support \
                                   aligned!=False')

    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor([5], name+'_5', kwargs['initializer'])

    nodes = [
        make_node('Slice', [input_nodes[1], name+'_1', name+'_5', name+'_1'], [name+'_rois']),
        make_node('Slice', [input_nodes[1], name+'_0', name+'_1', name+'_1'], [name+'_inds__']),
        make_node('Squeeze', [name+'_inds__'], [name+'_inds_'], axes=(1,)),
        make_node('Cast', [name+'_inds_'], [name+'_inds'], to=int(TensorProto.INT64)),
        make_node('RoiAlign', [input_nodes[0], name+'_rois', name+'_inds'], [name],
                  mode='avg', output_height=pooled_size[0], output_width=pooled_size[1],
                  sampling_ratio=sample_ratio, spatial_scale=spatial_scale)
    ]

    return nodes


@mx_op.register("batch_dot")
def convert_batch_dot(node, **kwargs):
    """Map MXNet's batch_dot operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    lhs = input_nodes[0]
    rhs = input_nodes[1]
    transpose_a = str(attrs.get('transpose_a', 'False'))
    transpose_b = str(attrs.get('transpose_b', 'False'))
    perm = [0, 2, 1]

    if transpose_a == 'False' and transpose_b == 'False':
        nodes = [
            make_node('MatMul', [lhs, rhs], [name]),
        ]
        return nodes

    create_tensor([-2], name+'_-2', kwargs['initializer'])
    create_tensor([-1], name+'_-1', kwargs['initializer'])
    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([100], name+'_100', kwargs['initializer'])

    nodes = []
    if transpose_a != 'False' and transpose_b == 'False':
        nodes += [
            make_node('Shape', [lhs], [name+'_lhs_shape']),
            make_node('Shape', [name+'_lhs_shape'], [name+'_lhs_dim']),
            make_node('Slice', [name+'_lhs_shape', name+'_0', name+'_-2'],
                      [name+'_lhs_slice0']),
            make_node('Slice', [name+'_lhs_shape', name+'_-2', name+'_100'],
                      [name+'_lhs_slice1']),
            make_node('Concat', [name+'_-1', name+'_lhs_slice1'], [name+'_lhs_concat1'], axis=0),
            make_node('Reshape', [lhs, name+'_lhs_concat1'], [name+'_lhs_3d']),
            make_node('Transpose', [name+'_lhs_3d'], [name+'_lhs_3d_transpose'], perm=perm),
            make_node('Shape', [name+'_lhs_3d_transpose'], [name+'_lhs_shape_3d']),
            make_node('Slice', [name+'_lhs_shape_3d', name+'_-2', name+'_100'],
                      [name+'_lhs_slice2']),
            make_node('Concat', [name+'_lhs_slice0', name+'_lhs_slice2'], [name+'_lhs_concat2'], axis=0),
            make_node('Reshape', [name+'_lhs_3d_transpose', name+'_lhs_concat2'], [name+'_lhs']),
            make_node('MatMul', [name+'_lhs', rhs], [name]),
        ]

    elif transpose_a == 'False' and transpose_b != 'False':
        nodes += [
            make_node('Shape', [rhs], [name+'_rhs_shape']),
            make_node('Shape', [name+'_rhs_shape'], [name+'_rhs_dim']),
            make_node('Slice', [name+'_rhs_shape', name+'_0', name+'_-2'],
                      [name+'_rhs_slice0']),
            make_node('Slice', [name+'_rhs_shape', name+'_-2', name+'_100'],
                      [name+'_rhs_slice1']),
            make_node('Concat', [name+'_-1', name+'_rhs_slice1'], [name+'_rhs_concat1'], axis=0),
            make_node('Reshape', [rhs, name+'_rhs_concat1'], [name+'_rhs_3d']),
            make_node('Transpose', [name+'_rhs_3d'], [name+'_rhs_3d_transpose'], perm=perm),
            make_node('Shape', [name+'_rhs_3d_transpose'], [name+'_rhs_shape_3d']),
            make_node('Slice', [name+'_rhs_shape_3d', name+'_-2', name+'_100'],
                      [name+'_rhs_slice2']),
            make_node('Concat', [name+'_rhs_slice0', name+'_rhs_slice2'], [name+'_rhs_concat2'], axis=0),
            make_node('Reshape', [name+'_rhs_3d_transpose', name+'_rhs_concat2'], [name+'_rhs']),
            make_node('MatMul', [lhs, name+'_rhs'], [name]),
        ]

    else:
        nodes += [
            make_node('Shape', [lhs], [name+'_lhs_shape']),
            make_node('Shape', [name+'_lhs_shape'], [name+'_lhs_dim']),
            make_node('Slice', [name+'_lhs_shape', name+'_0', name+'_-2'],
                      [name+'_lhs_slice0']),
            make_node('Slice', [name+'_lhs_shape', name+'_-2', name+'_100'],
                      [name+'_lhs_slice1']),
            make_node('Concat', [name+'_-1', name+'_lhs_slice1'], [name+'_lhs_concat1'], axis=0),
            make_node('Reshape', [lhs, name+'_lhs_concat1'], [name+'_lhs_3d']),
            make_node('Transpose', [name+'_lhs_3d'], [name+'_lhs_3d_transpose'], perm=perm),
            make_node('Shape', [name+'_lhs_3d_transpose'], [name+'_lhs_shape_3d']),
            make_node('Slice', [name+'_lhs_shape_3d', name+'_-2', name+'_100'],
                      [name+'_lhs_slice2']),
            make_node('Concat', [name+'_lhs_slice0', name+'_lhs_slice2'], [name+'_lhs_concat2'], axis=0),
            make_node('Reshape', [name+'_lhs_3d_transpose', name+'_lhs_concat2'], [name+'_lhs']),

            make_node('Shape', [rhs], [name+'_rhs_shape']),
            make_node('Shape', [name+'_rhs_shape'], [name+'_rhs_dim']),
            make_node('Slice', [name+'_rhs_shape', name+'_0', name+'_-2'],
                      [name+'_rhs_slice0']),
            make_node('Slice', [name+'_rhs_shape', name+'_-2', name+'_100'],
                      [name+'_rhs_slice1']),
            make_node('Concat', [name+'_-1', name+'_rhs_slice1'], [name+'_rhs_concat1'], axis=0),
            make_node('Reshape', [rhs, name+'_rhs_concat1'], [name+'_rhs_3d']),
            make_node('Transpose', [name+'_rhs_3d'], [name+'_rhs_3d_transpose'], perm=perm),
            make_node('Shape', [name+'_rhs_3d_transpose'], [name+'_rhs_shape_3d']),
            make_node('Slice', [name+'_rhs_shape_3d', name+'_-2', name+'_100'],
                      [name+'_rhs_slice2']),
            make_node('Concat', [name+'_rhs_slice0', name+'_rhs_slice2'], [name+'_rhs_concat2'], axis=0),
            make_node('Reshape', [name+'_rhs_3d_transpose', name+'_rhs_concat2'], [name+'_rhs']),
            make_node('MatMul', [name+'_lhs', name+'_rhs'], [name]),
        ]

    return nodes


@mx_op.register("log2")
def convert_log2(node, **kwargs):
    """Map MXNet's log2 operator attributes to onnx's operator.
    """
    from onnx.helper import make_node, make_tensor
    name, input_nodes, _ = get_inputs(node, kwargs)

    input_type = kwargs["in_type"]
    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type]

    ln2 = np.array([0.693147180559945309], dtype=dtype)
    if dtype == 'float16':
        ln2 = ln2.view(dtype=np.uint16)
    ln2v = make_tensor(name+'_ln2', input_type, [1], ln2)

    nodes = [
        make_node('Log', [input_nodes[0]], [name+'_log']),
        make_node('Constant', [], [name+'_ln2'], value=ln2v),
        make_node('Div', [name+'_log', name+'_ln2'], [name], name=name)
    ]

    return nodes


@mx_op.register('argsort')
def convert_argsort(node, **kwargs):
    """Map MXNet's argsort operator attributes to onnx's TopK operator
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')

    axis = int(attrs.get('axis', '-1'))
    is_ascend = attrs.get('is_ascend', 'True')
    is_ascend = is_ascend in ['True', '1']
    dtype = attrs.get('dtype', 'float32')

    create_tensor([axis], name+'_axis', kwargs['initializer'])
    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_shape']),
        make_node('Gather', [name+'_shape', name+'_axis'], [name+'_k'])
    ]
    if dtype == 'int64':
        nodes += [
            make_node('TopK', [input_nodes[0], name+'_k'], [name+'_', name], axis=axis,
                      largest=(not is_ascend), sorted=1),
        ]
    else:
        nodes += [
            make_node('TopK', [input_nodes[0], name+'_k'], [name+'_', name+'_temp'], axis=axis,
                      largest=(not is_ascend), sorted=1),
            make_node('Cast', [name+'_temp'], [name],
                      to=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)])
        ]

    return nodes


@mx_op.register('SequenceReverse')
def convert_sequece_reverse(node, **kwargs):
    """Map MXNet's SequenceReverse op
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    batch_axis = 1
    time_axis = 0
    use_sequence_length = attrs.get('use_sequence_length', 'False')

    nodes = []
    if use_sequence_length == 'False':
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('Split', [name+'_shape'], [name+'_dim0', name+'_dim1', name+'_dim2']),
            make_node('Expand', [name+'_dim0', name+'_dim1'], [name+'_seq_len']),
            make_node('ReverseSequence', [input_nodes[0], name+'_seq_len'], [name],
                      batch_axis=batch_axis, time_axis=time_axis)
        ]
    else:
        nodes += [
            make_node('Cast', [input_nodes[1]], [name+'_seq_len'], to=int(TensorProto.INT64)),
            make_node('ReverseSequence', [input_nodes[0], name+'_seq_len'], [name],
                      batch_axis=batch_axis, time_axis=time_axis)
        ]

    return nodes

