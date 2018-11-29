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
"""
Conversion Functions for common layers.
Add new functions here with a decorator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
    tuple_re = re.compile('\([0-9L|,| ]+\)')
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
        if val not in ("", "None"):
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
    proc_nodes = kwargs["proc_nodes"]
    index_lookup = kwargs["index_lookup"]
    inputs = node["inputs"]
    attrs = node.get("attrs", {})

    input_nodes = []
    for ip in inputs:
        input_node_id = index_lookup[ip[0]]
        input_nodes.append(proc_nodes[input_node_id].name)

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
                raw=False,
            )
        )

        return [tensor_node]
    else:
        tval_node = onnx.helper.make_tensor_value_info(name, kwargs["in_type"], kwargs["in_shape"])
        return [tval_node]


@mx_op.register("Convolution")
def convert_convolution(node, **kwargs):
    """Map MXNet's convolution operator attributes to onnx's Conv operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    kernel_dims = list(parse_helper(attrs, "kernel"))
    stride_dims = list(parse_helper(attrs, "stride", [1, 1]))
    pad_dims = list(parse_helper(attrs, "pad", [0, 0]))
    num_group = int(attrs.get("num_group", 1))
    dilations = list(parse_helper(attrs, "dilate", [1, 1]))

    pad_dims = pad_dims + pad_dims

    conv_node = onnx.helper.make_node(
        "Conv",
        inputs=input_nodes,
        outputs=[name],
        kernel_shape=kernel_dims,
        strides=stride_dims,
        dilations=dilations,
        pads=pad_dims,
        group=num_group,
        name=name
    )

    return [conv_node]


@mx_op.register("FullyConnected")
def convert_fully_connected(node, **kwargs):
    """Map MXNet's FullyConnected operator attributes to onnx's Gemm operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    initializer = kwargs["initializer"]

    no_bias = get_boolean_attribute_value(attrs, "no_bias")

    fcnode = []

    if no_bias:
        data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')]
        bias_name = "bias" + str(kwargs["idx"])
        tensor_node = onnx.helper.make_tensor_value_info(bias_name, data_type, (1,))
        initializer.append(
            onnx.helper.make_tensor(
                name=bias_name,
                data_type=data_type,
                dims=(1,),
                vals=[0],
                raw=False,
            )
        )
        input_nodes.append(bias_name)
        fcnode.append(tensor_node)

    node = onnx.helper.make_node(
        "Gemm",
        input_nodes,  # input (A, B, C) - C can be in place
        [name],  # output
        alpha=1.0,
        beta=1.0,
        transA=False,
        transB=True,
        name=name
    )

    fcnode.append(node)

    return fcnode


@mx_op.register("BatchNorm")
def convert_batchnorm(node, **kwargs):
    """Map MXNet's BatchNorm operator attributes to onnx's BatchNormalization operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    momentum = float(attrs.get("momentum", 0.9))
    eps = float(attrs.get("eps", 0.001))

    bn_node = onnx.helper.make_node(
        "BatchNormalization",
        input_nodes,
        [name],
        name=name,
        epsilon=eps,
        momentum=momentum,
        # MXNet computes mean and variance per feature for batchnorm
        # Default for onnx is across all spatial features. So disabling the parameter.
        spatial=0
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
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mxnet_pad_width = convert_string_to_list(attrs.get("pad_width"))
    onnx_pad_width = transform_padding(mxnet_pad_width)

    pad_mode = attrs.get("mode")

    if pad_mode == "constant":
        pad_value = float(attrs.get("constant_value")) \
            if "constant_value" in attrs else 0.0
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


def create_helper_trans_node(op_name, input_node, node_name):
    """create extra transpose node for dot operator"""
    node_name = op_name + "_" + node_name
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

    trans_a_node = None
    trans_b_node = None

    trans_a = get_boolean_attribute_value(attrs, "transpose_a")
    trans_b = get_boolean_attribute_value(attrs, "transpose_b")

    op_name = "transpose" + str(kwargs["idx"])

    if trans_a:
        trans_a_node = create_helper_trans_node(op_name, input_nodes[0], 'a')
        input_node_a = op_name+"_a"
    if trans_b:
        trans_b_node = create_helper_trans_node(op_name, input_nodes[1], 'b')
        input_node_b = op_name+"_b"

    matmul_node = onnx.helper.make_node(
        'MatMul',
        inputs=[input_node_a, input_node_b],
        outputs=[name],
        name=name
    )

    if not trans_a and not trans_b:
        return [matmul_node]
    elif trans_a and not trans_b:
        return [trans_a_node, matmul_node]
    elif trans_b and not trans_a:
        return [trans_b_node, matmul_node]
    else:
        return [trans_a_node, trans_b_node, matmul_node]


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


@mx_op.register("Pooling")
def convert_pooling(node, **kwargs):
    """Map MXNet's Pooling operator attributes to onnx's
    MaxPool/AveragePool/GlobalMaxPool/GlobalAveragePool operators
    based on the input node's attributes and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    kernel = eval(attrs["kernel"])
    pool_type = attrs["pool_type"]
    stride = eval(attrs["stride"]) if attrs.get("stride") else None
    global_pool = get_boolean_attribute_value(attrs, "global_pool")

    pooling_convention = attrs.get('pooling_convention', 'valid')

    if pooling_convention == 'full':
        pooling_warning = "Pooling: ONNX currently doesn't support pooling_convention. " \
                          "This might lead to shape or accuracy issues. " \
                          "https://github.com/onnx/onnx/issues/549"

        logging.warning(pooling_warning)

    pad_dims = list(parse_helper(attrs, "pad", [0, 0]))
    pad_dims = pad_dims + pad_dims
    pool_types = {"max": "MaxPool", "avg": "AveragePool"}
    global_pool_types = {"max": "GlobalMaxPool", "avg": "GlobalAveragePool"}

    if global_pool:
        node = onnx.helper.make_node(
            global_pool_types[pool_type],
            input_nodes,  # input
            [name],
            name=name
        )
    else:
        node = onnx.helper.make_node(
            pool_types[pool_type],
            input_nodes,  # input
            [name],
            kernel_shape=kernel,
            pads=pad_dims,
            strides=stride,
            name=name
        )

    return [node]


@mx_op.register("exp")
def convert_exp(node, **kwargs):
    """Map MXNet's exp operator attributes to onnx's Exp operator
    and return the created node.
    """
    return create_basic_op_node('Exp', node, kwargs)

@mx_op.register("_copy")
def convert_identity(node, **kwargs):
    """Map MXNet's _copy operator attributes to onnx's Identity operator
    and return the created node.
    """
    return create_basic_op_node('Identity', node, kwargs)


@mx_op.register("LeakyReLU")
def convert_leakyrelu(node, **kwargs):
    """Map MXNet's LeakyReLU operator attributes to onnx's Elu/LeakyRelu/PRelu operators
    based on the input node's attributes and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    act_type = attrs.get("act_type", "leaky")
    alpha = float(attrs.get("slope", 0.25))

    act_name = {"elu": "Elu", "leaky": "LeakyRelu", "prelu": "PRelu",
                "selu": "Selu"}

    if act_type == "prelu" or act_type == "selu":
        node = onnx.helper.make_node(
            act_name[act_type],
            inputs=input_nodes,
            outputs=[name],
            name=name)
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
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("axis", -1))

    softmax_node = onnx.helper.make_node(
        "Softmax",
        input_nodes,
        [name],
        axis=axis,
        name=name
    )

    return [softmax_node]


# There's also mx.sym.softmax(), which doesn't do cross-entropy loss,
# just softmax for inference - hence the name convert_softmax_output.
@mx_op.register("SoftmaxOutput")
def convert_softmax_output(node, **kwargs):
    """Map MXNet's SoftmaxOutput operator attributes to onnx's Softmax operator
    and return the created node.
    """
    name = node["name"]

    input1_idx = kwargs["index_lookup"][node["inputs"][0][0]]
    input1 = kwargs["proc_nodes"][input1_idx]

    softmax_node = onnx.helper.make_node(
        "Softmax",
        [input1.output[0]],
        [name],
        axis=1,
        name=name
    )

    return [softmax_node]


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
        raise AttributeError("ONNX currently supports channel mode only")

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
    name, input_nodes, attrs = get_inputs(node, kwargs)

    probability = float(attrs.get("p", 0.5))

    dropout_node = onnx.helper.make_node(
        "Dropout",
        input_nodes,
        [name],
        ratio=probability,
        name=name
    )
    return [dropout_node]


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
    name, input_nodes, attrs = get_inputs(node, kwargs)

    a_min = np.float(attrs.get('a_min', -np.inf))
    a_max = np.float(attrs.get('a_max', np.inf))

    clip_node = onnx.helper.make_node(
        "Clip",
        input_nodes,
        [name],
        name=name,
        min=a_min,
        max=a_max
    )
    return [clip_node]


def scalar_op_helper(node, op_name, **kwargs):
    """Helper function for scalar arithmetic operations"""
    name, input_nodes, attrs = get_inputs(node, kwargs)

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
                new_initializer = onnx.numpy_helper.to_array(i) * scalar_value[0]
            elif op_name == 'Sub':
                new_initializer = onnx.numpy_helper.to_array(i) - scalar_value[0]
            elif op_name == 'Add':
                new_initializer = onnx.numpy_helper.to_array(i) + scalar_value[0]
            elif op_name == 'Div':
                new_initializer = onnx.numpy_helper.to_array(i) / scalar_value[0]
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
    name, input_nodes, attrs = get_inputs(node, kwargs)

    output_shape_list = convert_string_to_list(attrs["shape"])

    initializer = kwargs["initializer"]
    output_shape_np = np.array(output_shape_list, dtype='int64')
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[output_shape_np.dtype]
    dims = np.shape(output_shape_np)

    output_shape_name = "reshape_attr_tensor" + str(kwargs["idx"])
    tensor_node = onnx.helper.make_tensor_value_info(output_shape_name, data_type, dims)

    initializer.append(
        onnx.helper.make_tensor(
            name=output_shape_name,
            data_type=data_type,
            dims=dims,
            vals=output_shape_list,
            raw=False,
        )
    )

    input_nodes.append(output_shape_name)

    not_supported_shape = [-2, -3, -4]

    for val in output_shape_list:
        if val in not_supported_shape:
            raise AttributeError("Shape value not supported in ONNX", val)

    reshape_node = onnx.helper.make_node(
        "Reshape",
        input_nodes,
        [name],
        name=name
    )

    return [tensor_node, reshape_node]

@mx_op.register("Cast")
def convert_cast(node, **kwargs):
    """Map MXNet's Cast operator attributes to onnx's Cast operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    dtype = attrs["dtype"]

    # dtype can be mapped only with types from TensorProto
    # float32 is mapped to float and float64 to double in onnx
    # following tensorproto mapping https://github.com/onnx/onnx/blob/master/onnx/mapping.py
    if dtype == 'float32':
        dtype = 'float'
    elif dtype == 'float64':
        dtype = 'double'

    node = onnx.helper.make_node(
        "Cast",
        input_nodes,
        [name],
        to=getattr(onnx.TensorProto, dtype.upper()),
        name=name,
    )
    return [node]


@mx_op.register("slice_axis")
def convert_slice_axis(node, **kwargs):
    """Map MXNet's slice_axis operator attributes to onnx's Slice operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axes = int(attrs.get("axis"))
    starts = int(attrs.get("begin"))
    ends = int(attrs.get("end", None))
    if not ends:
        raise ValueError("Slice: ONNX doesnt't support 'None' in 'end' attribute")

    node = onnx.helper.make_node(
        "Slice",
        input_nodes,
        [name],
        axes=[axes],
        starts=[starts],
        ends=[ends],
        name=name,
    )
    return [node]


@mx_op.register("SliceChannel")
def convert_slice_channel(node, **kwargs):
    """Map MXNet's SliceChannel operator attributes to onnx's Squeeze or Split
    operator based on squeeze_axis attribute
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    num_outputs = int(attrs.get("num_outputs"))
    axis = int(attrs.get("axis", 1))
    squeeze_axis = int(attrs.get("squeeze_axis", 0))

    if squeeze_axis == 1 and num_outputs == 1:
        node = onnx.helper.make_node(
            "Squeeze",
            input_nodes,
            [name],
            axes=[axis],
            name=name,
        )
        return [node]
    elif squeeze_axis == 0 and num_outputs > 1:
        node = onnx.helper.make_node(
            "Split",
            input_nodes,
            [name],
            axis=axis,
            split=[num_outputs],
            name=name,
        )
        return [node]
    else:
        raise NotImplementedError("SliceChannel operator with num_outputs>1 and"
                                  "squeeze_axis true is not implemented.")


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
        raise AttributeError("Missing axis attribute: ONNX currently requires axis to "
                             "be specified for squeeze operator")
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
    return create_basic_op_node('Equal', node, kwargs)
