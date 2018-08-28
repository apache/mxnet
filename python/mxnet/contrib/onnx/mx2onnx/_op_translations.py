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

def import_onnx_modules():
    """ To make sure ONNX is runtime dependency, it is imported used only when needed"""
    try:
        from onnx import helper, numpy_helper, mapping
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")
    return helper, numpy_helper, mapping


def parse_helper(attrs, attrs_name, alt_value=None):
    """Helper function to parse operator attributes in required format."""
    tuple_re = re.compile('\([0-9L|,| ]+\)')
    if attrs is None:
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

@mx_op.register("null")
def convert_weights_and_inputs(node, **kwargs):
    """Helper function to convert weights and inputs.
    """

    helper, _, mapping = import_onnx_modules()
    name = node["name"]

    if kwargs["is_input"] is False:
        weights = kwargs["weights"]
        initializer = kwargs["initializer"]
        np_arr = weights[name]
        data_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np_arr.dtype]
        dims = np.shape(np_arr)

        tensor_node = helper.make_tensor_value_info(name, data_type, dims)

        initializer.append(
            helper.make_tensor(
                name=name,
                data_type=data_type,
                dims=dims,
                vals=np_arr.flatten().tolist(),
                raw=False,
            )
        )

        return [tensor_node]
    else:
        tval_node = helper.make_tensor_value_info(name, kwargs["in_type"], kwargs["in_shape"])
        return [tval_node]


@mx_op.register("Convolution")
def convert_convolution(node, **kwargs):
    """Map MXNet's convolution operator attributes to onnx's Conv operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    inputs = node["inputs"]

    num_inputs = len(inputs)

    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[kwargs["index_lookup"][inputs[0][0]]].name
    weights_node = proc_nodes[kwargs["index_lookup"][inputs[1][0]]].name

    if num_inputs > 2:
        bias_node = proc_nodes[kwargs["index_lookup"][inputs[2][0]]].name

    attrs = node.get("attrs")

    kernel_dims = list(parse_helper(attrs, "kernel"))
    stride_dims = list(parse_helper(attrs, "stride", [1, 1]))
    pad_dims = list(parse_helper(attrs, "pad", [0, 0]))
    num_group = int(attrs.get("num_group", 1))
    dilations = list(parse_helper(attrs, "dilate", [1, 1]))

    pad_dims = pad_dims + pad_dims

    input_nodes = [input_node, weights_node]
    if num_inputs > 2:
        input_nodes.append(bias_node)

    conv_node = helper.make_node(
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    inputs = node["inputs"]
    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    weight_node_id = kwargs["index_lookup"][inputs[1][0]]
    bias_node_id = kwargs["index_lookup"][inputs[2][0]]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_node_id]
    weights_node = proc_nodes[weight_node_id]
    bias_node = proc_nodes[bias_node_id]

    input_name = input_node.name
    weights_name = weights_node.name
    bias_name = bias_node.name

    node = helper.make_node(
        "Gemm",
        [input_name, weights_name, bias_name],  # input (A, B, C) - C can be in place
        [name],  # output
        alpha=1.0,
        beta=1.0,
        transA=False,
        transB=True,
        name=name
    )

    return [node]


@mx_op.register("BatchNorm")
def convert_batchnorm(node, **kwargs):
    """Map MXNet's BatchNorm operator attributes to onnx's BatchNormalization operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    attrs = node["attrs"]
    momentum = float(node.get("attrs", {}).get("momentum", 0.9))
    eps = float(attrs.get("eps", 0.001))

    data_idx = kwargs["index_lookup"][inputs[0][0]]
    gamma_idx = kwargs["index_lookup"][inputs[1][0]]
    beta_idx = kwargs["index_lookup"][inputs[2][0]]
    moving_mean_idx = kwargs["index_lookup"][inputs[3][0]]
    moving_var_idx = kwargs["index_lookup"][inputs[4][0]]

    data_node = proc_nodes[data_idx].name
    gamma_node = proc_nodes[gamma_idx].name
    beta_node = proc_nodes[beta_idx].name

    mov_mean_node = proc_nodes[moving_mean_idx]
    mov_mean_node = mov_mean_node.name
    mov_var_node = proc_nodes[moving_var_idx].name

    bn_node = helper.make_node(
        "BatchNormalization",
        [data_node,
         gamma_node,  # scale
         beta_node,  # bias
         mov_mean_node,
         mov_var_node
        ],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    inputs = node["inputs"]
    input_node_idx = kwargs["index_lookup"][inputs[0][0]]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_node_idx].name

    node = helper.make_node(
        'Tanh',
        [input_node],
        [name],
        name=name
    )
    return [node]

#Basic neural network functions
@mx_op.register("sigmoid")
def convert_sigmoid(node, **kwargs):
    """Map MXNet's sigmoid operator attributes to onnx's Sigmoid operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    inputs = node["inputs"]
    input_node_idx = kwargs["index_lookup"][inputs[0][0]]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_node_idx].name

    node = helper.make_node(
        'Sigmoid',
        [input_node],
        [name],
        name=name
    )
    return [node]

@mx_op.register("relu")
def convert_relu(node, **kwargs):
    """Map MXNet's relu operator attributes to onnx's Relu operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    inputs = node["inputs"]
    input_node_idx = kwargs["index_lookup"][inputs[0][0]]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_node_idx].name

    node = helper.make_node(
        'Relu',
        [input_node],
        [name],
        name=name
    )

    return [node]

@mx_op.register("Activation")
def convert_activation(node, **kwargs):
    """Map MXNet's Activation operator attributes to onnx's Tanh/Relu operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]

    proc_nodes = kwargs["proc_nodes"]
    attrs = node["attrs"]
    act_type = attrs["act_type"]

    inputs = node["inputs"]
    input_node_idx = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_idx].output[0]

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
        node = helper.make_node(
            act_name,
            [input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    attrs = node["attrs"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    input_node_idx = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_idx].name

    mxnet_pad_width = convert_string_to_list(attrs.get("pad_width"))
    onnx_pad_width = transform_padding(mxnet_pad_width)

    pad_mode = attrs.get("mode")

    if pad_mode == "constant":
        pad_value = float(attrs.get("constant_value")) \
            if "constant_value" in attrs else 0.0
        node = helper.make_node(
            'Pad',
            inputs=[input_node],
            outputs=[name],
            mode='constant',
            value=pad_value,
            pads=onnx_pad_width,
            name=name
        )
    else:
        node = helper.make_node(
            'Pad',
            inputs=[input_node],
            outputs=[name],
            mode=pad_mode,
            pads=onnx_pad_width,
            name=name
        )

    return [node]


def create_helper_trans_node(op_name, input_node, node_name):
    """create extra transpose node for dot operator"""
    helper, _, _ = import_onnx_modules()

    node_name = op_name + "_" + node_name
    trans_node = helper.make_node(
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
    helper, _, _ = import_onnx_modules()
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]
    name = node["name"]

    input_a_idx = kwargs["index_lookup"][node_inputs[0][0]]
    input_node_a = proc_nodes[input_a_idx].name
    input_b_idx = kwargs["index_lookup"][node_inputs[1][0]]
    input_node_b = proc_nodes[input_b_idx].name

    attrs = node.get('attrs', {})
    trans_a_node = None
    trans_b_node = None

    trans_a = 1 if ("transpose_a" in attrs) and \
                   attrs.get("transpose_a") in ["True", "1"] else 0
    trans_b = 1 if ("transpose_b" in attrs) and \
                   attrs.get("transpose_b") in ["True", "1"] else 0

    op_name = "transpose" + str(kwargs["idx"])
    create_helper_trans_node(op_name, input_node_a, 'a')
    create_helper_trans_node(op_name, input_node_b, 'b')

    if trans_a:
        trans_a_node = create_helper_trans_node(op_name, input_node_a, 'a')
        input_node_a = op_name+"_a"
    if trans_b:
        trans_b_node = create_helper_trans_node(op_name, input_node_b, 'b')
        input_node_b = op_name+"_b"

    matmul_node = helper.make_node(
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
    helper, _, _ = import_onnx_modules()
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]
    name = node["name"]

    input_a_idx = kwargs["index_lookup"][node_inputs[0][0]]
    input_node_a = proc_nodes[input_a_idx].name
    input_b_idx = kwargs["index_lookup"][node_inputs[1][0]]
    input_node_b = proc_nodes[input_b_idx].name

    # Getting the attributes and assigning default values.
    if "attrs" in node:
        attrs = node["attrs"]
        alpha = float(attrs["alpha"])
        trans_a = int(attrs["transpose_a"])
        trans_b = int(attrs["transpose_b"])
    else:
        alpha = 1.0
        trans_a = 0
        trans_b = 0

    op_name = "transpose" + str(kwargs["idx"])

    if alpha == 1.0 and trans_a == 0 and trans_b == 0:
        matmul_node = helper.make_node(
            'MatMul',
            inputs=[input_node_a, input_node_b],
            outputs=[name],
            name=name
        )
        return [matmul_node]
    elif trans_a == 1 and trans_b == 0:
        op_name = "transpose" + str(kwargs["idx"])
        node_name = op_name+"_a"
        trans_a_node = helper.make_node(
            'Transpose',
            inputs=[input_node_a],
            outputs=[op_name+"_a"],
            name=node_name
        )

        matmul_node = helper.make_node(
            'MatMul',
            inputs=[node_name, input_node_b],
            outputs=[name],
            name=name
        )
        return [trans_a_node, matmul_node]

    elif trans_a == 0 and trans_b == 1:
        node_name = op_name + "_b"
        trans_b_node = helper.make_node(
            'Transpose',
            inputs=[input_node_b],
            outputs=[op_name+"_b"],
            name=node_name
        )

        matmul_node = helper.make_node(
            'MatMul',
            inputs=[input_node_a, node_name],
            outputs=[name],
            name=name
        )

        return [trans_b_node, matmul_node]
    else:
        node_name_a = op_name+"_a"
        trans_a_node = helper.make_node(
            'Transpose',
            inputs=[input_node_a],
            outputs=[op_name+"_a"],
            name=node_name_a
        )

        node_name_b = op_name + "_b"
        trans_b_node = helper.make_node(
            'Transpose',
            inputs=[input_node_b],
            outputs=[op_name+"_b"],
            name=node_name_b
        )

        matmul_node = helper.make_node(
            'MatMul',
            inputs=[node_name_a, node_name_b],
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
    helper, _, _ = import_onnx_modules()
    proc_nodes = kwargs["proc_nodes"]
    attrs = node["attrs"]
    kernel = eval(attrs["kernel"])
    pool_type = attrs["pool_type"]
    stride = eval(attrs["stride"]) if attrs.get("stride") else None
    global_pool = True if "global_pool" in attrs and\
                          attrs.get("global_pool") == "True" else False
    node_inputs = node["inputs"]
    input_node_idx = kwargs["index_lookup"][node_inputs[0][0]]
    input_node = proc_nodes[input_node_idx]
    name = node["name"]

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
        node = helper.make_node(
            global_pool_types[pool_type],
            [input_node.name],  # input
            [name],
            name=name
        )
    else:
        node = helper.make_node(
            pool_types[pool_type],
            [input_node.name],  # input
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Exp",
        [input_node],
        [name],
        name=name,
    )
    return [node]


@mx_op.register("_copy")
def convert_identity(node, **kwargs):
    """Map MXNet's _copy operator attributes to onnx's Identity operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Identity",
        [input_node],
        [name],
        name=name,
    )
    return [node]


@mx_op.register("LeakyReLU")
def convert_leakyrelu(node, **kwargs):
    """Map MXNet's LeakyReLU operator attributes to onnx's Elu/LeakyRelu/PRelu operators
    based on the input node's attributes and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name
    attrs = node["attrs"]

    act_type = attrs.get("act_type", "leaky")
    alpha = float(attrs.get("slope", 0.25))

    act_name = {"elu": "Elu", "leaky": "LeakyRelu", "prelu": "PRelu"}

    if act_type == "prelu":
        alpha_node_index = kwargs["index_lookup"][inputs[1][0]]
        alpha_node_name = proc_nodes[alpha_node_index].name

        node = helper.make_node(
            act_name[act_type],
            inputs=[input_node, alpha_node_name],
            outputs=[name],
            name=name)
    else:
        node = helper.make_node(
            act_name[act_type],
            inputs=[input_node],
            outputs=[name],
            name=name,
            alpha=alpha)

    return [node]


@mx_op.register("softmax")
def convert_softmax(node, **kwargs):
    """Map MXNet's softmax operator attributes to onnx's Softmax operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    inputs = node["inputs"]
    input_idx = kwargs["index_lookup"][inputs[0][0]]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_idx]

    name = node["name"]
    axis = int(node.get("attrs", {}).get("axis", -1))

    softmax_node = helper.make_node(
        "Softmax",
        [input_node.name],
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
    helper, _, _ = import_onnx_modules()
    inputs = node["inputs"]
    input1_idx = kwargs["index_lookup"][inputs[0][0]]
    proc_nodes = kwargs["proc_nodes"]
    input1 = proc_nodes[input1_idx]
    name = node["name"]

    softmax_node = helper.make_node(
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    inputs = node["inputs"]
    proc_nodes = kwargs["proc_nodes"]
    input_names = [proc_nodes[kwargs["index_lookup"][i[0]]].name for i in inputs]
    axis = int(node.get("attrs", {}).get("dim", 1))
    concat_node = helper.make_node(
        "Concat",
        input_names,
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    input_idx = kwargs["index_lookup"][node["inputs"][0][0]]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_idx].name
    axes = node.get("attrs", {}).get("axes", ())
    if axes:
        axes = tuple(map(int, re.findall(r'\d+', axes)))

        transpose_node = helper.make_node(
            "Transpose",
            [input_node],
            [name],
            perm=axes,
            name=name
        )
    else:
        transpose_node = helper.make_node(
            "Transpose",
            [input_node],
            [name],
            name=name
        )

    return [transpose_node]


@mx_op.register("LRN")
def convert_lrn(node, **kwargs):
    """Map MXNet's LRN operator attributes to onnx's LRN operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    input_idx = kwargs["index_lookup"][node["inputs"][0][0]]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_idx].name

    attrs = node["attrs"]
    alpha = float(attrs["alpha"]) if "alpha" in attrs else 0.0001
    beta = float(attrs["beta"]) if "beta" in attrs else 0.75
    bias = float(attrs["knorm"]) if "knorm" in attrs else 1.0
    size = int(attrs["nsize"])

    lrn_node = helper.make_node(
        "LRN",
        inputs=[input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    input_id = kwargs["index_lookup"][node["inputs"][0][0]]
    input_name = kwargs["proc_nodes"][input_id].name
    attrs = node["attrs"]
    mode = attrs.get("mode", "instance")

    if mode != "channel":
        raise AttributeError("ONNX currently supports channel mode only")

    l2norm_node = helper.make_node(
        "LpNormalization",
        [input_name],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    input_id = kwargs["index_lookup"][node["inputs"][0][0]]
    input_name = kwargs["proc_nodes"][input_id].name
    attrs = node["attrs"]
    probability = float(attrs["p"])

    dropout_node = helper.make_node(
        "Dropout",
        [input_name],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    input_idx = kwargs["index_lookup"][node["inputs"][0][0]]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_idx].name  # .output[0]

    flatten_node = helper.make_node(
        "Flatten",
        [input_node],
        [name],
        name=name
    )
    return [flatten_node]


def scalar_op_helper(node, op_name, **kwargs):
    """Helper function for scalar arithmetic operations"""
    helper, numpy_helper, mapping = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    scalar_value = [float(node.get("attrs", {}).get("scalar", 1))]

    input_name_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_name_id].name

    initializer = kwargs["initializer"]
    flag = True
    # If the input value is in initializer, just multiply with scalar input
    # and create a new initializer
    for i in initializer:
        if i.name == input_node:
            if op_name == 'Mul':
                new_initializer = numpy_helper.to_array(i) * scalar_value[0]
            elif op_name == 'Sub':
                new_initializer = numpy_helper.to_array(i) - scalar_value[0]
            elif op_name == 'Add':
                new_initializer = numpy_helper.to_array(i) + scalar_value[0]
            elif op_name == 'Div':
                new_initializer = numpy_helper.to_array(i) / scalar_value[0]
            flag = False
            break

    # else create a new tensor of the scalar value, add it in initializer
    if flag is True:
        np_arr = np.array(scalar_value)
        data_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np_arr.dtype]
        dims = np.shape(np_arr)

        scalar_op_name = "scalar_op" + str(kwargs["idx"])
        tensor_node = helper.make_tensor_value_info(scalar_op_name, data_type, dims)

        initializer.append(
            helper.make_tensor(
                name=scalar_op_name,
                data_type=data_type,
                dims=dims,
                vals=scalar_value,
                raw=False,
            )
        )

        mul_node = helper.make_node(
            op_name,
            [input_node, scalar_op_name],
            [name],
            name=name
        )

        return [tensor_node, mul_node]
    else:
        data_type = mapping.NP_TYPE_TO_TENSOR_TYPE[new_initializer.dtype]
        dims = np.shape(new_initializer)

        new_a_node = input_node + str(kwargs["idx"])
        tensor_node = helper.make_tensor_value_info(new_a_node, data_type, dims)

        initializer.append(
            helper.make_tensor(
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
    helper, _, _ = import_onnx_modules()
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]

    input_node_idx = kwargs["index_lookup"][node_inputs[0][0]]
    input_node = proc_nodes[input_node_idx].name
    name = node["name"]
    attrs = node["attrs"]

    axis = int(attrs.get("axis"))
    keepdims = int(attrs.get("keepdims")) if "keepdims" in attrs  else 1

    node = helper.make_node(
        'ArgMax',
        inputs=[input_node],
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
    helper, _, _ = import_onnx_modules()
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]

    input_node_idx = kwargs["index_lookup"][node_inputs[0][0]]
    input_node = proc_nodes[input_node_idx].name
    name = node["name"]
    attrs = node["attrs"]

    axis = int(attrs.get("axis"))
    keepdims = int(attrs.get("keepdims")) if "keepdims" in attrs  else 1

    node = helper.make_node(
        'ArgMin',
        inputs=[input_node],
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
    helper, _, _ = import_onnx_modules()
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]

    input_node_list = []
    for node_input in node_inputs:
        node_id = kwargs["index_lookup"][node_input[0]]
        input_node_list.append(proc_nodes[node_id].name)

    name = node["name"]

    node = helper.make_node(
        'Max',
        inputs=input_node_list,
        outputs=[name],
        name=name,
    )

    return [node]


@mx_op.register("_minimum")
def convert_minimum(node, **kwargs):
    """Map MXNet's _minimum operator attributes to onnx's Min operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]

    input_node_list = []
    for node_input in node_inputs:
        node_id = kwargs["index_lookup"][node_input[0]]
        input_node_list.append(proc_nodes[node_id].name)

    name = node["name"]

    node = helper.make_node(
        'Min',
        inputs=input_node_list,
        outputs=[name],
        name=name,
    )

    return [node]


@mx_op.register("min")
def convert_min(node, **kwargs):
    """Map MXNet's min operator attributes to onnx's ReduceMin operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    mx_axis = node.get("attrs", {}).get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None

    keepdims = int(node.get("attrs", {}).get("keepdims", 0))

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    if axes is not None:
        node = helper.make_node(
            'ReduceMin',
            inputs=[input_node],
            outputs=[name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )

        return [node]
    else:
        node = helper.make_node(
            'ReduceMin',
            inputs=[input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    mx_axis = node.get("attrs", {}).get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None

    keepdims = int(node.get("attrs", {}).get("keepdims", 0))

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    if axes is not None:
        node = helper.make_node(
            'ReduceMax',
            inputs=[input_node],
            outputs=[name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )

        return [node]
    else:
        node = helper.make_node(
            'ReduceMax',
            inputs=[input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    mx_axis = node.get("attrs", {}).get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None

    keepdims = int(node.get("attrs", {}).get("keepdims", 0))

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    if axes is not None:
        node = helper.make_node(
            'ReduceMean',
            inputs=[input_node],
            outputs=[name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )

        return [node]
    else:
        node = helper.make_node(
            'ReduceMean',
            inputs=[input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    mx_axis = node.get("attrs", {}).get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis is not None else None

    keepdims = int(node.get("attrs", {}).get("keepdims", 0))

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    if axes is not None:
        node = helper.make_node(
            'ReduceProd',
            inputs=[input_node],
            outputs=[name],
            axes=axes,
            keepdims=keepdims,
            name=name
        )

        return [node]
    else:
        node = helper.make_node(
            'ReduceProd',
            inputs=[input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_a_id = kwargs["index_lookup"][inputs[0][0]]
    input_node_b_id = kwargs["index_lookup"][inputs[1][0]]

    input_node_a = proc_nodes[input_node_a_id].name
    input_node_b = proc_nodes[input_node_b_id].name

    add_node = helper.make_node(
        "Add",
        [input_node_a, input_node_b],
        [name],
        name=name,
    )

    return [add_node]


@mx_op.register("broadcast_add")
def covert_broadcast_add(node, **kwargs):
    """Map MXNet's broadcast_add operator attributes to onnx's Add operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_a_id = kwargs["index_lookup"][inputs[0][0]]
    input_node_b_id = kwargs["index_lookup"][inputs[1][0]]

    input_node_a = proc_nodes[input_node_a_id].name
    input_node_b = proc_nodes[input_node_b_id].name

    add_node = helper.make_node(
        "Add",
        [input_node_a, input_node_b],
        [name],
        name=name,
    )

    return [add_node]


@mx_op.register("elemwise_sub")
def convert_elementwise_sub(node, **kwargs):
    """Map MXNet's elemwise_sub operator attributes to onnx's Sub operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_a_id = kwargs["index_lookup"][inputs[0][0]]
    input_node_b_id = kwargs["index_lookup"][inputs[1][0]]

    input_node_a = proc_nodes[input_node_a_id].name
    input_node_b = proc_nodes[input_node_b_id].name

    sub_node = helper.make_node(
        "Sub",
        [input_node_a, input_node_b],
        [name],
        name=name,
    )

    return [sub_node]

@mx_op.register("broadcast_sub")
def covert_broadcast_sub(node, **kwargs):
    """Map MXNet's broadcast_sub operator attributes to onnx's Sub operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_a_id = kwargs["index_lookup"][inputs[0][0]]
    input_node_b_id = kwargs["index_lookup"][inputs[1][0]]

    input_node_a = proc_nodes[input_node_a_id].name
    input_node_b = proc_nodes[input_node_b_id].name

    sub_node = helper.make_node(
        "Sub",
        [input_node_a, input_node_b],
        [name],
        name=name,
    )

    return [sub_node]


@mx_op.register("elemwise_mul")
def convert_elemwise_mul(node, **kwargs):
    """Map MXNet's elemwise_mul operator attributes to onnx's Mul operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_a_id = kwargs["index_lookup"][inputs[0][0]]
    input_node_b_id = kwargs["index_lookup"][inputs[1][0]]

    input_node_a = proc_nodes[input_node_a_id].name
    input_node_b = proc_nodes[input_node_b_id].name

    mul_node = helper.make_node(
        "Mul",
        [input_node_a, input_node_b],
        [name],
        name=name,
    )

    return [mul_node]

@mx_op.register("broadcast_mul")
def convert_broadcast_mul(node, **kwargs):
    """Map MXNet's broadcast_mul operator attributes to onnx's Mul operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_a_id = kwargs["index_lookup"][inputs[0][0]]
    input_node_b_id = kwargs["index_lookup"][inputs[1][0]]

    input_node_a = proc_nodes[input_node_a_id].name
    input_node_b = proc_nodes[input_node_b_id].name

    mul_node = helper.make_node(
        "Mul",
        [input_node_a, input_node_b],
        [name],
        name=name
    )

    return [mul_node]


@mx_op.register("elemwise_div")
def convert_elemwise_div(node, **kwargs):
    """Map MXNet's elemwise_div operator attributes to onnx's Div operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_a_id = kwargs["index_lookup"][inputs[0][0]]
    input_node_b_id = kwargs["index_lookup"][inputs[1][0]]

    input_node_a = proc_nodes[input_node_a_id].name
    input_node_b = proc_nodes[input_node_b_id].name

    div_node = helper.make_node(
        "Div",
        [input_node_a, input_node_b],
        [name],
        name=name
    )

    return [div_node]


@mx_op.register("broadcast_div")
def convert_broadcast_div(node, **kwargs):
    """Map MXNet's broadcast_div operator attributes to onnx's Div operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_a_id = kwargs["index_lookup"][inputs[0][0]]
    input_node_b_id = kwargs["index_lookup"][inputs[1][0]]

    input_node_a = proc_nodes[input_node_a_id].name
    input_node_b = proc_nodes[input_node_b_id].name

    div_node = helper.make_node(
        "Div",
        [input_node_a, input_node_b],
        [name],
        name=name
    )

    return [div_node]


@mx_op.register("negative")
def convert_negative(node, **kwargs):
    """Map MXNet's negative operator attributes to onnx's Neg operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]

    input_node = proc_nodes[input_node_id].name

    neg_node = helper.make_node(
        "Neg",
        [input_node],
        [name],
        name=name,
    )

    return [neg_node]


@mx_op.register("abs")
def convert_abs(node, **kwargs):
    """Map MXNet's abs operator attributes to onnx's Abs operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]

    input_node = proc_nodes[input_node_id].name

    abs_node = helper.make_node(
        "Abs",
        [input_node],
        [name],
        name=name
    )

    return [abs_node]


@mx_op.register("add_n")
def convert_addn(node, **kwargs):
    """Map MXNet's add_n operator attributes to onnx's Sum operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_list = []
    for input_val in inputs:
        input_list.append(proc_nodes[kwargs["index_lookup"][input_val[0]]].name)

    sum_node = helper.make_node(
        "Sum",
        input_list,
        [name],
        name=name
    )
    return [sum_node]

 # Rounding
@mx_op.register("ceil")
def convert_ceil(node, **kwargs):
    """Map MXNet's ceil operator attributes to onnx's Ceil operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Ceil",
        [input_node],
        [name],
        name=name
    )
    return [node]

@mx_op.register("floor")
def convert_floor(node, **kwargs):
    """Map MXNet's floor operator attributes to onnx's Floor operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Floor",
        [input_node],
        [name],
        name=name
    )
    return [node]

# Changing shape and type.
@mx_op.register("Reshape")
def convert_reshape(node, **kwargs):
    """Map MXNet's Reshape operator attributes to onnx's Reshape operator.
    Converts output shape attribute to output shape tensor
    and return multiple created nodes.
    """
    helper, _, mapping = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    attrs = node["attrs"]

    output_shape_list = convert_string_to_list(attrs["shape"])

    initializer = kwargs["initializer"]
    output_shape_np = np.array(output_shape_list)
    data_type = mapping.NP_TYPE_TO_TENSOR_TYPE[output_shape_np.dtype]
    dims = np.shape(output_shape_np)

    output_shape_name = "reshape_attr_tensor" + str(kwargs["idx"])
    tensor_node = helper.make_tensor_value_info(output_shape_name, data_type, dims)

    initializer.append(
        helper.make_tensor(
            name=output_shape_name,
            data_type=data_type,
            dims=dims,
            vals=output_shape_list,
            raw=False,
        )
    )

    input_node_idx = kwargs["index_lookup"][inputs[0][0]]
    input_node_name = proc_nodes[input_node_idx].name

    not_supported_shape = [-2, -3, -4]

    for val in output_shape_list:
        if val in not_supported_shape:
            raise AttributeError("Shape value not supported in ONNX", val)

    reshape_node = helper.make_node(
        "Reshape",
        [input_node_name, output_shape_name],
        [name],
        name=name
    )

    return [tensor_node, reshape_node]

@mx_op.register("Cast")
def convert_cast(node, **kwargs):
    """Map MXNet's Cast operator attributes to onnx's Cast operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    dtype = node["attrs"]["dtype"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Cast",
        [input_node],
        [name],
        to=dtype,
        name=name,
    )
    return [node]


@mx_op.register("slice_axis")
def convert_slice_axis(node, **kwargs):
    """Map MXNet's slice_axis operator attributes to onnx's Slice operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    axes = int(node["attrs"]["axis"])
    starts = int(node["attrs"]["begin"])
    if node["attrs"]["end"] == 'None':
        raise ValueError("Slice: ONNX doesnt't support 'None' in 'end' attribute")
    else:
        ends = int(node["attrs"]["end"])

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Slice",
        [input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    num_outputs = int(node.get("attrs", {})["num_outputs"])
    axis = int(node.get("attrs", {}).get("axis", 1))
    squeeze_axis = int(node.get("attrs", {}).get("squeeze_axis", 0))

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    if squeeze_axis == 1 and num_outputs == 1:
        node = helper.make_node(
            "Squeeze",
            [input_node],
            [name],
            axes=[axis],
            name=name,
        )
        return [node]
    elif squeeze_axis == 0 and num_outputs > 1:
        node = helper.make_node(
            "Split",
            [input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    axis = int(node["attrs"]["axis"])

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Unsqueeze",
        [input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    if "axis" in node["attrs"]:
        axis = convert_string_to_list(node["attrs"]["axis"])
    else:
        raise AttributeError("Missing axis attribute: ONNX currently requires axis to "
                             "be specified for squeeze operator")

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Squeeze",
        [input_node],
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
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Log",
        [input_node],
        [name],
        name=name,
    )
    return [node]


@mx_op.register("reciprocal")
def convert_reciprocal(node, **kwargs):
    """Map MXNet's reciprocal operator attributes to onnx's Reciprocal operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Reciprocal",
        [input_node],
        [name],
        name=name,
    )
    return [node]


@mx_op.register("_power")
def convert_power(node, **kwargs):
    """Map MXNet's _power operator attributes to onnx's Pow operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_a_id = kwargs["index_lookup"][inputs[0][0]]
    input_node_b_id = kwargs["index_lookup"][inputs[1][0]]

    input_node_a = proc_nodes[input_node_a_id].name
    input_node_b = proc_nodes[input_node_b_id].name

    node = helper.make_node(
        "Pow",
        [input_node_a, input_node_b],
        [name],
        name=None
    )
    return [node]

@mx_op.register("sqrt")
def convert_sqrt(node, **kwargs):
    """Map MXNet's sqrt operator attributes to onnx's Sqrt operator
    and return the created node.
    """
    helper, _, _ = import_onnx_modules()
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_node_id = kwargs["index_lookup"][inputs[0][0]]
    input_node = proc_nodes[input_node_id].name

    node = helper.make_node(
        "Sqrt",
        [input_node],
        [name],
        name=name,
    )
    return [node]
