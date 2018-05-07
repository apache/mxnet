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
"""
mx_to_uff_converter_functions.py

Conversion Functions for common layers.
Add new functions here with a decorator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import defs, checker, helper, numpy_helper, mapping

from .export_onnx import MxNetToONNXConverter as mx2onnx

import numpy as np

import re

import sys


def looks_like_weight(name):
    """Internal helper to figure out if node should be hidden with `hide_weights`.
    """
    if name.endswith("_weight") or name == 'W':
        return True
    if name.endswith("_bias") or name == "B":
        return True
    if name.endswith("_beta") or name.endswith("_gamma") or name.endswith("_moving_var") or name.endswith(
            "_moving_mean"):
        return True
    return False


@mx2onnx.register("null")
def convert_weights_and_inputs(node, **kwargs):
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

        return tensor_node
    else:
        tval_node = helper.make_tensor_value_info(name, kwargs["in_type"], kwargs["in_shape"])
        return tval_node


@mx2onnx.register("Convolution")
def convert_convolution(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]

    num_inputs = len(inputs)

    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[inputs[0][0]].name
    weights_node = proc_nodes[inputs[1][0]].name

    if num_inputs > 2:
        bias_node = proc_nodes[inputs[2][0]].name

    attrs = node.get("attrs")
    tuple_re = re.compile('\([0-9L|,| ]+\)')

    def parse_helper(attrs_name, alt_value=None):
        if attrs is None:
            return alt_value
        attrs_str = str(attrs.get(attrs_name))
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

    num_filter = int(attrs["num_filter"])
    kernel_dims = list(parse_helper("kernel"))
    stride_dims = list(parse_helper("stride", [1, 1]))
    pad_dims = list(parse_helper("pad", [0, 0]))
    num_group = int(attrs.get("num_group", 1))

    if len(pad_dims) < 2 * len(kernel_dims):
        pad_dims = [0] * (2 * len(kernel_dims) - len(pad_dims)) + pad_dims

    input_nodes = [input_node, weights_node]
    if num_inputs > 2:
        input_nodes.append(bias_node)

    conv_node = helper.make_node(
        "Conv",
        inputs=input_nodes,
        outputs=[name],
        kernel_shape=kernel_dims,
        strides=stride_dims,
        pads=pad_dims,
        group=num_group,
        name=name
    )

    return conv_node


@mx2onnx.register("FullyConnected")
def convert_fully_connected(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]
    input_node_id = inputs[0][0]
    weight_node_id = inputs[1][0]
    bias_node_id = inputs[2][0]
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
        broadcast=True,
        transA=False,
        transB=True,
        name=name
    )

    return node


@mx2onnx.register("BatchNorm")
def convert_batchnorm(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    attrs = node["attrs"]
    momentum = float(node.get("attrs", {}).get("momentum", 0.9))
    eps = float(attrs["eps"])

    data_idx = inputs[0][0]
    gamma_idx = inputs[1][0]
    beta_idx = inputs[2][0]
    moving_mean_idx = inputs[3][0]
    moving_var_idx = inputs[4][0]

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
        is_test=1,
        spatial=1
    )

    return bn_node


@mx2onnx.register("tanh")
def convert_tanh(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]
    input_node_idx = inputs[0][0]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_node_idx].name

    node = helper.make_node(
        'Tanh',
        [input_node],
        [name],
        name=name
    )
    return node

#Basic neural network functions
@mx2onnx.register("sigmoid")
def convert_sigmoid(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]
    input_node_idx = inputs[0][0]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_node_idx].name

    node = helper.make_node(
        'Sigmoid',
        [input_node],
        [name],
        name=name
    )
    return node

@mx2onnx.register("relu")
def convert_relu(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]
    input_node_idx = inputs[0][0]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_node_idx].name

    node = helper.make_node(
        'Relu',
        [input_node],
        [name],
        name=name
    )

    return node

@mx2onnx.register("Activation")
def convert_activation(node, **kwargs):
    name = node["name"]

    proc_nodes = kwargs["proc_nodes"]
    attrs = node["attrs"]
    act_type = attrs["act_type"]

    inputs = node["inputs"]
    input_node_idx = inputs[0][0]
    input_node = proc_nodes[input_node_idx].output[0]

    # Creating a dictionary here, but if this titlecase pattern
    # mxnet_name.title()
    act_types = {
        "tanh": "Tanh",
        "relu": "Relu"
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

    return node

def transform_padding(pad_width):
    num_pad_values = len(pad_width)
    pad_values_middle_index = int(num_pad_values/2)

    onnx_pad_width = [0]*num_pad_values

    start_index = 0
    end_index = int(num_pad_values/2)
    for idx in range(0,num_pad_values):
        if idx%2 == 0:
            onnx_pad_width[start_index] = pad_width[idx]
            start_index += 1
        else:
            onnx_pad_width[end_index] = pad_width[idx]
            end_index += 1

    return onnx_pad_width


def convert_string_to_list(string_val):
    result_list = []

    list_string = string_val.split(',')
    for val in list_string:
        val = str(val.strip())
        val = val.replace("(", "")
        val = val.replace(")", "")
        val = val.replace("L", "")
        if val is not "":
            result_list.append(int(val))

    return result_list


@mx2onnx.register("Pad")
def convert_pad(node, **kwargs):
    name = node["name"]
    attrs = node["attrs"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    input_node_idx = inputs[0][0]
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

    return node


@mx2onnx.register("_linalg_gemm2")
def convert_linalg_gemm2(node, **kwargs):
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]
    name = node["name"]
    input_a_idx = node_inputs[0][0]
    input_node_a = proc_nodes[input_a_idx].name

    input_b_idx = node_inputs[1][0]
    input_node_b = proc_nodes[input_b_idx].name

    if "attrs" in node:
        attrs = node["attrs"]
        alpha = float(attrs.get("alpha"))
    else:
        alpha = 1.0

    if alpha == 1.0:
        node = helper.make_node(
            'MatMul',
            inputs=[input_node_a, input_node_b],
            outputs=[name],
            name=name
        )
    else:
        raise AttributeError("TODO: Add support for alpha multiplication")

    return node


@mx2onnx.register("Pooling")
def convert_pooling(node, **kwargs):
    proc_nodes = kwargs["proc_nodes"]
    attrs = node["attrs"]
    kernel = eval(attrs["kernel"])
    pool_type = attrs["pool_type"]
    stride = eval(attrs["stride"]) if attrs.get("stride") else None
    node_inputs = node["inputs"]
    input_node_idx = node_inputs[0][0]
    input_node = proc_nodes[input_node_idx]
    name = node["name"]

    pool_types = {"max": "MaxPool", "avg": "AveragePool"}
    global_pool_types = {"max": "GlobalMaxPool", "avg": "GlobalAveragePool"}

    if stride:
        node = helper.make_node(
            pool_types[pool_type],
            [input_node.name],  # input
            [name],
            #        dilations = [0, 0],
            kernel_shape=kernel,
            pads=[0, 0],
            strides=stride,
            name=name
        )
    else:
        node = helper.make_node(
            global_pool_types[pool_type],
            [input_node.name],  # input
            [name],
            name=name
        )

    return node


@mx2onnx.register("exp")
def convert_exp(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    a_node = proc_nodes[a].name

    node = helper.make_node(
        "Exp",
        [a_node],
        [name],
        name=name,
    )
    return node


@mx2onnx.register("softmax")
def convert_softmax(node, **kwargs):
    inputs = node["inputs"]
    input_idx = inputs[0][0]
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

    return softmax_node


# There's also mx.sym.softmax(), which doesn't do cross-entropy loss,
# just softmax for inference - hence the name convert_softmax_output.
@mx2onnx.register("SoftmaxOutput")
def convert_softmax_output(node, **kwargs):
    inputs = node["inputs"]
    input1_idx = inputs[0][0]
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

    return softmax_node


@mx2onnx.register("Concat")
def convert_concat(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]
    proc_nodes = kwargs["proc_nodes"]
    input_names = [proc_nodes[i[0]].name for i in inputs]
    axis = int(node.get("attrs", {}).get("dim", 1))
    concat_node = helper.make_node(
        "Concat",
        input_names,
        [name],
        axis=axis,
        name=name
    )
    return concat_node


@mx2onnx.register("transpose")
def convert_transpose(node, **kwargs):
    name = node["name"]
    input_idx = node["inputs"][0][0]
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

    return transpose_node


@mx2onnx.register("Dropout")
def convert_dropout(node, **kwargs):
    name = node["name"]
    input_id = node["inputs"][0][0]
    input_name = kwargs["proc_nodes"][input_id].name
    attrs = node["attrs"]
    p = float(attrs["p"])
    is_test = 0 if str(attrs["mode"]) is "always" else 1
    dropout_node = helper.make_node(
        "Dropout",
        [input_name],
        [name],
        ratio=p,
        is_test=is_test,
        name=name
    )
    return dropout_node


@mx2onnx.register("Flatten")
def convert_flatten(node, **kwargs):
    name = node["name"]
    input_idx = node["inputs"][0][0]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_idx].name  # .output[0]

    flatten_node = helper.make_node(
        "Flatten",
        [input_node],
        [name],
        name=name
    )
    return flatten_node


@mx2onnx.register("_mul_scalar")
def convert_mul_scalar(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    scalar_mul_value = int(node.get("attrs", {}).get("scalar", 1))

    a = inputs[0][0]

    print(type(proc_nodes[a].name))
    a_node = proc_nodes[a].name
  #  b_node = str(scalar_mul_value).decode("utf-8")

    b_node = helper.make_tensor(
            name=name+"b",
            data_type=1,
            dims=[1],
            vals=[scalar_mul_value],
            raw=False,
        )

    mul_node = helper.make_node(
        "Mul",
        [a_node, b_node.name],
        [name],
        name=name,
    )

    return mul_node


# Sorting and Searching
@mx2onnx.register("argmax")
def convert_argmax(node, **kwargs):
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]

    input_node_idx = node_inputs[0][0]
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
    return node

@mx2onnx.register("argmin")
def convert_argmin(node, **kwargs):
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]

    input_node_idx = node_inputs[0][0]
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
    return node

@mx2onnx.register("_maximum")
def convert_max(node, **kwargs):
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]

    input_node_list = []
    for node_input in node_inputs:
        input_node_list.append(proc_nodes[node_input[0]].name)

    name = node["name"]

    node = helper.make_node(
        'Max',
        inputs=input_node_list,
        outputs=[name],
        name=name,
    )

    return node


@mx2onnx.register("_minimum")
def convert_min(node, **kwargs):
    proc_nodes = kwargs["proc_nodes"]
    node_inputs = node["inputs"]

    input_node_list = []
    for node_input in node_inputs:
        input_node_list.append(proc_nodes[node_input[0]].name)

    name = node["name"]

    node = helper.make_node(
        'Min',
        inputs=input_node_list,
        outputs=[name],
        name=name,
    )

    return node



# Arithmetic Operations
@mx2onnx.register("elemwise_add")
def convert_elementwise_add(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    weights = kwargs["weights"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    add_node = helper.make_node(
        "Add",
        [a_node, b_node],
        [name],
        name=name,
    )

    return add_node


@mx2onnx.register("broadcast_add")
def covert_broadcast_add(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    add_node = helper.make_node(
        "Add",
        [a_node, b_node],
        [name],
        broadcast=1,
        name=name,
    )

    return add_node


@mx2onnx.register("elemwise_sub")
def convert_elementwise_sub(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    sub_node = helper.make_node(
        "Sub",
        [a_node, b_node],
        [name],
        name=name,
    )

    return sub_node

@mx2onnx.register("broadcast_sub")
def covert_broadcast_sub(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    sub_node = helper.make_node(
        "Sub",
        [a_node, b_node],
        [name],
        broadcast=1,
        name=name,
    )

    return sub_node


@mx2onnx.register("elemwise_mul")
def convert_mul(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    mul_node = helper.make_node(
        "Mul",
        [a_node, b_node],
        [name],
        name=name,
    )

    return mul_node

@mx2onnx.register("broadcast_mul")
def convert_mul(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    mul_node = helper.make_node(
        "Mul",
        [a_node, b_node],
        [name],
        name=name,
        broadcast=1
    )

    return mul_node


@mx2onnx.register("elemwise_div")
def convert_mul(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    div_node = helper.make_node(
        "Div",
        [a_node, b_node],
        [name],
        name=name,
    )

    return div_node


@mx2onnx.register("broadcast_div")
def convert_div(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    div_node = helper.make_node(
        "Div",
        [a_node, b_node],
        [name],
        name=name,
        broadcast=1
    )

    return div_node


@mx2onnx.register("negative")
def convert_negative(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]

    a_node = proc_nodes[a].name

    neg_node = helper.make_node(
        "Neg",
        [a_node],
        [name],
        name=name,
    )

    return neg_node

@mx2onnx.register("abs")
def convert_abs(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]

    a_node = proc_nodes[a].name

    abs_node = helper.make_node(
        "Abs",
        [a_node],
        [name],
        name=name,
    )

    return abs_node

@mx2onnx.register("add_n")
def convert_addn(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    input_list = []
    for idx, input_val in enumerate(inputs):
        input_list.append(proc_nodes[input_val[0]].name)

    sum_node = helper.make_node(
        "Sum",
        input_list,
        [name],
        name=name,
    )
    return sum_node

 #Rounding
@mx2onnx.register("ceil")
def convert_floor(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    a_node = proc_nodes[a].name

    node = helper.make_node(
        "Ceil",
        [a_node],
        [name],
        name=name,
    )
    return node

@mx2onnx.register("floor")
def convert_floor(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    a_node = proc_nodes[a].name

    node = helper.make_node(
        "Floor",
        [a_node],
        [name],
        name=name,
    )
    return node

#Changing shape and type.
@mx2onnx.register("Reshape")
def convert_reshape(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    attrs = node["attrs"]

    output_shape = convert_string_to_list(attrs["shape"])
    input_node_idx = inputs[0][0]
    input_node_name = proc_nodes[input_node_idx].name

    not_supported_shape = [ -2, -3, -4]

    for val in output_shape:
        if val in not_supported_shape:
            raise AttributeError("Shape value not supported in ONNX", val)

    node = helper.make_node(
        "Reshape",
        [input_node_name],
        [name],
        name=name,
        shape=output_shape
    )

    return node

@mx2onnx.register("Cast")
def convert_cast(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    dtype = node["attrs"]["dtype"]

    a = inputs[0][0]
    a_node = proc_nodes[a].name

    node = helper.make_node(
        "Cast",
        [a_node],
        [name],
        to=dtype,
        name=name,
    )
    return node


@mx2onnx.register("slice_axis")
def convert_slice_axis(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    axes = int(node["attrs"]["axis"])
    starts = int(node["attrs"]["begin"])
    ends = int(node["attrs"]["end"])

    a = inputs[0][0]
    a_node = proc_nodes[a].name

    node = helper.make_node(
        "Slice",
        [a_node],
        [name],
        axes=[axes],
        starts=[starts],
        ends=[ends],
        name=name,
    )
    return node


# SliceChannel/split operators will be mapped to onnx's squeeze and split operator.
# [TODO] Address split with squeeze case
@mx2onnx.register("SliceChannel")
def convert_slice_channel(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    num_outputs = int(node.get("attrs", {})["num_outputs"])
    axis = int(node.get("attrs", {}).get("axis", 1))
    squeeze_axis = int(node.get("attrs", {}).get("squeeze_axis", 0))

    a = inputs[0][0]
    a_node = proc_nodes[a].name

    if num_outputs==1 and squeeze_axis==1:
        node = helper.make_node(
            "Squeeze",
            [a_node],
            [name],
            axes=[axis],
            name=name,
        )
    else:
        node = helper.make_node(
            "Split",
            [a_node],
            [name],
            axis=axis,
            split=[num_outputs],
            name=name,
        )

    return node


@mx2onnx.register("log")
def convert_log(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    a_node = proc_nodes[a].name

    node = helper.make_node(
        "Log",
        [a_node],
        [name],
        name=name,
    )
    return node


@mx2onnx.register("reciprocal")
def convert_reciprocal(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    a_node = proc_nodes[a].name

    node = helper.make_node(
        "Reciprocal",
        [a_node],
        [name],
        name=name,
    )
    return node


@mx2onnx.register("_power")
def convert_power(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    node = helper.make_node(
        "Pow",
        [a_node, b_node],
        [name],
        name=None
    )
    return node

#[TODO] broadcast_power with axis
@mx2onnx.register("broadcast_power")
def convert_power(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    node = helper.make_node(
        "Pow",
        [a_node, b_node],
        outputs=[name],
        name=name,
        axis=1,
        broadcast=1,
    )
    return node


@mx2onnx.register("sqrt")
def convert_sqrt(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    a = inputs[0][0]
    a_node = proc_nodes[a].name

    node = helper.make_node(
        "Sqrt",
        [a_node],
        [name],
        name=name,
    )
    return node
