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
    if name.endswith("_weight"):
        return True
    if name.endswith("_bias"):
        return True
    if name.endswith("_beta") or name.endswith("_gamma") or name.endswith("_moving_var") or name.endswith(
            "_moving_mean"):
        return True
    return False


@mx2onnx.register("null")
def convert_weights_and_inputs(node, **kwargs):
    name = node["name"]
    if looks_like_weight(name):
        weights = kwargs["weights"]
        initializer = kwargs["initializer"]
        weights = kwargs["weights"]
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
    tuple_re = re.compile('\([0-9|,| ]+\)')

    def parse_helper(attrs_name, alt_value=None):
        if attrs is None:
            return alt_value
        attrs_str = attrs.get(attrs_name)
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
    momentum = float(attrs["momentum"])
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
    # is consistent for other activations, this can be changed to
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
            [input_node.output[0]],  # input
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
            [input_node.output[0]],  # input
            [name],
            name=name
        )

    return node


@mx2onnx.register("exp")
def convert_exp(node, **kwargs):
    raise NotImplementedError


# There's also mx.sym.softmax(), which doesn't do cross-entropy loss,
# just softmax for inference - hence the name convert_softmax_output.
@mx2onnx.register("SoftmaxOutput")
def convert_softmax_output(node, **kwargs):
    #    print("\nIn convert_softmax_output")
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
    axis = int(node.get("attrs", {}).get("axis", 1))
    concat_node = helper.make_node(
        "Concat",
        input_names,
        [name],
        axis=axis,
        name=name
    )
    return concat_node


@mx2onnx.register("Dropout")
def convert_dropout(node, **kwargs):
    name = node["name"]
    input_id = node["inputs"][0][0]
    input_name = kwargs["proc_nodes"][input_id].name
    attrs = node["attrs"]
    p = float(attrs["p"])
    dropout_node = helper.make_node(
        "Dropout",
        [input_name],
        [name],
        ratio=p,
        is_test=0,
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
    raise NotImplementedError


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


@mx2onnx.register("_sub")
def convert_elementwise_sub(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("abs")
def convert_abs(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("_mul")
def convert_mul(node, proc_nodes):
    raise NotImplementedError


@mx2onnx.register("_div")
def convert_div(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("log")
def convert_log(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("max")
def convert_max(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("_maximum")
def convert_maximum(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("min")
def convert_min(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("_minimum")
def convert_minimum(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("_power")
def convert_power(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("sqrt")
def convert_sqrt(node, **kwargs):
    raise NotImplementedError
