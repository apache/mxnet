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
# pylint: disable=invalid-name, too-many-locals, fixme
# pylint: disable=too-many-branches, too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value
"""Visualization module"""
from __future__ import absolute_import

import re
import copy
import json
import math
import warnings
from .symbol import Symbol
import operator
from functools import reduce


def _str2tuple(string):
    """Convert shape string to list, internal use only.

    Parameters
    ----------
    string: str
        Shape string.

    Returns
    -------
    list of str
        Represents shape.
    """
    return re.findall(r"\d+", string)


def _str2ints(string):
    return map(int, _str2tuple(string))


def get_flops(feature_map_size, conv_filter, stride=1, padding=1):
    n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length
    flops_per_instance = n

    num_instances_per_filter = ((feature_map_size - conv_filter[2] + 2 * padding) // stride) + 1  # for rows
    num_instances_per_filter *= ((feature_map_size - conv_filter[2] + 2 * padding) // stride) + 1  # multiplying with cols

    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_per_layer = flops_per_filter * conv_filter[0]  # multiply with number of filters
    return total_flops_per_layer


def convert_size(size_bytes, base=1024):
    if size_bytes == 0:
        return "0B"
    if base == 1024:
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    elif base == 1000:
        size_name = ("", "thousand", "million", "billion", "trillion", "Peta", "Eta", "Zeta", "Y")
    i = int(math.floor(math.log(size_bytes, base)))
    p = math.pow(base, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def print_summary(symbol, shape=None, line_length=120, positions=[.40, .52, .60, .68, .73, 1.], quantized_bitwidth=32):
    """Convert symbol for detail information.

    Parameters
    ----------
    symbol: Symbol
        Symbol to be visualized.
    shape: dict
        A dict of shapes, str->shape (tuple), given input shapes.
    line_length: int
        Rotal length of printed lines
    positions: list
        Relative or absolute positions of log elements in each line.

    Returns
    ------
    None

    Notes
    -----
    If ``mxnet`` is imported, the visualization module can be used in its short-form.
    For example, if we ``import mxnet`` as follows::

        import mxnet

    this method in visualization module can be used in its short-form as::

        mxnet.viz.print_summary(...)

    """
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    show_shape = False
    if shape is not None:
        show_shape = True
        interals = symbol.get_internals()
        _, out_shapes, _ = interals.infer_shape(**shape)
        if out_shapes is None:
            raise ValueError("Input shape is incomplete")
        shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    heads = set(conf["heads"][0])
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'FLOPS #', 'Prec.', 'Previous Layer']
    def print_row(fields, positions):
        """Print format row.

        Parameters
        ----------
        fields: list
            Information field.
        positions: list
            Field length ratio.
        Returns
        ------
        None
        """
        line = ''
        for i, field in enumerate(fields):
            line += str(field)
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)
    print('_' * line_length)
    print_row(to_display, positions)
    print('=' * line_length)
    def print_layer_summary(node, out_shape):
        """print layer information

        Parameters
        ----------
        node: dict
            Node information.
        out_shape: dict
            Node shape information.
        Returns
        ------
            Node total parameters.
        """
        op = node["op"]
        pre_node = []
        pre_filter = 0
        pre_feature_map = 0
        if op != "null":
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                if input_node["op"] != "null" or item[0] in heads:
                    # add precede
                    pre_node.append(input_name)
                    if show_shape:
                        if input_node["op"] != "null":
                            key = input_name + "_output"
                        else:
                            key = input_name
                        if key in shape_dict:
                            shape = shape_dict[key][1:]
                            if not shape or op == 'Convolution' and pre_filter > 0:
                                continue
                            pre_filter += int(shape[0])
                            if op == 'Convolution':
                                pre_feature_map = int(shape[1])
        is_quantized = "qconv" in node["name"] or "qdense" in node["name"] or "scaledbinaryconv" in node["name"]
        cur_param = 0
        flops = 0
        if op == 'Convolution':
            num_group = int(node['attrs'].get('num_group', '1'))
            num_filter = int(node["attrs"]["num_filter"])
            cur_param = pre_filter * num_filter // num_group
            kernel_size = reduce(operator.mul, _str2ints(node["attrs"]["kernel"]))
            cur_param *= kernel_size

            conv_filter = (pre_filter, num_filter) + tuple(_str2ints(node["attrs"]["kernel"]))
            stride = tuple(_str2ints(node["attrs"].get("stride", "1")))[0]
            pad = tuple(_str2ints(node["attrs"].get("pad", "1")))[0]
            flops = get_flops(pre_feature_map, conv_filter, stride=stride, padding=pad) // num_group
            if node["attrs"].get("no_bias", 'False') != 'True':
                cur_param += num_filter
        elif op == 'FullyConnected':
            if node["attrs"].get("no_bias", 'False') == 'True':
                cur_param = pre_filter * int(node["attrs"]["num_hidden"])
            else:
                cur_param = (pre_filter+1) * int(node["attrs"]["num_hidden"])
            # FC layers are not counted in related work
            # flops = (pre_filter+1) * int(node["attrs"]["num_hidden"])
        elif op == 'BatchNorm':
            key = node["name"] + "_output"
            if show_shape:
                num_filter = shape_dict[key][1]
                cur_param = int(num_filter) * 2
        elif op == 'Embedding':
            cur_param = int(node["attrs"]['input_dim']) * int(node["attrs"]['output_dim'])
        if not pre_node:
            first_connection = ''
        else:
            first_connection = pre_node[0]
        fields = [node['name'] + '(' + op + ')',
                  "x".join([str(x) for x in out_shape]),
                  cur_param,
                  '{:.2E}'.format(flops) if flops > 0 else 0,
                  'Q/B' if is_quantized else 'FP',
                  first_connection]
        print_row(fields, positions)
        if len(pre_node) > 1:
            for i in range(1, len(pre_node)):
                fields = [''] * (len(fields) - 1) + [pre_node[i]]
                print_row(fields, positions)
        return cur_param, is_quantized, flops
    total_flops = 0
    quantized_flops = 0
    total_params = 0
    quantized_params = 0
    compressed_bytes = 0
    for i, node in enumerate(nodes):
        out_shape = []
        op = node["op"]
        if op == "null" and i > 0:
            continue
        if op != "null" or i in heads:
            if show_shape:
                if op != "null":
                    key = node["name"] + "_output"
                else:
                    key = node["name"]
                if key in shape_dict:
                    out_shape = shape_dict[key][1:]
        params, is_quantized, flops = print_layer_summary(nodes[i], out_shape)
        total_params += params
        total_flops += flops
        if is_quantized:
            quantized_params += params
            quantized_flops += flops
        compressed_bytes += params * (quantized_bitwidth if is_quantized else 32) / 8
        if i == len(nodes) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)
    print("Total params:     {params}".format(params=total_params))
    print('FP params:        {fp_params} ({percentage:.2f}%)'.format(
        fp_params=(total_params - quantized_params),
        percentage=(100 * (total_params - quantized_params) / total_params))
    )
    print('Quantized params: {q_params} ({percentage:.2f}%)'.format(
        q_params=quantized_params,
        percentage=(100 * quantized_params / total_params))
    )
    print('Model size (full-precision): ~{}'.format(convert_size(total_params * 4)))
    print('Model size (compressed):     ~{}'.format(convert_size(compressed_bytes)))
    print('FLOPS (total):               {}'.format(convert_size(total_flops, 1000)))
    print('FLOPS (full-precision):      {}'.format(convert_size(total_flops - quantized_flops, 1000)))
    print('FLOPS (binary):              {}'.format(convert_size(quantized_flops, 1000)))
    print('FLOPS (combined):            {}'.format(convert_size(quantized_flops/64 + (total_flops - quantized_flops), 1000)))
    print('_' * line_length)

def shrink_qlayers(nodes):
    delete = None
    for i, node in enumerate(nodes):
        op = node["op"]
        name = node["name"]

        qconv_hide_functions = ["__plusscalar", "__minusscalar", "__divscalar", "__mulscalar", "_det_sign", "_pad",
                                "_round_ste", "_broadcast_div", "_tanh", "_max", "_abs",
                                "_stop_gradient", "transpose"]

        if ("_qconv" in name or "_scaledbinaryconv" in name) and any(x in name for x in qconv_hide_functions):
            delete = i
            break

        qactivation_hide_functions = ["_gradcancel", "_clip", "__divscalar", "__mulscalar"]

        if "_qactivation" in name and any(x in name for x in qactivation_hide_functions):
            delete = i
            break

    if delete is None:
        return nodes, False

    deleted_node = nodes[delete].copy()
    deleted_node_inputs = deleted_node["inputs"]
    first_previous_input = deleted_node_inputs[0]
    assert len(deleted_node_inputs) == 1 or all(a == first_previous_input[0] for a, _, _ in deleted_node_inputs)

    del nodes[delete]

    for node in nodes:
        inputs = node["inputs"]
        new_inputs = []
        for a, b, c in inputs:
            if a == delete:
                a, b, c = first_previous_input
            elif a > delete:
                a = a-1
            new_inputs.append([a, b, c])
        node["inputs"] = new_inputs
    return nodes, True

def plot_network(symbol, title="plot", save_format='pdf', shape=None, dtype=None, node_attrs={},
                 hide_weights=True, consolidate_binary_layers=True):
    """Creates a visualization (Graphviz digraph object) of the given computation graph.
    Graphviz must be installed for this function to work.

    Parameters
    ----------
    title: str, optional
        Title of the generated visualization.
    symbol: Symbol
        A symbol from the computation graph. The generated digraph will visualize the part
        of the computation graph required to compute `symbol`.
    shape: dict, optional
        Specifies the shape of the input tensors. If specified, the visualization will include
        the shape of the tensors between the nodes. `shape` is a dictionary mapping
        input symbol names (str) to the corresponding tensor shape (tuple).
    dtype: dict, optional
        Specifies the type of the input tensors. If specified, the visualization will include
        the type of the tensors between the nodes. `dtype` is a dictionary mapping
        input symbol names (str) to the corresponding tensor type (e.g. `numpy.float32`).
    node_attrs: dict, optional
        Specifies the attributes for nodes in the generated visualization. `node_attrs` is
        a dictionary of Graphviz attribute names and values. For example::

            node_attrs={"shape":"oval","fixedsize":"false"}

        will use oval shape for nodes and allow variable sized nodes in the visualization.
    hide_weights: bool, optional
        If True (default), then inputs with names of form *_weight* (corresponding to weight
        tensors) or *_bias* (corresponding to bias vectors) will be hidden for a cleaner
        visualization.

    Returns
    -------
    dot: Digraph
        A Graphviz digraph object visualizing the computation graph to compute `symbol`.

    Example
    -------
    >>> net = mx.sym.Variable('data')
    >>> net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
    >>> net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
    >>> net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
    >>> net = mx.sym.SoftmaxOutput(data=net, name='out')
    >>> digraph = mx.viz.plot_network(net, shape={'data':(100,200)},
    ... node_attrs={"fixedsize":"false"})
    >>> digraph.view()

    Notes
    -----
    If ``mxnet`` is imported, the visualization module can be used in its short-form.
    For example, if we ``import mxnet`` as follows::

        import mxnet

    this method in visualization module can be used in its short-form as::

        mxnet.viz.plot_network(...)

    """
    # todo add shape support
    try:
        from graphviz import Digraph
    except:
        raise ImportError("Draw network requires graphviz library")
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be a Symbol")
    internals = symbol.get_internals()
    draw_shape = shape is not None
    if draw_shape:
        _, out_shapes, _ = internals.infer_shape(**shape)
        if out_shapes is None:
            raise ValueError("Input shape is incomplete")
        shape_dict = dict(zip(internals.list_outputs(), out_shapes))
    draw_type = dtype is not None
    if draw_type:
        _, out_types, _ = internals.infer_type(**dtype)
        if out_types is None:
            raise ValueError("Input type is incomplete")
        type_dict = dict(zip(internals.list_outputs(), out_types))
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    # check if multiple nodes have the same name
    if len(nodes) != len(set([node["name"] for node in nodes])):
        seen_nodes = set()
        # find all repeated names
        repeated = set(node['name'] for node in nodes if node['name'] in seen_nodes
                       or seen_nodes.add(node['name']))
        warning_message = "There are multiple variables with the same name in your graph, " \
                          "this may result in cyclic graph. Repeated names: " + ','.join(repeated)
        warnings.warn(warning_message, RuntimeWarning)
    # default attributes of node
    node_attr = {"shape": "box", "fixedsize": "true",
                 "width": "1.3", "height": "0.8034", "style": "filled"}
    # merge the dict provided by user and the default one
    node_attr.update(node_attrs)
    dot = Digraph(name=title, format=save_format)
    # color map
    cm = ("#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
          "#fdb462", "#b3de69", "#fccde5", "#fdbcb5")

    def looks_like_weight(name):
        """Internal helper to figure out if node should be hidden with `hide_weights`.
        """
        weight_like = ('_weight', '_bias', '_beta', '_gamma',
                       '_moving_var', '_moving_mean', '_running_var', '_running_mean')
        return name.endswith(weight_like)

    if consolidate_binary_layers:
        nodes, changed = shrink_qlayers(nodes)
        while changed:
            nodes, changed = shrink_qlayers(nodes)

    # make nodes
    hidden_nodes = set()
    for node in nodes:
        op = node["op"]
        name = node["name"]
        # input data
        attr = copy.deepcopy(node_attr)
        label = name

        if op == "null":
            if looks_like_weight(node["name"]):
                if hide_weights:
                    hidden_nodes.add(node["name"])
                # else we don't render a node, but
                # don't add it to the hidden_nodes set
                # so it gets rendered as an empty oval
                continue
            attr["shape"] = "oval" # inputs get their own shape
            label = node["name"]
            attr["fillcolor"] = cm[0]
        elif op == "Convolution":
            label = "{prefix}Convolution\n{kernel}/{stride}, {filter}".format(
                prefix="Q" if "qconv" in name or "scaledbinaryconv" in name else "",
                kernel="x".join(_str2tuple(node["attrs"]["kernel"])),
                stride="x".join(_str2tuple(node["attrs"]["stride"]))
                if "stride" in node["attrs"] else "1",
                filter=node["attrs"]["num_filter"]
            )
            attr["fillcolor"] = cm[1]
            num_group = int(node["attrs"]["num_group"])
            if num_group > 1:
                label = "{}\n({num_group} groups)".format(label, num_group=num_group)
                attr["fillcolor"] = cm[8]
        elif op == "FullyConnected":
            label = "FullyConnected\n{hidden}".format(hidden=node["attrs"]["num_hidden"])
            attr["fillcolor"] = cm[1]
        elif op == "BatchNorm":
            attr["fillcolor"] = cm[3]
        elif op == 'Activation' or "qactivation" in name:
            if "qactivation" in name:
                label = "QActivation"
            else:
                act_type = node["attrs"]["act_type"]
                label = 'Activation\n{activation}'.format(activation=act_type)
                attr["fillcolor"] = cm[2]
        elif op == 'LeakyReLU':
            attrs = node.get("attrs")
            act_type = attrs.get("act_type", "Leaky") if attrs else "Leaky"
            label = 'LeakyReLU\n{activation}'.format(activation=act_type)
            attr["fillcolor"] = cm[2]
        elif op == "Pooling":
            label = "Pooling\n{pooltype}, {kernel}/{stride}".format(pooltype=node["attrs"]["pool_type"],
                                                                    kernel="x".join(_str2tuple(node["attrs"]["kernel"]))
                                                                    if "kernel" in node["attrs"] else "[]",
                                                                    stride="x".join(_str2tuple(node["attrs"]["stride"]))
                                                                    if "stride" in node["attrs"] else "1")
            attr["fillcolor"] = cm[4]
        elif op in ("Concat", "Flatten", "Reshape"):
            attr["fillcolor"] = cm[5]
        elif op == "Softmax":
            attr["fillcolor"] = cm[6]
        else:
            attr["fillcolor"] = cm[7]
            if op == "Custom":
                label = node["attrs"]["op_type"]

        dot.node(name=name, label=label, **attr)

    # add edges
    for node in nodes:          # pylint: disable=too-many-nested-blocks
        op = node["op"]
        name = node["name"]
        if op == "null":
            continue
        else:
            inputs = node["inputs"]

            if node['op'] == '_contrib_BilinearResize2D':
                inputs = [inputs[0]]

            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                if input_name not in hidden_nodes:
                    attr = {"dir": "back", 'arrowtail':'open', 'label': ''}
                    # add shapes
                    if draw_shape:
                        if input_node["op"] != "null":
                            key = input_name + "_output"
                            if "attrs" in input_node:
                                params = input_node["attrs"]
                                if "num_outputs" in params:
                                    key += str(int(params["num_outputs"]) - 1)
                            if key not in shape_dict:
                                print("Warning: Key '{}' not found in shape_dict. Printing 0x0x0 instead.".format(key))
                                shape = (0, 0, 0)
                            else:
                                shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                        else:
                            key = input_name
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                    if draw_type:
                        if input_node["op"] != "null":
                            key = input_name + "_output"
                            if "attrs" in input_node:
                                params = input_node["attrs"]
                                if "num_outputs" in params:
                                    key += str(int(params["num_outputs"]) - 1)
                            dtype = type_dict[key]
                            attr["label"] += '(' + dtype.__name__ + ')'
                        else:
                            key = input_name
                            dtype = type_dict[key]
                            attr["label"] += '(' + dtype.__name__ + ')'
                    dot.edge(tail_name=name, head_name=input_name, **attr)

    return dot
