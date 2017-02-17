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

from .symbol import Symbol

def _str2tuple(string):
    """convert shape string to list, internal use only

    Parameters
    ----------
    string: str
        shape string

    Returns
    -------
    list of str to represent shape
    """
    return re.findall(r"\d+", string)

def print_summary(symbol, shape=None, line_length=120, positions=[.44, .64, .74, 1.]):
    """convert symbol for detail information

    Parameters
    ----------
    symbol: Symbol
        symbol to be visualized
    shape: dict
        dict of shapes, str->shape (tuple), given input shapes
    line_length: int
        total length of printed lines
    positions: list
        relative or absolute positions of log elements in each line
    Returns
    ------
        void
    """
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    show_shape = False
    if shape != None:
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
    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Previous Layer']
    def print_row(fields, positions):
        """print format row

        Parameters
        ----------
        fields: list
            information field
        positions: list
            field length ratio
        Returns
        ------
            void
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
            node information
        out_shape: dict
            node shape information
        Returns
        ------
            node total parameters
        """
        op = node["op"]
        pre_node = []
        pre_filter = 0
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
                            pre_filter = pre_filter + int(shape[0])
        cur_param = 0
        if op == 'Convolution':
            cur_param = pre_filter \
                * int(_str2tuple(node["attr"]["kernel"])[0]) \
                * int(_str2tuple(node["attr"]["kernel"])[1]) \
                * int(node["attr"]["num_filter"]) \
                + int(node["attr"]["num_filter"])
        elif op == 'FullyConnected':
            cur_param = pre_filter * (int(node["attr"]["num_hidden"]) + 1)
        elif op == 'BatchNorm':
            key = node["name"] + "_output"
            if show_shape:
                num_filter = shape_dict[key][1]
                cur_param = int(num_filter) * 2
        if not pre_node:
            first_connection = ''
        else:
            first_connection = pre_node[0]
        fields = [node['name'] + '(' + op + ')',
                  "x".join([str(x) for x in out_shape]),
                  cur_param,
                  first_connection]
        print_row(fields, positions)
        if len(pre_node) > 1:
            for i in range(1, len(pre_node)):
                fields = ['', '', '', pre_node[i]]
                print_row(fields, positions)
        return cur_param
    total_params = 0
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
        total_params += print_layer_summary(nodes[i], out_shape)
        if i == len(nodes) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)
    print('Total params: %s' % total_params)
    print('_' * line_length)

def plot_network(symbol, title="plot", save_format='pdf', shape=None, node_attrs={},
                 hide_weights=True):
    """convert symbol to dot object for visualization

    Parameters
    ----------
    title: str
        title of the dot graph
    symbol: Symbol
        symbol to be visualized
    shape: dict
        If supplied, the visualization will include the shape
        of each tensor on the edges between nodes.
        This is a dict of shapes, str->shape (tuple), given input shapes
    node_attrs: dict
        dict of node's attributes
        for example:
            node_attrs={"shape":"oval","fixedsize":"fasle"}
            means to plot the network in "oval"
    hide_weights: bool
        if True (default) then inputs with names like `*_weight`
        or `*_bias` will be hidden

    Returns
    ------
    dot: Diagraph
        dot object of symbol
    """
    # todo add shape support
    try:
        from graphviz import Digraph
    except:
        raise ImportError("Draw network requires graphviz library")
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    draw_shape = False
    if shape != None:
        draw_shape = True
        interals = symbol.get_internals()
        _, out_shapes, _ = interals.infer_shape(**shape)
        if out_shapes is None:
            raise ValueError("Input shape is incomplete")
        shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    # default attributes of node
    node_attr = {"shape": "box", "fixedsize": "true",
                 "width": "1.3", "height": "0.8034", "style": "filled"}
    # merge the dict provided by user and the default one
    node_attr.update(node_attrs)
    dot = Digraph(name=title, format=save_format)
    # color map
    cm = ("#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
          "#fdb462", "#b3de69", "#fccde5")

    def looks_like_weight(name):
        """Internal helper to figure out if node should be hidden with hide_weights
        """
        if name.endswith("_weight"):
            return True
        if name.endswith("_bias"):
            return True
        return False

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
            label = r"Convolution\n%sx%s/%s, %s" % (_str2tuple(node["attr"]["kernel"])[0],
                                                    _str2tuple(node["attr"]["kernel"])[1],
                                                    _str2tuple(node["attr"]["stride"])[0]
                                                    if "stride" in node["attr"] else '1',
                                                    node["attr"]["num_filter"])
            attr["fillcolor"] = cm[1]
        elif op == "FullyConnected":
            label = r"FullyConnected\n%s" % node["attr"]["num_hidden"]
            attr["fillcolor"] = cm[1]
        elif op == "BatchNorm":
            attr["fillcolor"] = cm[3]
        elif op == "Activation" or op == "LeakyReLU":
            label = r"%s\n%s" % (op, node["attr"]["act_type"])
            attr["fillcolor"] = cm[2]
        elif op == "Pooling":
            label = r"Pooling\n%s, %sx%s/%s" % (node["attr"]["pool_type"],
                                                _str2tuple(node["attr"]["kernel"])[0],
                                                _str2tuple(node["attr"]["kernel"])[1],
                                                _str2tuple(node["attr"]["stride"])[0]
                                                if "stride" in node["attr"] else '1')
            attr["fillcolor"] = cm[4]
        elif op == "Concat" or op == "Flatten" or op == "Reshape":
            attr["fillcolor"] = cm[5]
        elif op == "Softmax":
            attr["fillcolor"] = cm[6]
        else:
            attr["fillcolor"] = cm[7]
            if op == "Custom":
                label = node["attr"]["op_type"]

        dot.node(name=name, label=label, **attr)

    # add edges
    for node in nodes:
        op = node["op"]
        name = node["name"]
        if op == "null":
            continue
        else:
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                if input_name not in hidden_nodes:
                    attr = {"dir": "back", 'arrowtail':'open'}
                    # add shapes
                    if draw_shape:
                        if input_node["op"] != "null":
                            key = input_name + "_output"
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                        else:
                            key = input_name
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                    dot.edge(tail_name=name, head_name=input_name, **attr)

    return dot
