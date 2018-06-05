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
# pylint: disable=invalid-name,too-many-locals,no-self-use,too-many-arguments
"""MXNet to ONNX graph converter functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import numpy as np

from onnx import (checker, helper, onnx_pb2)
from onnx.helper import make_tensor_value_info

from .... import context
from .... import ndarray as nd
from .... import io
from .... import module as mod


class MxNetToONNXConverter:
    """Class to convert MXNet to ONNX graph"""
    registry_ = {}
    input_output_maps_ = {}

    def __init__(self):
        # topologically sorted nodes
        self.nodes = []
        self.input_tensors = []
        self.output_tensors = []

    @staticmethod
    def register(op_name):
        """Register operator"""
        def wrapper(func):
            """Helper function to map functions"""
            MxNetToONNXConverter.registry_[op_name] = func
            return func

        return wrapper

    @staticmethod
    def convert_layer(node, **kwargs):
        """Convert MXNet layer to ONNX"""
        op = str(node["op"])
        if op not in MxNetToONNXConverter.registry_:
            raise AttributeError("No conversion function registered for op type %s yet." % op)
        convert_fun = MxNetToONNXConverter.registry_[op]
        return convert_fun(node, **kwargs)

    @staticmethod
    def forward_pass(inputs, sym, arg_params, aux_params):
        """ Do a forward pass based on the sym and params"""
        data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg_params and graph_input not in aux_params
                      and graph_input != 'softmax_label']

        data_shapes = []
        # Adding extra dimension of batch_size 1 if the batch_size is different for multiple inputs.
        for idx, input_name in enumerate(data_names):
            data_shapes.append((input_name, inputs[idx].shape))

        # create module, passing cpu context
        ctx = context.cpu()
        test_mod = mod.Module(symbol=sym, data_names=data_names, context=ctx, label_names=None)
        test_mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)

        # initializing parameters for calculating result of each individual node
        if arg_params is None and aux_params is None:
            test_mod.init_params()
        else:
            test_mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)

        data_forward = []
        for idx, input_name in enumerate(data_names):
            val = inputs[idx]
            data_forward.append(nd.array(val))

        test_mod.forward(io.DataBatch(data_forward))
        result = test_mod.get_outputs()[0].asnumpy()

        return result.shape


    @staticmethod
    def split_params(sym, params):
        """splitting params into args and aux params"""
        arg_params = {}
        aux_params = {}
        for args in sym.list_arguments():
            if args in params:
                arg_params.update({args: nd.array(params[args])})
        for aux in sym.list_auxiliary_states():
            if aux in params:
                aux_params.update({aux: nd.array(params[aux])})
        return arg_params, aux_params


    @staticmethod
    def infer_output_shape(sym, params, in_shape):
        """Infer output shape by doing a forward pass using dummy inputs """
        #create dummy input
        inputs = [np.random.randn(*input_shape) for input_shape in in_shape]
        arg, aux = MxNetToONNXConverter.split_params(sym, params)
        return MxNetToONNXConverter.forward_pass(inputs, sym, arg, aux)


    @staticmethod
    def convert_weights_to_numpy(weights_dict):
        """Convert weights to numpy"""
        return dict([(k.replace("arg:", "").replace("aux:", ""), v.asnumpy())
                     for k, v in weights_dict.items()])

    def convert_mx2onnx_graph(self, sym, params, in_shape, in_type, log=False):
        """Convert MXNet graph to ONNX graph"""

        # Determine output shape
        output_shape = MxNetToONNXConverter.infer_output_shape(sym, params, in_shape)

        weights = MxNetToONNXConverter.convert_weights_to_numpy(params)

        mx_graph = json.loads(sym.tojson())["nodes"]

        initializer = []
        all_processed_nodes = []
        onnx_processed_nodes = []
        onnx_processed_inputs = []
        onnx_processed_outputs = []
        index_lookup = []

        graph_input_idx = 0
        for idx, node in enumerate(mx_graph):
            op = node["op"]
            name = node["name"]
            if log:
                print("Converting idx: %d, op: %s, name: %s" % (idx, op, name))

            if op == "null" and name not in params:
                #Handling graph input
                if name == 'softmax_label':
                    continue
                converted = MxNetToONNXConverter.convert_layer(
                    node,
                    is_input=True,
                    mx_graph=mx_graph,
                    weights=weights,
                    in_shape=in_shape[graph_input_idx],
                    in_type=in_type,
                    proc_nodes=all_processed_nodes,
                    initializer=initializer,
                    index_lookup=index_lookup)
                graph_input_idx += 1

            else:
                converted = MxNetToONNXConverter.convert_layer(
                    node,
                    is_input=False,
                    mx_graph=mx_graph,
                    weights=weights,
                    in_shape=in_shape,
                    in_type=in_type,
                    proc_nodes=all_processed_nodes,
                    initializer=initializer,
                    index_lookup=index_lookup,
                    idx=idx
                )

            if isinstance(converted, list):
                for converted_node in converted:
                    if isinstance(converted_node, onnx_pb2.ValueInfoProto):
                        onnx_processed_inputs.append(converted_node)
                    elif isinstance(converted_node, onnx_pb2.NodeProto):
                        if idx < (len(mx_graph) - 1):
                            onnx_processed_nodes.append(converted_node)
                        else:
                            onnx_processed_nodes.append(converted_node)
                            if not converted_node.name:
                                onnx_processed_outputs.append(
                                    make_tensor_value_info(
                                        name=converted_node.output[0],
                                        elem_type=in_type,
                                        shape=output_shape
                                    )
                                )
                            else:
                                onnx_processed_outputs.append(
                                    make_tensor_value_info(
                                        name=converted_node.name,
                                        elem_type=in_type,
                                        shape=output_shape
                                    )
                                )
                            if log:
                                print("Output node is: %s" % converted_node.name)
                    elif isinstance(converted_node, onnx_pb2.TensorProto):
                        raise ValueError("Did not expect TensorProto")
                    else:
                        raise ValueError("node is of an unrecognized type: %s" % type(node))

                    all_processed_nodes.append(converted_node)

                if idx > 0:
                    if name != 'softmax':
                        prev_index = index_lookup[idx-1]
                    else:
                        prev_index = index_lookup[idx - 2]
                    index_lookup.append(prev_index+len(converted))
                else:
                    index_lookup.append(len(converted) - 1)

        graph = helper.make_graph(
            onnx_processed_nodes,
            "main",
            onnx_processed_inputs,
            onnx_processed_outputs
        )

        graph.initializer.extend(initializer)

        checker.check_graph(graph)
        return graph
