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
# pylint: disable=invalid-name,too-many-locals,no-self-use

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import sys

from onnx import (defs, checker, helper, numpy_helper, mapping, onnx_pb2,
                  ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto)

from onnx.helper import make_tensor, make_tensor_value_info

class MxNetToONNXConverter:
    registry_ = {}
    input_output_maps_ = {}

    def __init__(self):
        # topologically sorted nodes
        self.nodes = []
        self.input_tensors = []
        self.output_tensors = []

    @staticmethod
    def register(op_name):

        def wrapper(func):
            MxNetToONNXConverter.registry_[op_name] = func
            return func

        return wrapper

    @staticmethod
    def convert_layer(node, **kwargs):
        op = str(node["op"])
        if op not in MxNetToONNXConverter.registry_:
            raise AttributeError("No conversion function registered for op type %s yet." % op)
        convert_fun = MxNetToONNXConverter.registry_[op]
        return convert_fun(node, **kwargs)

    # Add transpose?
    @staticmethod
    def convert_weights_to_numpy(weights_dict):
        return dict([(k.replace("arg:", "").replace("aux:", ""), v.asnumpy()) for k, v in weights_dict.items()])

    def convert_mx2onnx_graph(self, mx_graph, mx_weights, in_shape, in_type, log=False):
        print("\nconverting weights from MxNet NDArrays to NumPy arrays.\n")
        weights = MxNetToONNXConverter.convert_weights_to_numpy(mx_weights)

        onnx_graph = GraphProto()

        initializer = []
        all_processed_nodes = []
        onnx_processed_nodes = []
        onnx_processed_inputs = []
        onnx_processed_outputs = []

        for idx, node in enumerate(mx_graph):
            op = node["op"]
            name = node["name"]
            if log:
                print("Converting idx: %d, op: %s, name: %s" % (idx, op, name))
            converted = MxNetToONNXConverter.convert_layer(
                node,
                mx_graph=mx_graph,
                weights=weights,
                in_shape=in_shape,
                in_type=in_type,
                proc_nodes=all_processed_nodes,
                initializer=initializer
            )

            if isinstance(converted, onnx_pb2.ValueInfoProto):
                if idx < (len(mx_graph) - 1):
                    onnx_processed_inputs.append(converted)
                else:
                    onnx_processed_outputs.append(converted)
            elif isinstance(converted, onnx_pb2.NodeProto):
                if idx < (len(mx_graph) - 1):
                    onnx_processed_nodes.append(converted)
                else:
                    onnx_processed_nodes.append(converted)
                    onnx_processed_outputs.append(
                        make_tensor_value_info(
                            name=converted.name,
                            elem_type=mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
                            shape=(in_shape[0], -1)
                        )
                    )
                    if log:
                        print("Output node is: %s" % converted.name)
            elif isinstance(converted, onnx_pb2.TensorProto):
                raise ValueError("Did not expect TensorProto")
                if idx < (len(mx_graph) - 1):
                    onnx_processed_inputs.append(converted)
                else:
                    onnx_processed_outputs.append(converted)
            else:
                print(converted)
                raise ValueError("node is of an unrecognized type: %s" % type(node))

            all_processed_nodes.append(converted)

        graph = helper.make_graph(
            onnx_processed_nodes,
            "main",
            onnx_processed_inputs,
            onnx_processed_outputs
        )

        graph.initializer.extend(initializer)

        checker.check_graph(graph)
        return graph
