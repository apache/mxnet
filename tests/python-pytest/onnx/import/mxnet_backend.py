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
"""MXNet backend wrapper for onnx test infrastructure"""
import mxnet as mx
from mxnet.contrib.onnx._import.import_onnx import GraphProto
try:
    from onnx import helper, TensorProto
    from onnx.backend.base import Backend
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed. Instructions to"
                      + " install - https://github.com/onnx/onnx#installation")
from mxnet_backend_rep import MXNetBackendRep

# MXNetBackend class will take an ONNX model with inputs, perform a computation,
# and then return the output.
# Implemented by following onnx docs guide:
# https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md

class MXNetBackend(Backend):
    """MXNet backend for ONNX"""

    @staticmethod
    def make_graph(node, inputs):
        """ Created ONNX GraphProto from node"""
        initializer = []
        tensor_input_info = []
        tensor_output_info = []

        # Adding input tensor info.
        for index in range(len(node.input)):
            tensor_input_info.append(
                helper.make_tensor_value_info(str(node.input[index]), TensorProto.FLOAT, [1]))

            # Creating an initializer for Weight params.
            # Assumes that weight params is named as 'W'.
            if node.input[index] == 'W':
                dim = inputs[index].shape
                param_tensor = helper.make_tensor(
                    name=node.input[index],
                    data_type=TensorProto.FLOAT,
                    dims=dim,
                    vals=inputs[index].flatten())

                initializer.append(param_tensor)

        # Adding output tensor info.
        for index in range(len(node.output)):
            tensor_output_info.append(
                helper.make_tensor_value_info(str(node.output[index]), TensorProto.FLOAT, [1]))

        # creating graph proto object.
        graph_proto = helper.make_graph(
            [node],
            "test",
            tensor_input_info,
            tensor_output_info,
            initializer=initializer)

        return graph_proto

    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        """For running end to end model(used for onnx test backend)

        Parameters
        ----------
        model  : onnx ModelProto object
            loaded onnx graph
        device : 'CPU'
            specifying device to run test on
        kwargs :
            other arguments

        Returns
        -------
        MXNetBackendRep : object
            Returns object of MXNetBackendRep class which will be in turn
            used to run inference on the input model and return the result for comparison.
        """
        graph = GraphProto()
        sym, arg_params, aux_params = graph.from_onnx(model.graph)
        return MXNetBackendRep(sym, arg_params, aux_params, device)

    @classmethod
    def supports_device(cls, device):
        """Supports only CPU for testing"""
        return device == 'CPU'

prepare = MXNetBackend.prepare

run_node = MXNetBackend.run_node

supports_device = MXNetBackend.supports_device
