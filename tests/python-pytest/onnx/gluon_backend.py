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
"""Gluon backend wrapper for onnx test infrastructure"""
import mxnet as mx
from mxnet import nd
from mxnet.contrib.onnx._import.import_onnx import GraphProto
import numpy as np
try:
    from onnx import helper, TensorProto
    from onnx.backend.base import Backend
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed. Instructions to"
                      + " install - https://github.com/onnx/onnx#installation")
from gluon_backend_rep import GluonBackendRep

# GluonBackend class will take an ONNX model with inputs, perform a computation,
# and then return the output.
# Implemented by following onnx docs guide:
# https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md

class GluonBackend(Backend):
    """Gluon backend for ONNX"""

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
    def run_node(cls, node, inputs, device='CPU'):
        """Running individual node inference on gluon backend and
        return the result to onnx test infrastructure.

        Parameters
        ----------
        node   : onnx node object
            loaded onnx node (individual layer)
        inputs : numpy array
            input to run a node on
        device : 'CPU'
            device to run a node on

        Returns
        -------
        params : numpy array
            result obtained after running the operator
        """
        graph = GraphProto()
        net = graph.graph_to_gluon(GluonBackend.make_graph(node, inputs))

        # create module, passing cpu context
        if device == 'CPU':
            ctx = mx.cpu()
        else:
            raise NotImplementedError("Only CPU context is supported for now")

        if node.op_type in ['Conv']:
            inputs = inputs[:1]
        net_inputs = [nd.array(input_data, ctx=ctx) for input_data in inputs]
        net_outputs = net(*net_inputs)
        results = []
        results.extend([o for o in net_outputs.asnumpy()])
        result = np.array(results)
        return [result]

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
        GluonBackendRep : object
            Returns object of GluonBackendRep class which will be in turn
            used to run inference on the input model and return the result for comparison.
        """
        graph = GraphProto()
        net = graph.graph_to_gluon(model.graph)
        return GluonBackendRep(net, device)

    @classmethod
    def supports_device(cls, device):
        """Supports only CPU for testing"""
        return device == 'CPU'

prepare = GluonBackend.prepare

run_node = GluonBackend.run_node

supports_device = GluonBackend.supports_device
