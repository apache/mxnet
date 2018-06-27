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
"""backend wrapper for onnx test infrastructure"""
import numpy as np
from mxnet.contrib.onnx.onnx2mx.import_onnx import GraphProto
from mxnet.contrib.onnx.mx2onnx.export_onnx import MXNetGraph
try:
    from onnx import helper, TensorProto, mapping
    from onnx.backend.base import Backend
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed")
from backend_rep import MXNetBackendRep

# Using these functions for onnx test infrastructure.
# Implemented by following onnx docs guide:
# https://github.com/onnx/onnx/blob/master/docs/Implementing%20an%20ONNX%20backend.md
# MXNetBackend class will take an ONNX model with inputs, perform a computation,
# and then return the output.

class MXNetBackend(Backend):
    """MXNet backend for ONNX"""

    @staticmethod
    def perform_import_export(graph_proto, input_shape):
        """ Import ONNX model to mxnet model and then export to ONNX model
            and then import it back to mxnet for verifying the result"""
        graph = GraphProto()

        sym, arg_params, aux_params = graph.from_onnx(graph_proto)

        params = {}
        params.update(arg_params)
        params.update(aux_params)
        # exporting to onnx graph proto format
        converter = MXNetGraph()
        graph_proto = converter.create_onnx_graph_proto(sym, params, in_shape=input_shape, in_type=mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')])

        # importing back to MXNET for verifying result.
        sym, arg_params, aux_params = graph.from_onnx(graph_proto)

        return sym, arg_params, aux_params


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
        metadata = graph.get_graph_metadata(model.graph)
        input_data = metadata['input_tensor_data']
        input_shape = [data[1] for data in input_data]
        sym, arg_params, aux_params = MXNetBackend.perform_import_export(model.graph, input_shape)
        return MXNetBackendRep(sym, arg_params, aux_params, device)

    @classmethod
    def supports_device(cls, device):
        """Supports only CPU for testing"""
        return device == 'CPU'


prepare = MXNetBackend.prepare

run_node = MXNetBackend.run_node

supports_device = MXNetBackend.supports_device
