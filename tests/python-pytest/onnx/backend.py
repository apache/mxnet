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
"""MXNet/Gluon backend wrapper for onnx test infrastructure"""

from mxnet.contrib.onnx.onnx2mx.import_onnx import GraphProto
from mxnet.contrib.onnx.mx2onnx.export_onnx import MXNetGraph
import mxnet as mx
import numpy as np

try:
    from onnx import helper, TensorProto, mapping
    from onnx.backend.base import Backend
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed. Instructions to"
                      + " install - https://github.com/onnx/onnx#installation")
from backend_rep import MXNetBackendRep, GluonBackendRep


# MXNetBackend class will take an ONNX model with inputs, perform a computation,
# and then return the output.
# Implemented by following onnx docs guide:
# https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md

class MXNetBackend(Backend):
    """MXNet/Gluon backend for ONNX"""

    backend = 'mxnet'
    operation = 'import'

    @classmethod
    def set_params(cls, backend, operation):
        cls.backend = backend
        cls.operation = operation

    @staticmethod
    def perform_import_export(sym, arg_params, aux_params, input_shape):
        """ Import ONNX model to mxnet model and then export to ONNX model
            and then import it back to mxnet for verifying the result"""
        graph = GraphProto()

        params = {}
        params.update(arg_params)
        params.update(aux_params)
        # exporting to onnx graph proto format
        converter = MXNetGraph()
        graph_proto = converter.create_onnx_graph_proto(sym, params, in_shape=input_shape,
                                                        in_type=mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')])

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
        backend = kwargs.get('backend', cls.backend)
        operation = kwargs.get('operation', cls.operation)

        graph = GraphProto()
        if device == 'CPU':
            ctx = mx.cpu()
        else:
            raise NotImplementedError("ONNX tests are run only for CPU context.")

        if backend == 'mxnet':
            sym, arg_params, aux_params = graph.from_onnx(model.graph)
            if operation == 'export':
                metadata = graph.get_graph_metadata(model.graph)
                input_data = metadata['input_tensor_data']
                input_shape = [data[1] for data in input_data]
                sym, arg_params, aux_params = MXNetBackend.perform_import_export(sym, arg_params, aux_params,
                                                                                 input_shape)

            return MXNetBackendRep(sym, arg_params, aux_params, device)
        elif backend == 'gluon':
            if operation == 'import':
                net = graph.graph_to_gluon(model.graph, ctx)
                return GluonBackendRep(net, device)
            elif operation == 'export':
                raise NotImplementedError("Gluon->ONNX export not implemented.")

    @classmethod
    def supports_device(cls, device):
        """Supports only CPU for testing"""
        return device == 'CPU'


prepare = MXNetBackend.prepare

supports_device = MXNetBackend.supports_device
