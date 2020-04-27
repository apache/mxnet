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
"""MXNet backend rep for onnx test infrastructure"""
try:
    from onnx.backend.base import BackendRep
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed. Instructions to"
                      + " install - https://github.com/onnx/onnx#installation")
import numpy as np
import mxnet as mx
from mxnet import nd

# Using these functions for onnx test infrastructure.
# Implemented by following onnx docs guide:
# https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md
# MXNetBackendRep object will be returned by MXNetBackend's prepare method which is used to
# execute a model repeatedly.
# Inputs will be passed to the run method of MXNetBackendRep class, it will perform computation and
# retrieve the corresponding results for comparison to the onnx backend.
# https://github.com/onnx/onnx/blob/master/onnx/backend/test/runner/__init__.py.

class MXNetBackendRep(BackendRep):
    """Running model inference on mxnet engine and return the result
     to onnx test infrastructure for comparison."""
    def __init__(self, symbol, arg_params, aux_params, device):
        self.symbol = symbol
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.device = device

    def run(self, inputs, **kwargs):
        """Run model inference and return the result

        Parameters
        ----------
        inputs : numpy array
            input to run a layer on

        Returns
        -------
        params : numpy array
            result obtained after running the inference on mxnet
        """
        # create module, passing cpu context
        if self.device == 'CPU':
            ctx = mx.cpu()
        else:
            raise NotImplementedError("ONNX tests are run only for CPU context.")

        # To fetch the data names of the input to the model we list the inputs of the symbol graph
        # and exclude the argument and auxiliary parameters from the list
        data_names = [graph_input for graph_input in self.symbol.list_inputs()
                      if graph_input not in self.arg_params and graph_input not in self.aux_params]

        data_forward = []
        for idx, input_name in enumerate(data_names):
            val = inputs[idx]
            data_forward.append(mx.nd.array(val))

        if self.arg_params:
            for idx, input_name in enumerate(self.arg_params):
                val = self.arg_params[input_name]
                data_names.append(input_name)
                data_forward.append(mx.nd.array(val))

        args = dict(zip(data_names, data_forward))
        exe = self.symbol.bind(ctx, args=args, aux_states=self.aux_params)
        exe.forward(is_train=False)
        result = []
        for output in exe.outputs:
            result.append(output.asnumpy())
        return result


# GluonBackendRep object will be returned by GluonBackend's prepare method which is used to
# execute a model repeatedly.
# Inputs will be passed to the run method of MXNetBackendRep class, it will perform computation and
# retrieve the corresponding results for comparison to the onnx backend.
# https://github.com/onnx/onnx/blob/master/onnx/backend/test/runner/__init__.py.
# Implemented by following onnx docs guide:
# https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md

class GluonBackendRep(BackendRep):
    """Running model inference on gluon backend and return the result
     to onnx test infrastructure for comparison."""
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def run(self, inputs, **kwargs):
        """Run model inference and return the result

        Parameters
        ----------
        inputs : numpy array
            input to run a layer on

        Returns
        -------
        params : numpy array
            result obtained after running the inference on mxnet
        """
        # create module, passing cpu context
        if self.device == 'CPU':
            ctx = mx.cpu()
        else:
            raise NotImplementedError("ONNX tests are run only for CPU context.")

        # run inference
        net_inputs = [nd.array(input_data, ctx=ctx) for input_data in inputs]
        net_outputs = self.net(*net_inputs)
        results = []
        if isinstance(net_outputs, list):
            for output in net_outputs:
                results.append(output.asnumpy())
            result = results
        else:
            results.extend([o for o in net_outputs.asnumpy()])
            result = [np.array(results)]

        return result
