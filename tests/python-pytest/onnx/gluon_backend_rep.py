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
"""gluon backend rep for onnx test infrastructure"""
from collections import namedtuple
import numpy as np
try:
    from onnx.backend.base import BackendRep
except ImportError:
    raise ImportError("Onnx and protobuf need to be installed. Instructions to"
                      + " install - https://github.com/onnx/onnx#installation")
import mxnet as mx
from mxnet import nd

# Using these functions for onnx test infrastructure.
# Implemented by following onnx docs guide:
# https://github.com/onnx/onnx/blob/master/docs/Implementing%20an%20ONNX%20backend.md
# MXNetBackendRep object will be returned by MXNetBackend's prepare method which is used to
# execute a model repeatedly.
# Inputs will be passed to the run method of MXNetBackendRep class, it will perform computation and
# retrieve the corresponding results for comparison to the onnx backend.
# https://github.com/onnx/onnx/blob/master/onnx/backend/test/runner/__init__.py.

class GluonBackendRep(BackendRep):
    """Running model inference on mxnet engine and return the result
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
            raise NotImplementedError("Only CPU context is supported for now")

        # run inference
        net_outputs = self.net(nd.array(inputs[0], ctx=ctx))
        results = []
        results.extend([o for o in net_outputs.asnumpy()])
        result = np.array(results)

        return [result]
