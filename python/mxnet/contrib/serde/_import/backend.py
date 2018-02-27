# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# coding: utf-8
# pylint: disable=too-many-locals,invalid-name
"""backend wrapper for onnx test infrastructure"""
from collections import namedtuple
from onnx.backend.base import Backend
from .import_onnx import GraphProto
from .backend_rep import MXNetBackendRep
from .... import context
from .... import module
from .... import ndarray as nd

# Using these functions for onnx test infrastructure.
# Implemented by following onnx docs guide:
# https://github.com/onnx/onnx/blob/master/docs/Implementing%20an%20ONNX%20backend.md
# MXNetBackend class will take an ONNX model with inputs, perform a computation,
# and then return the output.

class MXNetBackend(Backend):
    """MXNet backend for ONNX"""
    @classmethod
    def run_node(cls, node, inputs, device='CPU'):
        """Running individual node inference on mxnet engine and
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
        sym = graph.run_node(node)
        data_names = [i for i in node.input]
        data_shapes = []
        reduce_op_types = set(['ReduceMin', 'ReduceMax', 'ReduceMean',
                               'ReduceProd', 'ReduceSum', 'Slice', 'Pad',
                               'Squeeze', 'Upsample', 'Reshape'])

        # Adding extra dimension of batch_size 1 if the batch_size is different for multiple inputs.
        for idx, input_name in enumerate(data_names):
            batch_size = 1
            if len(inputs[idx].shape) < 4 and len(inputs) > 1 and \
                            len(set(x.shape[0] for x in inputs)) != 1:
                tuples = ((batch_size,), inputs[idx].shape)
                new_shape = sum(tuples, ())
                data_shapes.append((input_name, new_shape))
            else:
                data_shapes.append((input_name, inputs[idx].shape))

        # create module, passing cpu context
        if device == 'CPU':
            ctx = context.cpu()
        else:
            raise NotImplementedError("Only CPU context is supported for now")

        # create a module
        mod = module.Module(symbol=sym, data_names=data_names, context=ctx, label_names=None)
        mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)

        # initializing parameters for calculating result of each individual node
        mod.init_params()

        batch = namedtuple('Batch', ['data'])

        data_forward = []
        for val in inputs:
            # slice and pad operator tests needs 1 less dimension in forward pass
            # otherwise it will throw an error.
            # for squeeze operator, need to retain shape of input as provided
            if node.op_type in reduce_op_types:
                data_forward.append(nd.array(val))
            else:
                data_forward.append(nd.array([val]))

        mod.forward(batch(data_forward))
        result = mod.get_outputs()[0].asnumpy()
        if node.op_type in reduce_op_types:
            return [result]
        return result

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
        sym, params = graph.from_onnx(model.graph)
        return MXNetBackendRep(sym, params, device)

    @classmethod
    def supports_device(cls, device):
        """Supports only CPU for testing"""
        return device == 'CPU'

prepare = MXNetBackend.prepare

run_node = MXNetBackend.run_node

supports_device = MXNetBackend.supports_device
