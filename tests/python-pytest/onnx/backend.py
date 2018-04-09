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
import mxnet as mx
from mxnet.contrib.onnx._import.import_onnx import GraphProto
try:
    from onnx import helper, TensorProto
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
        sym, arg_params, aux_params = graph.from_onnx(MXNetBackend.make_graph(node, inputs))
        data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg_params and graph_input not in aux_params]
        data_shapes = []
        dim_change_op_types = set(['ReduceMin', 'ReduceMax', 'ReduceMean',
                                   'ReduceProd', 'ReduceSum', 'Slice', 'Pad',
                                   'Squeeze', 'Upsample', 'Reshape', 'Conv',
                                   'Concat', 'Softmax', 'Flatten', 'Transpose',
                                   'GlobalAveragePool', 'GlobalMaxPool'])

        # Adding extra dimension of batch_size 1 if the batch_size is different for multiple inputs.
        for idx, input_name in enumerate(data_names):
            batch_size = 1
            if len(inputs) > 1 and len(inputs[idx].shape) < 4 and  \
                            len(set(x.shape[0] for x in inputs)) != 1:
                tuples = ((batch_size,), inputs[idx].shape)
                new_shape = sum(tuples, ())
                data_shapes.append((input_name, new_shape))
            else:
                data_shapes.append((input_name, inputs[idx].shape))

        # create module, passing cpu context
        if device == 'CPU':
            ctx = mx.cpu()
        else:
            raise NotImplementedError("Only CPU context is supported for now")

        # create a module
        mod = mx.mod.Module(symbol=sym, data_names=data_names, context=ctx, label_names=None)
        mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)

        # initializing parameters for calculating result of each individual node
        if arg_params is None and aux_params is None:
            mod.init_params()
        else:
            mod.set_params(arg_params=arg_params, aux_params=aux_params)

        data_forward = []
        for idx, input_name in enumerate(data_names):
            # slice and pad operator tests needs 1 less dimension in forward pass
            # otherwise it will throw an error.
            # for squeeze operator, need to retain shape of input as provided
            val = inputs[idx]
            if node.op_type in dim_change_op_types:
                data_forward.append(mx.nd.array(val))
            else:
                data_forward.append(mx.nd.array([val]))

        mod.forward(mx.io.DataBatch(data_forward))
        result = mod.get_outputs()[0].asnumpy()
        if node.op_type in dim_change_op_types:
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
        sym, arg_params, aux_params = graph.from_onnx(model.graph)
        return MXNetBackendRep(sym, arg_params, aux_params, device)

    @classmethod
    def supports_device(cls, device):
        """Supports only CPU for testing"""
        return device == 'CPU'

prepare = MXNetBackend.prepare

run_node = MXNetBackend.run_node

supports_device = MXNetBackend.supports_device
