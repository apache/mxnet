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
"""import function"""
# pylint: disable=no-member

from .import_onnx import GraphProto

def import_model(model_file):
    """Imports the ONNX model file, passed as a parameter, into MXNet symbol and parameters.
    Operator support and coverage - https://cwiki.apache.org/confluence/display/MXNET/ONNX

    Parameters
    ----------
    model_file : str
        ONNX model file name

    Returns
    -------
    sym : :class:`~mxnet.symbol.Symbol`
        MXNet symbol object

    arg_params : dict of ``str`` to :class:`~mxnet.ndarray.NDArray`
        Dict of converted parameters stored in ``mxnet.ndarray.NDArray`` format

    aux_params : dict of ``str`` to :class:`~mxnet.ndarray.NDArray`
        Dict of converted parameters stored in ``mxnet.ndarray.NDArray`` format
    """
    graph = GraphProto()

    try:
        import onnx
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")
    # loads model file and returns ONNX protobuf object
    model_proto = onnx.load(model_file)
    sym, arg_params, aux_params = graph.from_onnx(model_proto.graph)
    return sym, arg_params, aux_params

def get_model_metadata(model_file):
    """
    Returns the name and shape information of input and output tensors of the given ONNX model file.
    
    Parameters
    ----------
    model_file : str
        ONNX model file name

    Returns
    -------
    model_metadata – A dictionary object mapping various metadata to its corresponding value.
    The dictionary will have the following template.
    {
        “input_tensor_data”: <list of tuples representing the shape of the input paramters>,
        “output_tensor_data”: <list of tuples representing the shape of the output of the model>
    }
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")
    model_proto = onnx.load(model_file)
    graph = model_proto.graph

    _params = set()
    for tensor_vals in graph.initializer:
        _params.add(tensor_vals.name)

    input_data = []
    for graph_input in graph.input:
        shape = []
        if graph_input.name not in _params:
            for val in graph_input.type.tensor_type.shape.dim:
                shape.append(val.dim_value)
            input_data.append((graph_input.name, tuple(shape)))

    output_data = []
    for graph_out in graph.output:
        shape = []
        for val in graph_out.type.tensor_type.shape.dim:
            shape.append(val.dim_value)
        output_data.append((graph_out.name, tuple(shape)))
    metadata = {'input_tensor_data' : input_data,
                'output_tensor_data' : output_data
               }
    return metadata
