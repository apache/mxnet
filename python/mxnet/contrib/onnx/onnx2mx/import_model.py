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
"""Functions for importing ONNX models to MXNet and for checking metadata"""
# pylint: disable=no-member

from .import_onnx import GraphProto

def import_model(model_file):
    """Imports the ONNX model file, passed as a parameter, into MXNet symbol and parameters.
    Operator support and coverage -
    https://cwiki.apache.org/confluence/display/MXNET/ONNX+Operator+Coverage

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

    Notes
    -----
    This method is available when you ``import mxnet.contrib.onnx``

    """
    graph = GraphProto()

    try:
        import onnx
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")
    # loads model file and returns ONNX protobuf object
    model_proto = onnx.load_model(model_file)
    model_opset_version = max([x.version for x in model_proto.opset_import])
    sym, arg_params, aux_params = graph.from_onnx(model_proto.graph, opset_version=model_opset_version)
    return sym, arg_params, aux_params

def get_model_metadata(model_file):
    """
    Returns the name and shape information of input and output tensors of the given ONNX model file.

    Notes
    -----
    This method is available when you ``import mxnet.contrib.onnx``

    Parameters
    ----------
    model_file : str
        ONNX model file name

    Returns
    -------
    model_metadata : dict
        A dictionary object mapping various metadata to its corresponding value.
        The dictionary will have the following template::

          'input_tensor_data' : list of tuples representing the shape of the input paramters
          'output_tensor_data' : list of tuples representing the shape of the output of the model
    """
    graph = GraphProto()

    try:
        import onnx
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")
    model_proto = onnx.load_model(model_file)
    metadata = graph.get_graph_metadata(model_proto.graph)
    return metadata
