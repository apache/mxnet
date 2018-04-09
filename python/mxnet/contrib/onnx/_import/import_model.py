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
