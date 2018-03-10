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
    """Imports the supplied ONNX model file into MXNet symbol and parameters.
    :parameters model_file
    ----------
    model_file : ONNX model file name

    :returns (sym, params)
    -------
    sym : mx.symbol
        Compatible mxnet symbol
    params : dict of str to mx.ndarray
        Dict of converted parameters stored in mx.ndarray format
    """
    graph = GraphProto()

    # loads model file and returns ONNX protobuf object
    try:
        import onnx
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed")
    model_proto = onnx.load(model_file)
    sym, params = graph.from_onnx(model_proto.graph)
    return sym, params
