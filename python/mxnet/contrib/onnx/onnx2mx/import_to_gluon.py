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
"""Import ONNX model to gluon interface"""
# pylint: disable=no-member

from .import_onnx import GraphProto

def import_to_gluon(model_file, ctx):
    """
    Imports the ONNX model files, passed as a parameter, into Gluon SymbolBlock object.

    Parameters
    ----------
    model_file : str
        ONNX model file name
    ctx : Context or list of Context
        Loads the model into one or many context(s).

    Returns
    -------
    sym_block : :class:`~mxnet.gluon.SymbolBlock`
        A SymbolBlock object representing the given model file.

    Notes
    -----
    This method is available when you ``import mxnet.contrib.onnx``

    """
    graph = GraphProto()
    try:
        import onnx
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. Instructions to"
                          + " install - https://github.com/onnx/onnx#installation")
    model_proto = onnx.load_model(model_file)
    model_opset_version = max([x.version for x in model_proto.opset_import])
    net = graph.graph_to_gluon(model_proto.graph, ctx, model_opset_version)
    return net
