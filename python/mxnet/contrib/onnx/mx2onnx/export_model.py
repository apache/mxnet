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
#pylint: disable-msg=too-many-arguments

"""Exports an MXNet model to the ONNX model format"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
import numpy as np

from ....base import string_types
from .... import symbol
from .export_onnx import MXNetGraph
from ._export_helper import load_module


def export_model(sym, params, input_shape, input_type=np.float32,
                 onnx_file_path='model.onnx', verbose=False):
    """Exports the MXNet model file, passed as a parameter, into ONNX model.
    Accepts both symbol,parameter objects as well as json and params filepaths as input.
    Operator support and coverage -
    https://cwiki.apache.org/confluence/display/MXNET/MXNet-ONNX+Integration

    Parameters
    ----------
    sym : str or symbol object
        Path to the json file or Symbol object
    params : str or symbol object
        Path to the params file or params dictionary. (Including both arg_params and aux_params)
    input_shape : List of tuple
        Input shape of the model e.g [(1,3,224,224)]
    input_type : data type
        Input data type e.g. np.float32
    onnx_file_path : str
        Path where to save the generated onnx file
    verbose : Boolean
        If true will print logs of the model conversion

    Returns
    -------
    onnx_file_path : str
        Onnx file path

    Notes
    -----
    This method is available when you ``import mxnet.contrib.onnx``

    """

    try:
        from onnx import helper, mapping
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")

    converter = MXNetGraph()

    data_format = np.dtype(input_type)
    # if input parameters are strings(file paths), load files and create symbol parameter objects
    if isinstance(sym, string_types) and isinstance(params, string_types):
        logging.info("Converting json and weight file to sym and params")
        sym_obj, params_obj = load_module(sym, params)
        onnx_graph = converter.create_onnx_graph_proto(sym_obj, params_obj, input_shape,
                                                       mapping.NP_TYPE_TO_TENSOR_TYPE[data_format],
                                                       verbose=verbose)
    elif isinstance(sym, symbol.Symbol) and isinstance(params, dict):
        onnx_graph = converter.create_onnx_graph_proto(sym, params, input_shape,
                                                       mapping.NP_TYPE_TO_TENSOR_TYPE[data_format],
                                                       verbose=verbose)
    else:
        raise ValueError("Input sym and params should either be files or objects")

    # Create the model (ModelProto)
    onnx_model = helper.make_model(onnx_graph)

    # Save model on disk
    with open(onnx_file_path, "wb") as file_handle:
        serialized = onnx_model.SerializeToString()
        file_handle.write(serialized)
        logging.info("Input shape of the model %s ", input_shape)
        logging.info("Exported ONNX file %s saved to disk", onnx_file_path)

    return onnx_file_path
