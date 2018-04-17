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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import defs, checker, helper, numpy_helper, mapping
from .export_onnx import MxNetToONNXConverter

import mxnet as mx
import numpy as np

def load_module(json_path, params_path, input_shape):
    if not (os.path.isfile(json_path) and os.path.isfile(params_path)):
        raise ValueError("Provide valid path to the json and params file")
    else:
        model_name = json_path.rsplit('.')[0].rsplit('-', 1)[0]
        num_epochs = int(params_path.rsplit('.')[0].rsplit('-', 1)[1])
        trained_model = mx.mod.Module.load(model_name, num_epochs)
        trained_model.bind(data_shapes=[('data', input_shape)], label_shapes=None, for_training=False, force_rebind=True)

        sym = trained_model.symbol
        arg_params, aux_params = trained_model.get_params()

        # Merging arg and aux parameters
        arg_params.update(aux_params)
        return sym, arg_params

def export_model(model, weights, input_shape, input_type, log=False):
    converter = MxNetToONNXConverter()

    if isinstance(model, basestring) and isinstance(weights, basestring):
        print("Converting json and params file to sym and weights")
        sym, params = load_module(model, weights, input_shape)
        onnx_graph = converter.convert_mx2onnx_graph(sym, params, input_shape,
                                                     mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(input_type)], log=log)
    else:
        onnx_graph = converter.convert_mx2onnx_graph(model, weights, input_shape,
                                                 mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(input_type)], log=log)
    onnx_model = helper.make_model(onnx_graph)
    return onnx_model
