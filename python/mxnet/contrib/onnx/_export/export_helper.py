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
"""export helper functions"""
# coding: utf-8
import os
import mxnet as mx
import numpy as np

def load_module(json_path, params_path, input_shape):
    """Loads the MXNet model file, retrieves symbol and parameters and returns.

    Parameters
    ----------
    json_path : str
        Path to the json file
    params_path : str
        Path to the params file
    input_shape :
        Input shape of the model e.g (1,3,224,224)

    Returns
    -------
    sym : MXNet symbol
        Model symbol object

    params : params object
        Model weights including both arg and aux params.
    """
    if not (os.path.isfile(json_path) and os.path.isfile(params_path)):
        raise ValueError("Provide valid path to the json and params file")
    else:
        try:
            model_name = json_path.rsplit('.', 1)[0].rsplit('-', 1)[0]
            num_epochs = int(params_path.rsplit('.', 1)[0].rsplit('-', 1)[1])
        except IndexError:
            print("Model and params name should be in format: prefix-symbol.json, prefix-epoch.params")
            raise

        sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, num_epochs)
        trained_model = mx.mod.Module(symbol=sym, label_names=None)
        trained_model.bind(data_shapes=[('data', input_shape[0])], label_shapes=trained_model._label_shapes,
                           for_training=False)

        # Merging arg and aux parameters
        params = {}
        params.update(arg_params)
        params.update(aux_params)

        return sym, params
