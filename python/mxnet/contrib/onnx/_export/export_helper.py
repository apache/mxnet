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
import logging
import mxnet as mx

def load_module(sym_filepath, params_filepath, input_shape):
    """Loads the MXNet model file, retrieves symbol and parameters and
    returns MXNet symbol and params (weights).

    Parameters
    ----------
    json_path : str
        Path to the json file
    params_path : str
        Path to the params file
    input_shape : List of tuple
        Input shape of the model e.g [(1,3,224,224)]

    Returns
    -------
    sym : MXNet symbol
        Model symbol object

    params : params object
        Model weights including both arg and aux params.
    """
    if not (os.path.isfile(sym_filepath) and os.path.isfile(params_filepath)):
        raise ValueError("symbol and params files provided are invalid")
    else:
        try:
            # reads symbol.json file from given path and
            # retrieves model prefix and number of epochs
            model_name = sym_filepath.rsplit('.', 1)[0].rsplit('-', 1)[0]
            num_epochs = int(params_filepath.rsplit('.', 1)[0].rsplit('-', 1)[1])
        except IndexError:
            logging.info("Model and params name should be in format: "
                         "prefix-symbol.json, prefix-epoch.params")
            raise

        sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, num_epochs)

        # To fetch the data names of the input to the model we list the inputs of the symbol graph
        # and exclude the argument and auxiliary parameters from the list
        data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg_params and graph_input not in aux_params]

        # Creating data_shapes list having data name and data shape tuples
        data_shapes = []
        for idx, input_name in enumerate(data_names):
            data_shapes.append((input_name, input_shape[idx]))

        trained_model = mx.mod.Module(symbol=sym, data_names=data_names, label_names=None)
        trained_model.bind(data_shapes=data_shapes, label_shapes=None, for_training=False)

        # Merging arg and aux parameters
        params = {}
        params.update(arg_params)
        params.update(aux_params)

        return sym, params
