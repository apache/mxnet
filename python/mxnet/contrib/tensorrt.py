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

""" Module to enable the use of TensorRT optimized graphs."""
import os

def set_use_fp16(status):
    """
    Set an environment variable which will enable or disable the use of FP16 precision in
    TensorRT
    Note: The mode FP16 force the whole TRT node to be executed in FP16
    :param status: Boolean, True if TensorRT should run in FP16, False for FP32
    """
    os.environ["MXNET_TENSORRT_USE_FP16"] = str(int(status))

def get_use_fp16():
    """
    Get an environment variable which describes if TensorRT is currently running in FP16
    :return: Boolean, true if TensorRT is running in FP16, False for FP32
    """
    return bool(int(os.environ.get("MXNET_TENSORRT_USE_FP16", 1)) == 1)

def init_tensorrt_params(sym, arg_params, aux_params):
    """
    Set weights in attributes of TensorRT nodes
    :param sym: Symbol, the symbol graph should contains some TensorRT nodes
    :param arg_params: arg_params
    :param aux_params: aux_params
    :return arg_params, aux_params: remaining params that are not in TensorRT nodes
    """
    for s in sym.get_internals():
        new_params_names = ""
        tensorrt_params = {}
        if 'subgraph_params_names' in s.list_attr():
            keys = s.list_attr()['subgraph_params_names'].split(';')
            for k in keys:
                if k in arg_params:
                    new_params_names += k + ";"
                    tensorrt_params['subgraph_param_' + k] = arg_params[k]
                    arg_params.pop(k)
                elif k in aux_params:
                    new_params_names += k + ";"
                    tensorrt_params['subgraph_param_' + k] = aux_params[k]
                    aux_params.pop(k)
            new_attrs = {}
            for k, v in tensorrt_params.items():
                new_attrs[k] = str(v.handle.value)
            if len(new_attrs) > 0:
                s._set_attr(**new_attrs)
                s._set_attr(subgraph_params_names=new_params_names[:-1])
    return arg_params, aux_params
