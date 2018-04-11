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
from .import_model import import_model, get_model_metadata
from .... import gluon

def import_to_gluon(model_file):
    sym, arg_params, aux_params = import_model(model_file)
    metadata = get_model_metadata(model_file)
    data_name = metadata['input_tensor_data'][0][0]
    net = gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var(data_name))
    net_params = net.collect_params()
    for param in arg_params:
        if param in net_params:
            net_params[param]._load_init(arg_params[param], ctx=ctx)
    for param in aux_params:
        if param in net_params:
            net_params[param]._load_init(aux_params[param], ctx=ctx)
    return net
