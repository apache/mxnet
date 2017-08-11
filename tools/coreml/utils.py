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

import mxnet as mx


def load_model(model_name, epoch_num, data_shapes, label_shapes, label_names, gpus=''):
    """

    :param model_name:
    :param epoch_num:
    :param data_shapes:
    :param label_shapes:
    :param label_names:
    :param gpus:
    :return:
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, epoch_num)
    if gpus == '':
        devices = mx.cpu()
    else:
        devices = [mx.gpu(int(i)) for i in gpus.split(',')]
    mod = mx.mod.Module(
        symbol=sym,
        context=devices,
        label_names=label_names
    )
    mod.bind(
        for_training=False,
        data_shapes=data_shapes,
        label_shapes=label_shapes
    )
    mod.set_params(
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True
    )
    return mod


