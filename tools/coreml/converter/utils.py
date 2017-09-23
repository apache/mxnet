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
    """Returns a module loaded with the provided model.

    Parameters
    ----------
    model_name: str
        Prefix of the MXNet model name as stored on the local directory.

    epoch_num : int
        Epoch number of model we would like to load.

    input_shape: tuple
        The shape of the input data in the form of (batch_size, channels, height, width)

    files: list of strings
        List of URLs pertaining to files that need to be downloaded in order to use the model.

    data_shapes: list of tuples.
        List of tuples where each tuple is a pair of input variable name and its shape.

    label_shapes: list of (str, tuple)
        Typically is ``data_iter.provide_label``.

    label_names: list of str
        Name of the output labels in the MXNet symbolic graph.

    gpus: str
        Comma separated string of gpu ids on which inferences are executed. E.g. 3,5,6 would refer to GPUs 3, 5 and 6.
        If empty, we use CPU.

    Returns
    -------
    MXNet module
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, epoch_num)

    mod = create_module(sym, data_shapes, label_shapes, label_names, gpus)

    mod.set_params(
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True
    )

    return mod


def create_module(sym, data_shapes, label_shapes, label_names, gpus=''):
    """Creates a new MXNet module.

    Parameters
    ----------
    sym : Symbol
        An MXNet symbol.

    input_shape: tuple
        The shape of the input data in the form of (batch_size, channels, height, width)

    files: list of strings
        List of URLs pertaining to files that need to be downloaded in order to use the model.

    data_shapes: list of tuples.
        List of tuples where each tuple is a pair of input variable name and its shape.

    label_shapes: list of (str, tuple)
        Typically is ``data_iter.provide_label``.

    label_names: list of str
        Name of the output labels in the MXNet symbolic graph.

    gpus: str
        Comma separated string of gpu ids on which inferences are executed. E.g. 3,5,6 would refer to GPUs 3, 5 and 6.
        If empty, we use CPU.

    Returns
    -------
    MXNet module
    """
    if gpus == '':
        devices = mx.cpu()
    else:
        devices = [mx.gpu(int(i)) for i in gpus.split(',')]

    data_names = [data_shape[0] for data_shape in data_shapes]

    mod = mx.mod.Module(
        symbol=sym,
        data_names=data_names,
        context=devices,
        label_names=label_names
    )
    mod.bind(
        for_training=False,
        data_shapes=data_shapes,
        label_shapes=label_shapes
    )
    return mod

