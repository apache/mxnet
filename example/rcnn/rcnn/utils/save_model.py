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


def save_checkpoint(prefix, epoch, arg_params, aux_params):
    """Checkpoint the model data into file.
    :param prefix: Prefix of model name.
    :param epoch: The epoch number of the model.
    :param arg_params: dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    :param aux_params: dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    :return: None
    prefix-epoch.params will be saved for parameters.
    """
    save_dict = {('arg:%s' % k) : v for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v for k, v in aux_params.items()})
    param_name = '%s-%04d.params' % (prefix, epoch)
    mx.nd.save(param_name, save_dict)
