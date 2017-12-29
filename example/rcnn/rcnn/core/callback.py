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


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        arg['bbox_pred_weight_test'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
        arg['bbox_pred_bias_test'] = arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop('bbox_pred_weight_test')
        arg.pop('bbox_pred_bias_test')
    return _callback
