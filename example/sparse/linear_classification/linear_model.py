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
from weighted_softmax_ce import *

def linear_model(num_features, positive_cls_weight):
    # data with csr storage type to enable feeding data with CSRNDArray
    x = mx.symbol.Variable("data", stype='csr')
    norm_init = mx.initializer.Normal(sigma=0.01)
    # weight with row_sparse storage type to enable sparse gradient updates
    weight = mx.symbol.Variable("weight", shape=(num_features, 2),
                                init=norm_init, stype='row_sparse')
    bias = mx.symbol.Variable("bias", shape=(2,))
    dot = mx.symbol.sparse.dot(x, weight)
    pred = mx.symbol.broadcast_add(dot, bias)
    y = mx.symbol.Variable("softmax_label")
    model = mx.sym.Custom(pred, y, op_type='weighted_softmax_ce_loss',
                          positive_cls_weight=positive_cls_weight, name='out')
    return mx.sym.MakeLoss(model)
