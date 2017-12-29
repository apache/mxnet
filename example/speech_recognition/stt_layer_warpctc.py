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


def warpctc_layer(net, label, num_label, seq_len, character_classes_count):
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    fc_seq = []
    for seqidx in range(seq_len):
        hidden = net[seqidx]
        hidden = mx.sym.FullyConnected(data=hidden,
                                       num_hidden=character_classes_count,
                                       weight=cls_weight,
                                       bias=cls_bias)
        fc_seq.append(hidden)
    net = mx.sym.Concat(*fc_seq, dim=0)

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')

    net = mx.sym.WarpCTC(data=net, label=label, label_length=num_label, input_length=seq_len)

    return net
