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


def slice_symbol_to_seq_symobls(net, seq_len, axis=1, squeeze_axis=True):
    net = mx.sym.SliceChannel(data=net, num_outputs=seq_len, axis=axis, squeeze_axis=squeeze_axis)
    hidden_all = []
    for seq_index in range(seq_len):
        hidden_all.append(net[seq_index])
    net = hidden_all
    return net
