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


def conv(net,
         channels,
         filter_dimension,
         stride,
         weight=None,
         bias=None,
         act_type="relu",
         no_bias=False,
         name=None
         ):
    # 2d convolution's input should have the shape of 4D (batch_size,1,seq_len,feat_dim)
    if weight is None or bias is None:
        # ex) filter_dimension = (41,11) , stride=(2,2)
        net = mx.sym.Convolution(data=net, num_filter=channels, kernel=filter_dimension, stride=stride, no_bias=no_bias,
                                 name=name)
    elif weight is None or bias is not None:
        net = mx.sym.Convolution(data=net, num_filter=channels, kernel=filter_dimension, stride=stride, bias=bias,
                                 no_bias=no_bias, name=name)
    elif weight is not None or bias is None:
        net = mx.sym.Convolution(data=net, num_filter=channels, kernel=filter_dimension, stride=stride, weight=weight,
                                 no_bias=no_bias, name=name)
    else:
        net = mx.sym.Convolution(data=net, num_filter=channels, kernel=filter_dimension, stride=stride, weight=weight,
                                 bias=bias, no_bias=no_bias, name=name)
    net = mx.sym.Activation(data=net, act_type=act_type)
    return net
