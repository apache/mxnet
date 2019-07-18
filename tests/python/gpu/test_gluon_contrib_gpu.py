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
"""Tests of the contrib APIs in Gluon only with gpu"""

from __future__ import print_function
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon import contrib
from mxnet.gluon.contrib.cnn import DeformableConvolution


def test_DeformableConvolution():
    """test of the deformable convolution layer with possible combinations of arguments,
    currently this layer only supports gpu
    """
    net = nn.HybridSequential()
    net.add(
        DeformableConvolution(10, kernel_size=(3, 3), strides=1, padding=0),
        DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                               offset_use_bias=False, use_bias=False),
        DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                               offset_use_bias=False),
        DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, activation='relu',
                               use_bias=False),
        DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, offset_use_bias=False, use_bias=False),
        DeformableConvolution(10, kernel_size=(3, 2), strides=1, padding=0, offset_use_bias=False),
        DeformableConvolution(12, kernel_size=(3, 2), strides=1, padding=0, use_bias=False),
        DeformableConvolution(12, kernel_size=(3, 2), strides=1, padding=0, use_bias=False, num_deformable_group=4),
    )

    try:
        ctx = mx.gpu()
        _ = mx.nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        print("deformable_convolution only supports GPU")
        return

    net.initialize(force_reinit=True, ctx=ctx)
    net.hybridize()

    x = mx.nd.random.uniform(shape=(8, 5, 30, 31), ctx=ctx)
    with mx.autograd.record():
        y = net(x)
        y.backward()


if __name__ == '__main__':
    import nose
    nose.runmodule()
