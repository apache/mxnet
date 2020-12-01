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

import sys
import os
import mxnet as mx
import numpy as np
import unittest
import ctypes
import pytest
from mxnet.test_utils import assert_almost_equal

def test_float64_fallback():
    sym = mx.sym.FullyConnected(
        mx.sym.Variable('in'),
        mx.sym.Variable('w'),
        mx.sym.Variable('b'),
        num_hidden=2)

    dtype = 'float64'
    args = {'in': mx.nd.array([[2, 3, 4]], dtype=dtype),
        'w': mx.nd.array([[1, 2, 3], [4, 5, 6]], dtype=dtype),
        'b': mx.nd.array([7, 8], dtype=dtype)}
    ex = sym._bind(mx.cpu(), args, args_grad=None, grad_req='write')
    ex.forward()
    ex.outputs[0].wait_to_read()


@pytest.mark.parametrize('axis', [0, 1, 2, 3])
def test_bn_relu_fusion(axis):
    dummy_data = mx.nd.uniform(-1.0, 1.0, shape=(32, 3, 224, 224))

    net = mx.gluon.nn.HybridSequential()
    net.add(mx.gluon.nn.BatchNorm(axis=axis))
    net.add(mx.gluon.nn.Activation('relu'))
    net.initialize()

    out1 = net(dummy_data)
    out1.wait_to_read()
    net.optimize_for(dummy_data, backend='MKLDNN')
    out2 = net(dummy_data)

    assert_almost_equal(out1, out2)
