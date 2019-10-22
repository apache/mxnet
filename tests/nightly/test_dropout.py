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
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'unittest'))
from common import with_seed
import unittest

import mxnet as mx
import numpy as np
from mxnet.test_utils import assert_almost_equal
from nose.tools import assert_raises

@with_seed()
def test_dropout_with_seed():
    info = np.iinfo(np.int32)
    seed = np.random.randint(info.min, info.max)
    _test_dropout(seed, mx.cpu())
    _test_dropout(seed, mx.gpu())

def _test_dropout(seed, ctx):
    data = mx.nd.ones((100, 100), ctx=ctx)
    dropout = mx.gluon.nn.Dropout(0.5)

    mx.random.seed(seed)
    with mx.autograd.record():
        result1 = dropout(data)

    mx.random.seed(seed)
    with mx.autograd.record():
        result2 = dropout(data)
    # dropout on gpu should return same result with fixed seed
    assert_almost_equal(result1.asnumpy(), result2.asnumpy())

@with_seed()
@unittest.skipIf(mx.context.num_gpus() < 2,
                 "test_dropout_with_seed_multi_gpu needs more than 1 GPU")
def test_dropout_with_seed_multi_gpu():
    assert mx.context.num_gpus() > 1
    data1 = mx.nd.ones((100, 100), ctx=mx.gpu(0))
    data2 = mx.nd.ones((100, 100), ctx=mx.gpu(1))

    dropout = mx.gluon.nn.Dropout(0.5)

    info = np.iinfo(np.int32)
    seed = np.random.randint(info.min, info.max)

    mx.random.seed(seed, ctx=mx.gpu(0))
    with mx.autograd.record():
        result1 = dropout(data1)

    mx.random.seed(seed, ctx=mx.gpu(0))
    with mx.autograd.record():
        result2 = dropout(data1)

    mx.random.seed(seed, ctx=mx.gpu(0))
    with mx.autograd.record():
        result3 = dropout(data2)

    assert_almost_equal(result1.asnumpy(), result2.asnumpy())
    # dropout on gpu1 should return different result
    # with fixed seed only on gpu0
    with assert_raises(AssertionError):
        assert_almost_equal(result2.asnumpy(), result3.asnumpy())