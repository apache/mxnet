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

import mxnet as mx
import numpy as np
from mxnet.test_utils import assert_almost_equal

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
def test_rnn_with_seed():
    info = np.iinfo(np.int32)
    seed = np.random.randint(info.min, info.max)
    _test_rnn(seed, mx.cpu())
    _test_rnn(seed, mx.gpu())

def _test_rnn(seed, ctx):
    data = mx.nd.ones((5, 3, 10), ctx=ctx)
    rnn = mx.gluon.rnn.RNN(100, 3, dropout=0.5)
    rnn.initialize(ctx=ctx)
    mx.random.seed(seed)
    with mx.autograd.record():
        result1 = rnn(data)

    mx.random.seed(seed)
    with mx.autograd.record():
        result2 = rnn(data)
    # dropout on gpu should return same result with fixed seed
    assert_almost_equal(result1.asnumpy(), result2.asnumpy())