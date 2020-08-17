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
from mxnet.test_utils import same
from common import with_seed
import random

@with_seed()
def test_contrib_intgemm_maxabsolute():
    if "intgemm_maxabsolute" not in dir(mx.nd.contrib):
        return
    shapes = [
        (3, 2),
        (9, 17),
        (2, 7, 1, 8),
    ]
    # Test all sizes relevant to register lengths too.
    for i in range(1, 65):
        shapes.append((i,))
    for shape in shapes:
        m = mx.nd.random_uniform(low=-100.0, high=100.0, shape=shape)
        fast = mx.nd.contrib.intgemm_maxabsolute(m)
        slow = mx.nd.max(mx.nd.abs(m))
        assert same(fast, slow)

@with_seed()
def test_contrib_intgemm_prepare_data():
    if "intgemm_prepare_data" not in dir(mx.nd.contrib):
        return
    # Try all weird overhang cases
    shapes = [(i,) for i in range(1, 67)] + [(2,3), (130, 12)]
    for shape in shapes:
        for max_quant in [2.0]:#, 1.0, 3.0]:
            m = mx.nd.random_uniform(low=-3.0, high=3.0, shape=shape)
            scaled = m * 127.0 / max_quant
            # Rounding 0.5 can go up or down.  Move values away from 0.5.
            too_close = mx.nd.abs(mx.nd.round(scaled) - scaled) > 0.45
            m += max_quant / 127.0 * 0.05 * too_close
            test = mx.nd.contrib.intgemm_prepare_data(m, mx.nd.array([max_quant]))
            # Reference: scale and round
            ref = mx.nd.round(m * 127.0 / max_quant)
            # Clip to [-127, 127].  Because otherwise e.g. -129 casts to +127.
            ref = mx.nd.broadcast_maximum(ref, mx.nd.array([-127.0]))
            ref = mx.nd.broadcast_minimum(ref, mx.nd.array([127.0]))
            # Reference: cast to int8
            ref = mx.nd.cast(ref, dtype='int8')
            # Reference: ban -128
            ref = mx.nd.broadcast_maximum(ref, mx.nd.array([-127], dtype = 'int8'))
            assert same(test, ref)
  
@with_seed()
def test_contrib_intgemm_weight_consistent():
    # The weight format is actually CPU-dependent so we don't directly test the
    # output, but indirectly that it works.
    if "intgemm_prepare_weight" not in dir(mx.nd.contrib):
        return
    max_quant = mx.nd.array([2.0])
    for shape in [(8, 64), (16, 64), (8, 128), (16, 128), (2, 4, 64)]:
        m = mx.nd.random_uniform(low=-3.0, high=3.0, shape=shape)
        direct = mx.nd.contrib.intgemm_prepare_weight(m, max_quant)
        quant = mx.nd.contrib.intgemm_prepare_data(m, max_quant) 
        indirect = mx.nd.contrib.intgemm_prepare_weight(quant, already_quantized=True)
        #Should get the same data from direct call and already_quantized version.
        assert same(direct, indirect)

@with_seed()
def test_contrib_intgemm_take_weight():
    if "intgemm_take_weight" not in dir(mx.nd.contrib):
        return
    indices_to_try = [
        [0,1,2,3,4,5,6,7],
        [1,2,1,2,1,2,1,2],
        [7,6,5,4,3,2,1,0],
        [3,1,4,1,5,9,2,6],
        [random.randint(0,15) for i in range(8)],
        [random.randint(0,15) for i in range(16)],
        [random.randint(0,15) for i in range(24)]
    ]
    # Since random_uniform doesn't support int8, use python
    m = mx.nd.array([random.randint(-127,127) for i in range(16 * 64)], dtype='int8')
    m = m.reshape((16, 64))
    for indices in indices_to_try:
        indices = mx.nd.array(indices, dtype='int32')
        # Prepare weight then take.
        test = mx.nd.contrib.intgemm_prepare_weight(m, already_quantized=True)
        test = mx.nd.contrib.intgemm_take_weight(test, indices)
        # Take then prepare.
        ref = m.take(indices, axis=0)
        ref = mx.nd.contrib.intgemm_prepare_weight(ref, already_quantized=True)
        assert same(test, ref)

# Test a particular shape of matrix multiplication.
def single_multiply_shape(data_rows, inner, weight_cols):
    # Don't use full range (-127, 127) to avoid saturation.
    data = [random.randint(-64, 64) for i in range(data_rows * inner)]
    data = mx.nd.array(data, dtype='int8').reshape((data_rows, inner))
    weight = [random.randint(-64, 64) for i in range(inner * weight_cols)]
    weight = mx.nd.array(weight, dtype='int8').reshape((weight_cols, inner))
    weight_prepared = mx.nd.contrib.intgemm_prepare_weight(weight, already_quantized=True)

    # int32 output, no bias
    test = mx.nd.contrib.intgemm_fully_connected(data,
                                                 weight_prepared,
                                                 no_bias=True,
                                                 flatten=False,
                                                 out_type='int32',
                                                 num_hidden=weight_cols)
    ref = mx.nd.FullyConnected(mx.nd.cast(data, dtype='float32'),
                               mx.nd.cast(weight, dtype='float32'),
                               no_bias=True,
                               flatten=False,
                               num_hidden=weight_cols)
    assert (mx.nd.cast(test, dtype='float32') - ref).norm().asscalar() < 0.01

    # float32 output, no bias
    scale = 3.0
    test = mx.nd.contrib.intgemm_fully_connected(data,
                                                 weight_prepared,
                                                 mx.nd.array(scale),
                                                 no_bias=True,
                                                 flatten=False,
                                                 out_type='float32',
                                                 num_hidden=weight_cols)
    assert (test - ref * scale).norm().asscalar() < 0.01

    # int32 output, bias
    bias = mx.nd.array([random.randint(-60000, 60000) for i in range(weight_cols)], dtype = 'int32')
    test = mx.nd.contrib.intgemm_fully_connected(data,
                                                 weight_prepared,
                                                 bias,
                                                 no_bias=False,
                                                 flatten=False,
                                                 out_type='int32',
                                                 num_hidden=weight_cols)
    ref = mx.nd.FullyConnected(mx.nd.cast(data, dtype='float32'),
                               mx.nd.cast(weight, dtype='float32'),
                               mx.nd.cast(bias, dtype='float32'),
                               no_bias=False,
                               flatten=False,
                               num_hidden=weight_cols)
    assert (mx.nd.cast(test, dtype='float32') - ref).norm().asscalar() < 0.01

    # float32 output, bias
    # Scaling is applied before bias (and bias is not scaled). So to make the
    # reference comparison easy, just scale the bias beforehand.
    test = mx.nd.contrib.intgemm_fully_connected(data,
                                                 weight_prepared,
                                                 mx.nd.array(scale),
                                                 mx.nd.cast(bias, dtype='float32') * scale,
                                                 no_bias=False,
                                                 flatten=False,
                                                 out_type='float32',
                                                 num_hidden=weight_cols)
    assert (test - ref * scale).norm().asscalar() < 0.01

    # float32 input should work the same as manually prepared int8 input.
    data_float = mx.nd.array([random.uniform(-3.14, 3.14) for i in range(data_rows * inner)])
    data_float = data_float.reshape(data_rows, inner)
    direct = mx.nd.contrib.intgemm_fully_connected(data_float,
                                                   weight_prepared,
                                                   mx.nd.array(scale),
                                                   mx.nd.cast(bias, dtype='float32'),
                                                   no_bias=False,
                                                   flatten=False,
                                                   out_type='float32',
                                                   num_hidden=weight_cols)
    maxabs = mx.nd.contrib.intgemm_maxabsolute(data_float)
    data_prepared = mx.nd.contrib.intgemm_prepare_data(data_float, maxabs)
    cooked = mx.nd.contrib.intgemm_fully_connected(data_prepared,
                                                   weight_prepared,
                                                   mx.nd.array(scale * maxabs / 127.0),
                                                   mx.nd.cast(bias, dtype='float32'),
                                                   no_bias=False,
                                                   flatten=False,
                                                   out_type='float32',
                                                   num_hidden=weight_cols)
    assert (direct - cooked).norm().asscalar() < 0.01


def test_contrib_intgemm_multiply():
    if "intgemm_fully_connected" not in dir(mx.nd.contrib):
        return
    #The multiplication routine has approximations so everything is tested
    #deterministically to ensure bounds are met.
    random.seed(1)
    for data_rows in range(1, 5):
        for inner in range(64, 256, 64):
            for weight_cols in range(8, 24, 8):
                single_multiply_shape(data_rows, inner, weight_cols)

