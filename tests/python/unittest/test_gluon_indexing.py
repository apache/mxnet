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

import pytest
import numpy as np
import mxnet as mx
from mxnet.gluon import HybridBlock

def test_getitem_hybridized():
    class picking_np(HybridBlock):
        def __init__(self, **kwargs):
            super(picking_np, self).__init__(**kwargs)
        def hybrid_forward(self, F, sequence, pick_ids):
            """
            new implementation in deep numpy
            """
            idx_arange = F.npx.arange_like(pick_ids.reshape((-1, )), axis=0)
            batch_idx = F.np.floor(idx_arange / 2).astype(np.int32)

            encoded = sequence[batch_idx, pick_ids.reshape((-1,))]
            encoded = F.npx.reshape_like(encoded, pick_ids, lhs_begin=-2, lhs_end=-1, rhs_begin=0)
            return encoded

    sequence = mx.nd.array(np.random.normal(0, 1, (8, 32, 768)), dtype=np.float32)
    # pick_ids: [batch_size, picked_index]
    pick_ids = mx.nd.random.randint(0, 32, (8,2), dtype=np.int32)

    mx.npx.set_np()
    picker_np = picking_np()
    seq_np = sequence.as_np_ndarray()
    np_output = picker_np(seq_np, pick_ids.as_np_ndarray())
    seq_np.attach_grad()
    with mx.autograd.record():
        z = picker_np(seq_np, pick_ids.as_np_ndarray())
    z.backward()

    picker_np.initialize()
    picker_np.hybridize()
    nd_output_hybridized = picker_np(sequence.as_np_ndarray(), pick_ids.as_np_ndarray())
    seq_np_hybridized = sequence.as_np_ndarray()
    seq_np_hybridized.attach_grad()
    with mx.autograd.record():
        z_hybridized = picker_np(seq_np_hybridized, pick_ids.as_np_ndarray())
    z_hybridized.backward()
    mx.npx.reset_np()

    mx.test_utils.assert_almost_equal(nd_output_hybridized.asnumpy(), np_output.asnumpy())
    mx.test_utils.assert_almost_equal(seq_np.grad.asnumpy(), seq_np_hybridized.grad.asnumpy())

def test_getitem_hybridized_no_F_argument():
    class picking_np(HybridBlock):
        def __init__(self, **kwargs):
            super(picking_np, self).__init__(**kwargs)
        def forward(self, sequence, pick_ids):
            """
            new implementation in deep numpy
            """
            idx_arange = mx.npx.arange_like(pick_ids.reshape((-1, )), axis=0)
            batch_idx = np.floor(idx_arange / 2).astype(np.int32)

            encoded = sequence[batch_idx, pick_ids.reshape((-1,))]
            encoded = mx.npx.reshape_like(encoded, pick_ids, lhs_begin=-2, lhs_end=-1, rhs_begin=0)
            return encoded

    sequence = mx.nd.array(np.random.normal(0, 1, (8, 32, 768)), dtype=np.float32)
    # pick_ids: [batch_size, picked_index]
    pick_ids = mx.nd.random.randint(0, 32, (8,2), dtype=np.int32)

    mx.npx.set_np()
    picker_np = picking_np()
    seq_np = sequence.as_np_ndarray()
    np_output = picker_np(seq_np, pick_ids.as_np_ndarray())
    seq_np.attach_grad()
    with mx.autograd.record():
        z = picker_np(seq_np, pick_ids.as_np_ndarray())
    z.backward()

    picker_np.initialize()
    picker_np.hybridize()
    nd_output_hybridized = picker_np(sequence.as_np_ndarray(), pick_ids.as_np_ndarray())
    seq_np_hybridized = sequence.as_np_ndarray()
    seq_np_hybridized.attach_grad()
    with mx.autograd.record():
        z_hybridized = picker_np(seq_np_hybridized, pick_ids.as_np_ndarray())
    z_hybridized.backward()
    mx.npx.reset_np()

    mx.test_utils.assert_almost_equal(nd_output_hybridized.asnumpy(), np_output.asnumpy())
    mx.test_utils.assert_almost_equal(z_hybridized.asnumpy(), np_output.asnumpy())
    mx.test_utils.assert_almost_equal(seq_np.grad.asnumpy(), seq_np_hybridized.grad.asnumpy())
