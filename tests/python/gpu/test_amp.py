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
from pathlib import Path
curr_path = Path(__file__).resolve().parent
sys.path.insert(0, str(curr_path.parent))
sys.path.insert(0, str(curr_path.parent/'unittest'))

import mxnet as mx
import pytest
from mxnet import amp
from mxnet.test_utils import set_default_device
from mxnet.gluon import nn, rnn

import amp.common as amp_common_tests
from common import assert_raises_cudnn_not_satisfied

AMP_DTYPE = 'float16'

set_default_device(mx.gpu(0))


def test_fp16_coverage():
    amp_common_tests.test_amp_coverage(AMP_DTYPE, 'FP16')


@mx.util.use_np
def test_fp16_basic_use():
    amp_common_tests.test_amp_basic_use(AMP_DTYPE)


@mx.util.use_np
def test_fp16_offline_casting():
    amp_common_tests.test_amp_offline_casting(AMP_DTYPE)


@mx.util.use_np
def test_fp16_offline_casting_shared_params():
    amp_common_tests.test_amp_offline_casting_shared_params(AMP_DTYPE)


@mx.util.use_np
def test_fp16_fp32_ops_order_independence():
    amp_common_tests.test_lp16_fp32_ops_order_independence(AMP_DTYPE)


@mx.util.use_np
def test_fp16_test_node_excluding():
    amp_common_tests.test_amp_node_excluding(AMP_DTYPE)


@pytest.mark.skip(reason='Error during waitall(). Tracked in #18099')
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_amp_conversion_rnn(amp_tests):
    with mx.Device(mx.gpu(0)):
        model = nn.HybridSequential()
        model.add(rnn.LSTM(hidden_size=10, num_layers=2, bidirectional=True))
        model.add(nn.Dense(2))
        model.initialize()
        model.hybridize()
        out = model(mx.nd.ones((2, 3, 4)))
        new_model = amp.convert_hybrid_block(model)
        out2 = new_model(mx.nd.ones((2, 3, 4)))
        mx.test_utils.assert_almost_equal(out.asnumpy(), out2.asnumpy(), atol=1e-2, rtol=1e-2)
