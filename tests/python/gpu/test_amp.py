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

from common import assert_raises_cudnn_not_satisfied
import os
import sys
import mxnet as mx
import pytest
from mxnet import amp
from mxnet.test_utils import set_default_device
from mxnet.gluon import nn, rnn
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
sys.path.insert(0, os.path.join(curr_path, '../train'))
set_default_device(mx.gpu(0))


@pytest.fixture()
def amp_tests(request):
    def teardown():
        mx.nd.waitall()

    request.addfinalizer(teardown)


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
