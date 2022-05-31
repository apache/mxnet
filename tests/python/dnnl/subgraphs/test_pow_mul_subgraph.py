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
import pytest
from subgraph_common import check_fusion
from subgraph_common import DATA_SHAPE
from mxnet.gluon import nn

@mx.util.use_np
@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('input_type', ['float32', 'int32', 'int8'])
@pytest.mark.parametrize('exponent', [2, 2.0])
@pytest.mark.parametrize('multiplier', [3, 3.0])
def test_pow_mul_fuse(data_shape, input_type, exponent, multiplier):
    class TestPowMulFuse(nn.HybridBlock):
        def __init__(self):
            super(TestPowMulFuse, self).__init__()

        def forward(self, input, *args):
            return (input**exponent)*multiplier

    net = TestPowMulFuse()
    attrs = {'sg_pow_mul_scalar' : []}
    check_fusion(net, data_shape, attrs,
                input_type=input_type,
                check_quantization=False)
