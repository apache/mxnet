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
import numpy as np
from mxnet.test_utils import assert_almost_equal


def test_dropout_with_seed():
    assert mx.context.num_gpus() > 0
    a = mx.nd.ones((100, 100), ctx=mx.gpu())
    dropout = mx.gluon.nn.Dropout(0.5)

    info = np.iinfo(np.int32)
    seed = np.random.randint(info.min, info.max)
    mx.random.seed(seed)
    with mx.autograd.record():
        b = dropout(a)

    mx.random.seed(seed)
    with mx.autograd.record():
        c = dropout(a)
    # dropout on gpu should return same result with fixed seed
    assert_almost_equal(b.asnumpy(), c.asnumpy())

if __name__ == '__main__':
    import nose

    nose.runmodule()
