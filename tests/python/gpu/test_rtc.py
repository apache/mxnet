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

# pylint: skip-file
import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose

if __name__ == '__main__':
    x = mx.nd.zeros((10,), ctx=mx.gpu(0))
    x[:] = 1
    y = mx.nd.zeros((10,), ctx=mx.gpu(0))
    y[:] = 2
    rtc = mx.rtc('abc', [('x', x)], [('y', y)], """
        __shared__ float s_rec[10];
        s_rec[threadIdx.x] = x[threadIdx.x];
        y[threadIdx.x] = expf(s_rec[threadIdx.x]*5.0);""")
    rtc.push([x], [y], (1, 1, 1), (10,1,1))
    assert_allclose(y.asnumpy(), np.exp(x.asnumpy()*5.0))
