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

import time
import mxnet as mx
import numpy as _np
from mxnet import np, npx

def measure_cost(repeat, func_name, *args, **kwargs):
    """Measure time cost of running a function
    """
    mx.nd.waitall()
    start = time.time()
    for _ in range(repeat):
        func_name(*args, **kwargs)
    mx.nd.waitall()
    end = time.time()
    diff = end - start
    return diff / repeat


def test_tvm_dot():
    # benchmark
    for i in list(range(1000, 1100, 4)):
        m = i
        k = i
        n = i
        print("{} * {} X {} * {}".format(m, k, k, n))
        a = mx.nd.random.uniform(shape=(m, k), dtype='float32')
        b = mx.nd.random.uniform(shape=(k, n), dtype='float32')
        cost = measure_cost(2, mx.nd.contrib.tvm_dot, a, b)
        print("dispatch cost: {} ms".format(cost * 1000))
        a = mx.nd.random.uniform(shape=(m, k), dtype='float32')
        b = mx.nd.random.uniform(shape=(k, n), dtype='float32')
        cost = measure_cost(2, mx.nd.contrib.tvm_dot_fallback, a, b)
        print("fallback cost: {} ms".format(cost * 1000))
        a = mx.nd.random.uniform(shape=(m, k), dtype='float32')
        b = mx.nd.random.uniform(shape=(k, n), dtype='float32')
        cost = measure_cost(2, mx.nd.dot, a, b)
        print("dot cost: {} ms".format(cost * 1000))

if __name__ == "__main__":
    test_tvm_dot()
