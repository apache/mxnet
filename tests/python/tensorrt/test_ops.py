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

from mxnet.test_utils import assert_almost_equal
import mxnet as mx
import numpy as np
import os

def check_elementwise_random(op='sum', shape=(1, 3, 224, 224)):
    """
    Check elementwise operators with vanilla/TensorRT executors with uniform random tensors
    """
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    if op == 'sum':
        sym = a + b
    elif op == 'sub':
        sym = a - b
    elif op == 'mul':
        sym = a * b

    a_data = mx.ndarray.random.uniform(shape=shape, ctx=mx.gpu())
    b_data = mx.ndarray.random.uniform(shape=shape, ctx=mx.gpu())

    executor = sym.simple_bind(ctx=mx.gpu(), a=shape, b=shape,
                               grad_req='null', force_rebind=True)
    y = executor.forward(is_train=False, a=a_data, b=b_data)
    trt_sym = sym.get_backend_symbol('TensorRT')
    original_precision_value = mx.contrib.tensorrt.get_use_fp16()
    try:
        mx.contrib.tensorrt.set_use_fp16(True)
        executor = trt_sym.simple_bind(ctx=mx.gpu(), a=shape, b=shape,
                                       grad_req='null', force_rebind=True)
        y_trt = executor.forward(is_train=False, a=a_data, b=b_data)
        mx.contrib.tensorrt.set_use_fp16(False)
        executor = trt_sym.simple_bind(ctx=mx.gpu(), a=shape, b=shape,
                                       grad_req='null', force_rebind=True)
        y_trt_fp32 = executor.forward(is_train=False, a=a_data, b=b_data)
        assert_almost_equal(y[0].asnumpy(), y_trt[0].asnumpy(), 1e-1, 1e-2)
        assert_almost_equal(y[0].asnumpy(), y_trt_fp32[0].asnumpy(), 1e-4, 1e-4)
    finally:
        mx.contrib.tensorrt.set_use_fp16(original_precision_value)


def test_elementwise():
    for op in ['sum', 'sub', 'mul']:
        for shape in [(20, 25), (3, 4, 20), (1, 3, 20, 25), (10, 10, 100, 100)]:
            for itry in range(10):
                check_elementwise_random(op, shape)


if __name__ == '__main__':
    import nose
    nose.runmodule()
