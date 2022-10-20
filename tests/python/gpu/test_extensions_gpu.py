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

# This test checks if dynamic loading of library into MXNet is successful

import os
import platform
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from mxnet.base import MXNetError
from mxnet.test_utils import download, is_cd_run, assert_almost_equal, default_device
import pytest

base_path = os.path.join(os.path.dirname(__file__), "../../..")
def check_platform(supported_platforms=['x86_64', 'AMD64']):
    return platform.machine() not in supported_platforms

@pytest.mark.skipif(check_platform(), reason="not all machine types supported")
@pytest.mark.skipif(is_cd_run(), reason="continuous delivery run - ignoring test")
def test_custom_op_gpu():
    # possible places to find library file
    if (os.name=='posix'):
        lib = 'libcustomop_gpu_lib.so'
        if os.path.exists(lib):
            fname = lib
        elif os.path.exists(os.path.join(base_path, 'build/'+lib)):
            fname = os.path.join(base_path, 'build/'+lib)
        else:
            raise MXNetError(f"library {lib} not found ")
    elif (os.name=='nt'):
        lib = 'libcustomop_gpu_lib.dll'
        if os.path.exists('windows_package\\lib\\'+lib):
            fname = 'windows_package\\lib\\'+lib
        else:
            raise MXNetError(f"library {lib} not found ")

    fname = os.path.abspath(fname)
    # load the library containing gemm custom operators
    mx.library.load(fname)

    # test symbol custom relu operator in gpu
    b = mx.nd.array([[-2,-1],[1,2]], ctx=mx.gpu())
    c = mx.sym.Variable('c')
    d = mx.sym.Variable('d')
    e = mx.sym.my_relu(c)
    base = mx.sym.relu(d)
    in_grad = [mx.nd.empty((2,2), ctx=mx.gpu())]
    in_grad_base = [mx.nd.empty((2,2), ctx=mx.gpu())]
    exe = e._bind(ctx=mx.gpu(), args={'c':b}, args_grad=in_grad)
    exe_base = base._bind(ctx=mx.gpu(), args={'d':b}, args_grad=in_grad_base)
    out = exe.forward()
    out_base = exe_base.forward()
    assert_almost_equal(out_base[0].asnumpy(), out[0].asnumpy(), rtol=1e-3, atol=1e-3)

    # test custom relu backward
    out_grad = mx.nd.ones((2,2), ctx=mx.gpu())
    exe.backward([out_grad])
    exe_base.backward([out_grad])
    assert_almost_equal(in_grad_base[0].asnumpy(), in_grad[0].asnumpy(), rtol=1e-3, atol=1e-3)

    # test custom noisy relu producing deterministic result given same seed managed by mxnet
    d1 = mx.nd.ones(shape=(10,10,10), ctx=mx.cpu())
    d2 = mx.nd.ones(shape=(10,10,10), ctx=mx.gpu())

    mx.random.seed(128, ctx=mx.cpu())
    r1 = mx.nd.my_noisy_relu(d1)
    mx.random.seed(128, ctx=mx.cpu())
    r2 = mx.nd.my_noisy_relu(d1)
    assert_almost_equal(r1.asnumpy(), r2.asnumpy(), rtol=1e-3, atol=1e-3)

    mx.random.seed(128, ctx=mx.gpu())
    r3 = mx.nd.my_noisy_relu(d2)
    mx.random.seed(128, ctx=mx.gpu())
    r4 = mx.nd.my_noisy_relu(d2)
    assert_almost_equal(r3.asnumpy(), r4.asnumpy(), rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(check_platform(['x86_64']), reason="not all machine types supported")
@pytest.mark.skipif(is_cd_run(), reason="continuous delivery run - ignoring test")
def test_external_op():
    # check if operator already exists
    if hasattr(mx.nd, 'min_ex'):
        raise MXNetError('Operator already loaded')

    lib = 'libexternal_lib.so'
    fname = os.path.join(base_path,'example/extensions/lib_external_ops/build/'+lib)
    if not os.path.exists(fname):
        raise MXNetError(f"library {lib} not found ")

    fname = os.path.abspath(fname)
    mx.library.load(fname, False)

    # execute operator
    try:
        mx.nd.min_ex()
    except:
        raise MXNetError('Operator not loaded successfully')
