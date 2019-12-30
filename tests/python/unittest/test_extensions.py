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
import unittest
import mxnet as mx
import numpy as np
from mxnet.base import MXNetError
from mxnet.test_utils import download, is_cd_run, assert_almost_equal

def check_platform():
    return platform.machine() not in ['x86_64', 'AMD64']

@unittest.skipIf(check_platform(), "not all machine types supported")
@unittest.skipIf(is_cd_run(), "continuous delivery run - ignoring test")
def test_custom_op():
    if (os.name=='posix'):
        lib = 'libsample_lib.so'
        if os.path.exists(lib):
            fname = lib
        elif os.path.exists('build/'+lib):
            fname = 'build/'+lib
        else:
            raise MXNetError("library %s not found " % lib)
    elif (os.name=='nt'):
        lib = 'libsample_lib.dll'
        if os.path.exists('windows_package\\lib\\'+lib):
            fname = 'windows_package\\lib\\'+lib
        else:
            raise MXNetError("library %s not found " % lib)

    fname = os.path.abspath(fname)
    mx.library.load(fname)

    # test simple 2D gemm custom op loaded from sample library
    s = mx.sym.Variable('s')
    t = mx.sym.Variable('t')
    c = mx.sym.my_gemm(s,t)
    d = mx.sym.state_gemm(s,t)
    base = mx.sym.linalg.gemm2(s,t)  # baseline

    dim_n, dim_k, dim_m = tuple(np.random.randint(1, 5, size=3))

    mat1 = mx.nd.random.uniform(-10, 10, shape=(dim_n, dim_k), ctx=mx.cpu())
    mat2 = mx.nd.random.uniform(-10, 10, shape=(dim_k, dim_m), ctx=mx.cpu())

    in_grad1 = [mx.nd.empty((dim_n,dim_k),ctx=mx.cpu()),mx.nd.empty((dim_k,dim_m),ctx=mx.cpu())]
    in_grad2 = [mx.nd.empty((dim_n,dim_k),ctx=mx.cpu()),mx.nd.empty((dim_k,dim_m),ctx=mx.cpu())]
    in_grad_base = [mx.nd.empty((dim_n,dim_k),ctx=mx.cpu()),mx.nd.empty((dim_k,dim_m),ctx=mx.cpu())]

    exe1 = c.bind(ctx=mx.cpu(),args={'s':mat1,'t':mat2},args_grad=in_grad1)
    exe2 = d.bind(ctx=mx.cpu(),args={'s':mat1,'t':mat2},args_grad=in_grad2)
    exe_base = base.bind(ctx=mx.cpu(),args={'s':mat1,'t':mat2},args_grad=in_grad_base)

    out1 = exe1.forward()
    out2 = exe2.forward()
    out2 = exe2.forward()  # stateful
    out_base = exe_base.forward()

    assert_almost_equal(out_base[0].asnumpy(), out1[0].asnumpy(), rtol=1e-3, atol=1e-3)
    assert_almost_equal(out_base[0].asnumpy(), out2[0].asnumpy(), rtol=1e-3, atol=1e-3)

    out_grad = mx.nd.ones((dim_n, dim_m), ctx=mx.cpu())
    exe1.backward([out_grad])
    exe2.backward([out_grad])
    exe_base.backward([out_grad])

    assert_almost_equal(in_grad_base[0].asnumpy(), in_grad1[0].asnumpy(), rtol=1e-3, atol=1e-3)
    assert_almost_equal(in_grad_base[0].asnumpy(), in_grad2[0].asnumpy(), rtol=1e-3, atol=1e-3)
