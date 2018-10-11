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

"""Test TVM bridge, only enable this when TVM is available"""
import logging
import mxnet as mx
import numpy as np
import unittest

def test_tvm_bridge():
    # only enable test if TVM is available
    try:
        import tvm
        import tvm.contrib.mxnet
        import topi
    except ImportError:
        logging.warn("TVM bridge test skipped because TVM is missing...")
        return

    def check(target, dtype):
        shape = (20,)
        scale = tvm.var("scale", dtype="float32")
        x = tvm.placeholder(shape, dtype=dtype)
        y = tvm.placeholder(shape, dtype=dtype)
        z = tvm.compute(shape, lambda i: x[i] + y[i])
        zz = tvm.compute(shape, lambda *i: z(*i) * scale.astype(dtype))
        ctx = mx.gpu(0) if target == "cuda" else mx.cpu(0)
        target = tvm.target.create(target)

        # build the function
        with target:
            s = topi.generic.schedule_injective(zz)
            f = tvm.build(s, [x, y, zz, scale])

        # get a mxnet version
        mxf = tvm.contrib.mxnet.to_mxnet_func(f, const_loc=[0, 1])
        xx = mx.nd.uniform(shape=shape, ctx=ctx).astype(dtype)
        yy = mx.nd.uniform(shape=shape, ctx=ctx).astype(dtype)
        zz = mx.nd.empty(shape=shape, ctx=ctx).astype(dtype)
        # invoke myf: this runs in mxnet engine
        mxf(xx, yy, zz, 10.0)
        np.testing.assert_allclose(
            zz.asnumpy(), (xx.asnumpy() + yy.asnumpy()) * 10)

    for tgt in ["llvm", "cuda"]:
        for dtype in ["int8", "uint8", "int64",
                      "float32", "float64"]:
            check(tgt, dtype)


if __name__ == "__main__":
    import nose
    nose.runmodule()
