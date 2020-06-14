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

from __future__ import print_function
import sys, os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../../../amalgamation/python/"))
from mxnet_predict import Predictor, load_ndarray_file

import ctypes
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.ndarray import NDArray
from mxnet import gluon
from mxnet.test_utils import assert_almost_equal
from mxnet.contrib.amp import amp
from mxnet.base import NDArrayHandle, py_str
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown_module

@with_seed()
def test_predictor_with_dtype():
    prefix = 'test_predictor_simple_dense'
    symbol_file = "%s-symbol.json" % prefix
    param_file = "%s-0000.params" % prefix

    input1 = np.random.uniform(size=(1, 3))
    input1 = input1.astype(np.float16)

    block = mx.gluon.nn.HybridSequential()
    block.add(mx.gluon.nn.Dense(7))
    block.add(mx.gluon.nn.Dense(3))
    block.cast(np.float16)
    block.hybridize()
    block.initialize(ctx=mx.gpu(0))
    tmp = mx.nd.array(input1, dtype=np.float16, ctx=mx.gpu(0))
    out1 = block.forward(tmp)
    block.export(prefix)

    predictor = Predictor(open(symbol_file, "r").read(),
                          open(param_file, "rb").read(),
                          {"data": input1.shape},
                          dev_type="gpu",
                          dev_id=0,
                          type_dict={"data": input1.dtype})
    predictor.forward(data=input1)
    predictor_out1 = predictor.get_output(0)

    assert_almost_equal(out1.asnumpy(), predictor_out1, rtol=1e-5, atol=1e-6)
