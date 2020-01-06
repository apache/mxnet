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

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.test_utils import assert_almost_equal
from common import setup_module, with_seed, teardown

@with_seed()
def test_predictor():
    prefix = 'test_predictor_simple_dense'
    symbol_file = "%s-symbol.json" % prefix
    param_file = "%s-0000.params" % prefix

    # two inputs with different batch sizes
    input1 = np.random.uniform(size=(1,3))
    input2 = np.random.uniform(size=(3,3))

    # define a simple model
    block = gluon.nn.HybridSequential()
    block.add(gluon.nn.Dense(7))
    block.add(gluon.nn.Dense(3))
    block.hybridize()
    block.initialize()
    out1 = block.forward(nd.array(input1))
    out2 = block.forward(nd.array(input2))
    block.export(prefix)

    # create a predictor
    predictor = Predictor(open(symbol_file, "r").read(),
                      open(param_file, "rb").read(),
                      {'data':input1.shape})

    # forward and get output
    predictor.forward(data=input1)
    predictor_out1 = predictor.get_output(0)
    assert_almost_equal(out1.asnumpy(), predictor_out1, rtol=1e-5, atol=1e-6)

    # reshape
    predictor.reshape({'data':input2.shape})
    predictor.forward(data=input2)
    predictor_out2 = predictor.get_output(0)
    assert_almost_equal(out2.asnumpy(), predictor_out2, rtol=1e-5, atol=1e-6)

    # destroy the predictor
    del predictor

@with_seed()
def test_load_ndarray():
    nd_file = 'test_predictor_load_ndarray.params'
    a = nd.random.uniform(shape=(7, 3))
    b = nd.random.uniform(shape=(7,))
    nd_data = {'a':a, 'b':b}
    nd.save(nd_file, nd_data)

    # test load_ndarray_file
    nd_load = load_ndarray_file(open(nd_file, "rb").read())
    assert(set(nd_data.keys()) == set(nd_load.keys()))
    for k in nd_data.keys():
        assert_almost_equal(nd_data[k].asnumpy(), nd_load[k], rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    import nose
    nose.runmodule()
