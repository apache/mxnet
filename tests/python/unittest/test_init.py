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

def test_default_init():
    data = mx.sym.Variable('data')
    sym = mx.sym.LeakyReLU(data=data, act_type='prelu')
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (10,10))])
    mod.init_params()
    assert (list(mod.get_params()[0].values())[0].asnumpy() == 0.25).all()

def test_variable_init():
    data = mx.sym.Variable('data')
    gamma = mx.sym.Variable('gamma', init=mx.init.One())
    sym = mx.sym.LeakyReLU(data=data, gamma=gamma, act_type='prelu')
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (10,10))])
    mod.init_params()
    assert (list(mod.get_params()[0].values())[0].asnumpy() == 1).all()

def test_aux_init():
    data = mx.sym.Variable('data')
    sym = mx.sym.BatchNorm(data=data, name='bn')
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (10, 10, 3, 3))])
    mod.init_params()
    assert (mod.get_params()[1]['bn_moving_var'].asnumpy() == 1).all()
    assert (mod.get_params()[1]['bn_moving_mean'].asnumpy() == 0).all()

def test_rsp_const_init():
    def check_rsp_const_init(init, val):
        shape = (10, 10)
        x = mx.symbol.Variable("data", stype='csr')
        weight = mx.symbol.Variable("weight", shape=(shape[1], 2),
                                    init=init, stype='row_sparse')
        dot = mx.symbol.sparse.dot(x, weight)
        mod = mx.mod.Module(dot, label_names=None)
        mod.bind(data_shapes=[('data', shape)])
        mod.init_params()
        assert (list(mod.get_params()[0].values())[0].asnumpy() == val).all()

    check_rsp_const_init(mx.initializer.Constant(value=2.), 2.)
    check_rsp_const_init(mx.initializer.Zero(), 0.)
    check_rsp_const_init(mx.initializer.One(), 1.)

if __name__ == '__main__':
    test_variable_init()
    test_default_init()
    test_aux_init()
    test_rsp_const_init()
