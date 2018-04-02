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

import os
import numpy as np
import mxnet as mx

def test_ctx_group():
    def check_ctx_group(group2ctx, grad_req, mlp, set_stage1):
        texec = mlp.simple_bind(mx.cpu(0),
                                group2ctx=group2ctx,
                                data=(1,200), grad_req=grad_req)

        for arr, name in zip(texec.arg_arrays, mlp.list_arguments()):
            if name in set_stage1:
                assert arr.context == group2ctx['stage1']
            else:
                assert arr.context == group2ctx['stage2']

    with mx.AttrScope(ctx_group='stage1'):
        data = mx.symbol.Variable('data')
        fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
        act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")

    set_stage1 = set(act1.list_arguments())
    with mx.AttrScope(ctx_group='stage2'):
        fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
        act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
        fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
        fc3 = mx.symbol.BatchNorm(fc3)
        mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

    set_stage2 = set(mlp.list_arguments()) - set_stage1

    group2ctx = {
        'stage1' : mx.cpu(1),
        'stage2' : mx.cpu(2)
    }

    # generate reqs with null
    grad_req_with_null = {}
    for arg in mlp.list_arguments():
        grad_req_with_null[arg] = 'null' if arg == 'data' else 'write'

    grad_reqs = ['write', grad_req_with_null]
    for grad_req in grad_reqs:
        check_ctx_group(group2ctx, grad_req, mlp, set_stage1)

def test_ctx_group_sparse():
    with mx.AttrScope(ctx_group='stage1'):
        lhs = mx.symbol.Variable('lhs', stype='csr')
        rhs = mx.symbol.Variable('rhs', stype='row_sparse')
        dot  = mx.symbol.dot(lhs, rhs, name='dot')

    set_stage1 = set(dot.list_arguments())
    with mx.AttrScope(ctx_group='stage2'):
        softmax  = mx.symbol.SoftmaxOutput(data = dot, name = 'softmax')

    set_stage2 = set(softmax.list_arguments()) - set_stage1

    group2ctx = {
        'stage1' : mx.cpu(1),
        'stage2' : mx.cpu(2)
    }
    texec = softmax.simple_bind(mx.cpu(0), group2ctx=group2ctx,
                                lhs=(32,200), rhs=(200, 5))

    for arr, name in zip(texec.arg_arrays, softmax.list_arguments()):
        if name in set_stage1:
            assert arr.context == group2ctx['stage1']
        else:
            assert arr.context == group2ctx['stage2']

if __name__ == '__main__':
    test_ctx_group()
    test_ctx_group_sparse()
