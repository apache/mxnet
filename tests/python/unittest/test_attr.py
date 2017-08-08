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
import mxnet as mx
from common import models
import pickle as pkl

def test_attr_basic():
    with mx.AttrScope(group='4', data='great'):
        data = mx.symbol.Variable('data',
                                  attr={'dtype':'data',
                                        'group': '1',
                                        'force_mirroring': 'True'},
                                  lr_mult=1)
        gdata = mx.symbol.Variable('data2')
    assert gdata.attr('group') == '4'
    assert data.attr('group') == '1'
    assert data.attr('lr_mult') == '1'
    assert data.attr('__lr_mult__') == '1'
    assert data.attr('force_mirroring') == 'True'
    assert data.attr('__force_mirroring__') == 'True'
    data2 = pkl.loads(pkl.dumps(data))
    assert data.attr('dtype') == data2.attr('dtype')

def test_operator():
    data = mx.symbol.Variable('data')
    with mx.AttrScope(__group__='4', __data__='great'):
        fc1 = mx.symbol.Activation(data, act_type='relu')
        with mx.AttrScope(__init_bias__='0.0'):
            fc2 = mx.symbol.FullyConnected(fc1, num_hidden=10, name='fc2')
    assert fc1.attr('__data__') == 'great'
    assert fc2.attr('__data__') == 'great'
    assert fc2.attr('__init_bias__') == '0.0'
    fc2copy = pkl.loads(pkl.dumps(fc2))
    assert fc2copy.tojson() == fc2.tojson()
    fc2weight = fc2.get_internals()['fc2_weight']

def contain(x, y):
    for k, v in x.items():
        if k not in y:
            return False
        if isinstance(y[k], dict):
            if not isinstance(v, dict):
                return False
            if not contain(v, y[k]):
                return False
        elif y[k] != v:
            return False
    return True

def test_list_attr():
    data = mx.sym.Variable('data', attr={'mood': 'angry'})
    op = mx.sym.Convolution(data=data, name='conv', kernel=(1, 1),
                            num_filter=1, attr={'__mood__': 'so so', 'wd_mult': 'x'})
    assert contain({'__mood__': 'so so', 'wd_mult': 'x', '__wd_mult__': 'x'}, op.list_attr())

def test_attr_dict():
    data = mx.sym.Variable('data', attr={'mood': 'angry'})
    op = mx.sym.Convolution(data=data, name='conv', kernel=(1, 1),
                            num_filter=1, attr={'__mood__': 'so so'}, lr_mult=1)
    assert contain({
        'data': {'mood': 'angry'},
        'conv_weight': {'__mood__': 'so so'},
        'conv': {'kernel': '(1, 1)', '__mood__': 'so so', 'num_filter': '1', 'lr_mult': '1', '__lr_mult__': '1'},
        'conv_bias': {'__mood__': 'so so'}}, op.attr_dict())

if __name__ == '__main__':
    test_attr_basic()
    test_operator()
    test_list_attr()
    test_attr_dict()
