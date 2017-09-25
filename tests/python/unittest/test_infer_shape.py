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

# pylint: skip-file
import mxnet as mx
from common import models
from nose.tools import *

def test_mlp2_infer_shape():
    # Build MLP
    out = models.mlp2()
    # infer shape
    data_shape = (100, 100)
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(data=data_shape)
    arg_shape_dict = dict(zip(out.list_arguments(), arg_shapes))
    assert len(out_shapes) == 1
    assert out_shapes[0] == (100, 10)
    true_shapes = {'fc2_bias': (10,),
                   'fc2_weight' : (10, 1000),
                   'fc1_bias' : (1000,),
                   'fc1_weight' : (1000,100) }
    for k, v in true_shapes.items():
        assert arg_shape_dict[k] == v

@raises(mx.MXNetError)
def test_mlp2_infer_error():
    # Test shape inconsistent case
    out = models.mlp2()
    weight_shape= (1, 100)
    data_shape = (100, 100)
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(data=data_shape, fc1_weight=weight_shape)


def test_backward_infer():
    w = mx.sym.Variable("weight")
    wshift = mx.sym.Variable("wshift", shape=(1,))
    data = mx.sym.Variable("data")
    # broadcast add here, not being able to deduce shape correctly
    wt = mx.sym.broadcast_add(w, wshift)
    # shape constraint, this is what enables backward shape inference
    wt = mx.symbol._internal._identity_with_attr_like_rhs(wt, w)
    net = mx.sym.FullyConnected(data=data, weight=wt, num_hidden=11, no_bias=True)
    data_shape = (7, 100)
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=data_shape)
    arg_shape_dict = dict(zip(net.list_arguments(), arg_shapes))
    true_shapes = {'weight': (11, 100)}
    for k, v in true_shapes.items():
        assert arg_shape_dict[k] == v


def test_incomplete_infer_elewise():
    a = mx.sym.Variable('a', shape=(0, 10))
    b = mx.sym.Variable('b', shape=(12, 0))
    c = a + b
    arg_shapes, _, _ = c.infer_shape()
    arg_names = c.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (12, 10)
    assert arg_shapes['b'] == (12, 10)


def test_incomplete_infer_mlp():
    a = mx.sym.Variable('a', shape=(0, 10))
    b = mx.sym.FullyConnected(data=a, num_hidden=21)
    c = mx.sym.Variable('c', shape=(5, 0))
    d = b + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (5, 10)
    assert arg_shapes['c'] == (5, 21)


def test_incomplete_infer_slicechannel():
    a = mx.sym.Variable('a', shape=(0, 10))
    b = mx.sym.SliceChannel(data=a, num_outputs=10, axis=1, squeeze_axis=True)
    c = mx.sym.Variable('c', shape=(5,))
    d = b[1] + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (5, 10)

    a = mx.sym.Variable('a', shape=(0, 15, 0))
    b = mx.sym.SliceChannel(data=a, num_outputs=3, squeeze_axis=False)
    c = mx.sym.Variable('c', shape=(3, 5, 2))
    d = b[1] + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (3, 15, 2)


def test_incomplete_infer_convolution():
    a = mx.sym.Variable('a', shape=(0, 10, 0, 0))
    b = mx.sym.Convolution(data=a, num_filter=21, kernel=(3, 3), dilate=(1, 1), pad=(1, 1))
    c = mx.sym.Variable('c', shape=(5, 21, 32, 32))
    d = b + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (5, 10, 32, 32)


def test_incomplete_infer_concat():
    a = mx.sym.Variable('a', shape=(0, 10))
    b = mx.sym.Variable('b', shape=(0, 5))
    c = mx.sym.Concat(a, b, num_args=2, dim=1)
    d = mx.sym.Variable('d', shape=(2, 0))
    d = d + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (2, 10)
    assert arg_shapes['b'] == (2, 5)
    assert arg_shapes['d'] == (2, 15)

def test_fc_infer_type():
    mx_real_t = mx.base.mx_real_t
    data = mx.symbol.Variable('data')
    out = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=1000)

    # infer type
    data_type = mx_real_t
    arg_types, out_types, aux_types = out.infer_type(data=data_type)
    arg_type_dict = dict(zip(out.list_arguments(), arg_types))
    assert len(out_types) == 1
    assert out_types[0] == mx_real_t
    true_types = {
                   'fc1_bias' : mx_real_t,
                   'fc1_weight' : mx_real_t }
    for k, v in true_types.items():
        assert arg_type_dict[k] == v


if __name__ == "__main__":
    test_mlp2_infer_shape()
    test_mlp2_infer_error()
    test_backward_infer()
    test_incomplete_infer_elewise()
    test_incomplete_infer_mlp()
    test_incomplete_infer_slicechannel()
    test_incomplete_infer_convolution()
    test_incomplete_infer_concat()
