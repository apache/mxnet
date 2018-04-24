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

import json
import mxnet as mx
from mxnet.contrib import graph
from mxnet import sym


def test_print_graph_ir():
    x = sym.Variable("x", shape=(1, 1, 10, 20))
    y = sym.relu(x)
    g = graph.create(y)
    g._set_json_attr("shape_attr_key", "__shape__")
    g = g.apply("InferShape")
    ir1 = g.ir()
    ir2 = g.ir(join_entry_attrs=["shape"])
    assert("shape=" in ir2)


def test_infer_type():
    x = sym.Variable('x', dtype="float32")
    y = sym.elemwise_add(x, x, name='add1')
    y = sym.cast(y, dtype="float64", name="cast1")
    g = graph.create(y)
    g._set_json_attr("dtype_attr_key", "__dtype__")
    g = g.apply('InferType')
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('dtype')[jnode_row_ptr[nindex["cast1"]]] == 1
    assert g.json_attr('dtype')[jnode_row_ptr[nindex["add1"]]] == 0


def test_json_pass():
    x = sym.Variable('x')
    y = sym.FullyConnected(data=x, name='conv', num_hidden=30)
    g = graph.create(y)
    ret = g.apply('SaveJSON')
    ret._set_json_attr('json', ret.json_attr('json'))
    g2 = ret.apply('LoadJSON')
    assert g2.apply('SaveJSON').json_attr('json') == ret.json_attr('json')
    json = g.json()
    g2 = graph.load_json(json)
    assert json == g2.json()


def test_graph_json_attr():
    x = sym.Variable('x')
    y = sym.FullyConnected(data=x, name='conv', num_hidden=30)
    g = graph.create(y)
    g._set_json_attr('ilist', [1, 2, 3], 'list_int')
    assert g.json_attr('ilist') == [1, 2, 3]


def test_infer_shape():
    x = sym.Variable('x', shape=(2, 4, 2))
    y = sym.elemwise_add(x, x, name='add1')
    y = sym.flatten(y, name="flatten")
    g = graph.create(y)
    g._set_json_attr("shape_attr_key", "__shape__")
    g = g.apply('InferShape')
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('shape')[jnode_row_ptr[nindex["flatten"]]] == [2, 8]
    assert g.json_attr('shape')[jnode_row_ptr[nindex["add1"]]] == [2, 4, 2]


if __name__ == "__main__":
    test_infer_type()
    test_print_graph_ir()
    test_json_pass()
    test_graph_json_attr()
    test_infer_shape()
