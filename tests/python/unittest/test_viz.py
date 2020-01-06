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

import unittest
import warnings

import mxnet as mx
import numpy as np


def test_print_summary():
    data = mx.sym.Variable('data')
    bias = mx.sym.Variable('fc1_bias', lr_mult=1.0)
    emb1= mx.symbol.Embedding(data = data, name='emb1', input_dim=100, output_dim=28)
    conv1= mx.symbol.Convolution(data = emb1, name='conv1', num_filter=32, kernel=(3,3), stride=(2,2))
    bn1 = mx.symbol.BatchNorm(data = conv1, name="bn1")
    act1 = mx.symbol.Activation(data = bn1, name='relu1', act_type="relu")
    mp1 = mx.symbol.Pooling(data = act1, name = 'mp1', kernel=(2,2), stride=(2,2), pool_type='max')
    fc1 = mx.sym.FullyConnected(data=mp1, bias=bias, name='fc1', num_hidden=10, lr_mult=0)
    fc2 = mx.sym.FullyConnected(data=fc1, name='fc2', num_hidden=10, wd_mult=0.5)
    sc1 = mx.symbol.SliceChannel(data=fc2, num_outputs=10, name="slice_1", squeeze_axis=0)
    mx.viz.print_summary(sc1)
    shape = {}
    shape["data"]=(1,3,28)
    mx.viz.print_summary(sc1, shape)

def graphviz_exists():
    try:
        import graphviz
    except ImportError:
        return False
    else:
        return True

@unittest.skipIf(not graphviz_exists(), "Skip test_plot_network as Graphviz could not be imported")
def test_plot_network():
    # Test warnings for cyclic graph
    net = mx.sym.Variable('data')
    net = mx.sym.FullyConnected(data=net, name='fc', num_hidden=128)
    net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
    net = mx.sym.FullyConnected(data=net, name='fc', num_hidden=10)
    net = mx.sym.SoftmaxOutput(data=net, name='out')
    with warnings.catch_warnings(record=True) as w:
        digraph = mx.viz.plot_network(net, shape={'data': (100, 200)},
                                      dtype={'data': np.float32},
                                      node_attrs={"fixedsize": "false"})
    assert len(w) == 1
    assert "There are multiple variables with the same name in your graph" in str(w[-1].message)
    assert "fc" in str(w[-1].message)

if __name__ == "__main__":
    import nose
    nose.runmodule()
