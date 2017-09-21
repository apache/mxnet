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
import mxnet as mx
from mxnet.gluon import contrib
import numpy as np
from numpy.testing import assert_allclose


def check_rnn_cell(cell, prefix, in_shape=(10, 50), out_shape=(10, 100), begin_state=None):
    inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
    outputs, _ = cell.unroll(3, inputs, begin_state=begin_state)
    outputs = mx.sym.Group(outputs)
    assert sorted(cell.collect_params().keys()) == [prefix+'h2h_bias', prefix+'h2h_weight',
                                                    prefix+'i2h_bias', prefix+'i2h_weight']
    assert outputs.list_outputs() == [prefix+'t0_out_output', prefix+'t1_out_output', prefix+'t2_out_output']

    args, outs, auxs = outputs.infer_shape(rnn_t0_data=in_shape,
                                           rnn_t1_data=in_shape,
                                           rnn_t2_data=in_shape)
    assert outs == [out_shape]*3


def check_rnn_forward(layer, inputs):
    inputs.attach_grad()
    layer.collect_params().initialize()
    with mx.autograd.record():
        layer.unroll(3, inputs, merge_outputs=True)[0].backward()
        mx.autograd.backward(layer.unroll(3, inputs, merge_outputs=False)[0])
    mx.nd.waitall()


def test_rnn_cells():
    check_rnn_forward(contrib.rnn.Conv1DLSTMCell((5, 7), 10, (3,), (3,)),
                      mx.nd.ones((8, 3, 5, 7)))
    check_rnn_forward(contrib.rnn.Conv1DRNNCell((5, 7), 10, (3,), (3,)),
                      mx.nd.ones((8, 3, 5, 7)))
    check_rnn_forward(contrib.rnn.Conv1DGRUCell((5, 7), 10, (3,), (3,)),
                      mx.nd.ones((8, 3, 5, 7)))

    net = mx.gluon.rnn.SequentialRNNCell()
    net.add(contrib.rnn.Conv1DLSTMCell((5, 7), 10, (3,), (3,)))
    net.add(contrib.rnn.Conv1DRNNCell((10, 5), 11, (3,), (3,)))
    net.add(contrib.rnn.Conv1DGRUCell((11, 3), 12, (3,), (3,)))
    check_rnn_forward(net, mx.nd.ones((8, 3, 5, 7)))


def test_convrnn():
    cell = contrib.rnn.Conv1DRNNCell((10, 50), 100, 3, 3, prefix='rnn_')
    check_rnn_cell(cell, prefix='rnn_', in_shape=(1, 10, 50), out_shape=(1, 100, 48))

    cell = contrib.rnn.Conv2DRNNCell((10, 20, 50), 100, 3, 3, prefix='rnn_')
    check_rnn_cell(cell, prefix='rnn_', in_shape=(1, 10, 20, 50), out_shape=(1, 100, 18, 48))

    cell = contrib.rnn.Conv3DRNNCell((10, 20, 30, 50), 100, 3, 3, prefix='rnn_')
    check_rnn_cell(cell, prefix='rnn_', in_shape=(1, 10, 20, 30, 50), out_shape=(1, 100, 18, 28, 48))


def test_convlstm():
    cell = contrib.rnn.Conv1DLSTMCell((10, 50), 100, 3, 3, prefix='rnn_')
    check_rnn_cell(cell, prefix='rnn_', in_shape=(1, 10, 50), out_shape=(1, 100, 48))

    cell = contrib.rnn.Conv2DLSTMCell((10, 20, 50), 100, 3, 3, prefix='rnn_')
    check_rnn_cell(cell, prefix='rnn_', in_shape=(1, 10, 20, 50), out_shape=(1, 100, 18, 48))

    cell = contrib.rnn.Conv3DLSTMCell((10, 20, 30, 50), 100, 3, 3, prefix='rnn_')
    check_rnn_cell(cell, prefix='rnn_', in_shape=(1, 10, 20, 30, 50), out_shape=(1, 100, 18, 28, 48))


def test_convgru():
    cell = contrib.rnn.Conv1DGRUCell((10, 50), 100, 3, 3, prefix='rnn_')
    check_rnn_cell(cell, prefix='rnn_', in_shape=(1, 10, 50), out_shape=(1, 100, 48))

    cell = contrib.rnn.Conv2DGRUCell((10, 20, 50), 100, 3, 3, prefix='rnn_')
    check_rnn_cell(cell, prefix='rnn_', in_shape=(1, 10, 20, 50), out_shape=(1, 100, 18, 48))

    cell = contrib.rnn.Conv3DGRUCell((10, 20, 30, 50), 100, 3, 3, prefix='rnn_')
    check_rnn_cell(cell, prefix='rnn_', in_shape=(1, 10, 20, 30, 50), out_shape=(1, 100, 18, 28, 48))


if __name__ == '__main__':
    import nose
    nose.runmodule()
