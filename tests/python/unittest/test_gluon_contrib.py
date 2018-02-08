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
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import Concurrent, HybridConcurrent, Identity
from mxnet.test_utils import almost_equal
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


def test_conv_fill_shape():
    cell = contrib.rnn.Conv1DLSTMCell((0, 7), 10, (3,), (3,))
    cell.hybridize()
    check_rnn_forward(cell, mx.nd.ones((8, 3, 5, 7)))
    assert cell.i2h_weight.shape[1] == 5, cell.i2h_weight.shape[1]


def test_vardrop():
    def check_vardrop(drop_inputs, drop_states, drop_outputs):
        cell = contrib.rnn.VariationalDropoutCell(mx.gluon.rnn.RNNCell(100, prefix='rnn_'),
                                                  drop_outputs=drop_outputs,
                                                  drop_states=drop_states,
                                                  drop_inputs=drop_inputs)
        cell.collect_params().initialize(init='xavier')
        input_data = mx.nd.random_uniform(shape=(10, 3, 50), ctx=mx.context.current_context())
        with mx.autograd.record():
            outputs1, _ = cell.unroll(3, input_data, merge_outputs=True)
            mask1 = cell.drop_outputs_mask.asnumpy()
            mx.nd.waitall()
            outputs2, _ = cell.unroll(3, input_data, merge_outputs=True)
            mask2 = cell.drop_outputs_mask.asnumpy()
        assert not almost_equal(mask1, mask2)
        assert not almost_equal(outputs1.asnumpy(), outputs2.asnumpy())

        inputs = [mx.sym.Variable('rnn_t%d_data'%i) for i in range(3)]
        outputs, _ = cell.unroll(3, inputs, merge_outputs=False)
        outputs = mx.sym.Group(outputs)

        args, outs, auxs = outputs.infer_shape(rnn_t0_data=(10,50), rnn_t1_data=(10,50), rnn_t2_data=(10,50))
        assert outs == [(10, 100), (10, 100), (10, 100)]

        cell.reset()
        cell.hybridize()
        with mx.autograd.record():
            outputs3, _ = cell.unroll(3, input_data, merge_outputs=True)
            mx.nd.waitall()
            outputs4, _ = cell.unroll(3, input_data, merge_outputs=True)
        assert not almost_equal(outputs3.asnumpy(), outputs4.asnumpy())
        assert not almost_equal(outputs1.asnumpy(), outputs3.asnumpy())

    check_vardrop(0.5, 0.5, 0.5)
    check_vardrop(0.5, 0, 0.5)


def test_concurrent():
    model = HybridConcurrent(axis=1)
    model.add(nn.Dense(128, activation='tanh', in_units=10))
    model.add(nn.Dense(64, activation='tanh', in_units=10))
    model.add(nn.Dense(32, in_units=10))
    model2 = Concurrent(axis=1)
    model2.add(nn.Dense(128, activation='tanh', in_units=10))
    model2.add(nn.Dense(64, activation='tanh', in_units=10))
    model2.add(nn.Dense(32, in_units=10))

    # symbol
    x = mx.sym.var('data')
    y = model(x)
    assert len(y.list_arguments()) == 7

    # ndarray
    model.initialize(mx.init.Xavier(magnitude=2.24))
    model2.initialize(mx.init.Xavier(magnitude=2.24))
    x = model(mx.nd.zeros((32, 10)))
    x2 = model2(mx.nd.zeros((32, 10)))
    assert x.shape == (32, 224)
    assert x2.shape == (32, 224)
    x.wait_to_read()
    x2.wait_to_read()


def test_identity():
    model = Identity()
    x = mx.nd.random.uniform(shape=(128, 33, 64))
    mx.test_utils.assert_almost_equal(model(x).asnumpy(),
                                      x.asnumpy())

def test_datasets():
    wikitext2_train = contrib.data.text.WikiText2(root='data/wikitext-2', segment='train')
    wikitext2_val = contrib.data.text.WikiText2(root='data/wikitext-2', segment='validation',
                                                vocab=wikitext2_train.vocabulary)
    wikitext2_test = contrib.data.text.WikiText2(root='data/wikitext-2', segment='test')
    assert len(wikitext2_train) == 42780
    assert len(wikitext2_train.vocabulary) == 33278
    assert len(wikitext2_train.frequencies) == 33277
    assert len(wikitext2_val) == 632
    assert len(wikitext2_val.vocabulary) == 33278
    assert len(wikitext2_val.frequencies) == 13776
    assert len(wikitext2_test) == 15941
    assert len(wikitext2_test.vocabulary) == 14143, len(wikitext2_test.vocabulary)
    assert len(wikitext2_test.frequencies) == 14142, len(wikitext2_test.frequencies)
    assert wikitext2_test.frequencies['English'] == 32


def test_sampler():
    interval_sampler = contrib.data.IntervalSampler(10, 3)
    assert sorted(list(interval_sampler)) == list(range(10))
    interval_sampler = contrib.data.IntervalSampler(10, 3, rollover=False)
    assert list(interval_sampler) == [0, 3, 6, 9]


if __name__ == '__main__':
    import nose
    nose.runmodule()
