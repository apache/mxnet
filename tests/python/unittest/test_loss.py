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
from mxnet import gluon
from mxnet.test_utils import assert_almost_equal, default_context
import unittest


def test_loss_ndarray():
    output = mx.nd.array([1, 2, 3, 4])
    label = mx.nd.array([1, 3, 5, 7])
    weighting = mx.nd.array([0.5, 1, 0.5, 1])

    loss = gluon.loss.L1Loss()
    assert mx.nd.sum(loss(output, label)).asscalar() == 6.
    loss = gluon.loss.L1Loss(weight=0.5)
    assert mx.nd.sum(loss(output, label)).asscalar() == 3.
    loss = gluon.loss.L1Loss()
    assert mx.nd.sum(loss(output, label, weighting)).asscalar() == 5.

    loss = gluon.loss.L2Loss()
    assert mx.nd.sum(loss(output, label)).asscalar() == 7.
    loss = gluon.loss.L2Loss(weight=0.25)
    assert mx.nd.sum(loss(output, label)).asscalar() == 1.75
    loss = gluon.loss.L2Loss()
    assert mx.nd.sum(loss(output, label, weighting)).asscalar() == 6

    output = mx.nd.array([[0, 2], [1, 4]])
    label = mx.nd.array([0, 1])
    weighting = mx.nd.array([[0.5], [1.0]])

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    L = loss(output, label).asnumpy()
    mx.test_utils.assert_almost_equal(L, np.array([ 2.12692809,  0.04858733]))

    L = loss(output, label, weighting).asnumpy()
    mx.test_utils.assert_almost_equal(L, np.array([ 1.06346405,  0.04858733]))


def get_net(num_hidden, flatten=True):
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128, flatten=flatten)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64, flatten=flatten)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=num_hidden, flatten=flatten)
    return fc3


def test_ce_loss():
    np.random.seed(1234)
    nclass = 10
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, nclass))
    label = mx.nd.array(np.random.randint(0, nclass, size=(N,)), dtype='int32')
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(nclass)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.SoftmaxCrossEntropyLoss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            eval_metric=mx.metric.Loss(), optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05


def test_bce_loss():
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 20))
    label = mx.nd.array(np.random.randint(2, size=(N,)), dtype='float32')
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            eval_metric=mx.metric.Loss(), optimizer='adam',
            initializer=mx.init.Xavier(magnitude=2))
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.01
    # Test against npy
    data = mx.random.uniform(-5, 5, shape=(10,))
    label = mx.random.uniform(0, 1, shape=(10,))
    mx_bce_loss = Loss(data, label).asnumpy()
    prob_npy = 1.0 / (1.0 + np.exp(-data.asnumpy()))
    label_npy = label.asnumpy()
    npy_bce_loss = - label_npy * np.log(prob_npy) - (1 - label_npy) * np.log(1 - prob_npy)
    assert_almost_equal(mx_bce_loss, npy_bce_loss)

def test_bce_equal_ce2():
    N = 100
    loss1 = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
    loss2 = gluon.loss.SoftmaxCELoss(from_logits=True)
    out1 = mx.random.uniform(0.1, 0.9, shape=(N, 1))
    out2 = mx.nd.log(mx.nd.concat(1-out1, out1, dim=1) + 1e-8)
    label = mx.nd.round(mx.random.uniform(0, 1, shape=(N, 1)))
    assert_almost_equal(loss1(out1, label).asnumpy(), loss2(out2, label).asnumpy())

def test_logistic_loss_equal_bce():
    N = 100
    loss_binary = gluon.loss.LogisticLoss(label_format='binary')
    loss_signed = gluon.loss.LogisticLoss(label_format='signed')
    loss_bce = gluon.loss.SigmoidBCELoss(from_sigmoid=False)
    data = mx.random.uniform(-10, 10, shape=(N, 1))
    label = mx.nd.round(mx.random.uniform(0, 1, shape=(N, 1)))
    assert_almost_equal(loss_binary(data, label).asnumpy(), loss_bce(data, label).asnumpy())
    assert_almost_equal(loss_signed(data, 2 * label - 1).asnumpy(), loss_bce(data, label).asnumpy())

def test_kl_loss():
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.nd.softmax(mx.random.uniform(0, 1, shape=(N, 2)))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = mx.sym.log_softmax(get_net(2))
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.KLDivLoss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            eval_metric=mx.metric.Loss(), optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05


def test_l2_loss():
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label', shuffle=True)
    output = get_net(1)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.L2Loss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(magnitude=2), eval_metric=mx.metric.Loss(),
            optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05


def test_l1_loss():
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label', shuffle=True)
    output = get_net(1)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.L1Loss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(magnitude=2), eval_metric=mx.metric.Loss(),
            optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.1


@unittest.skip("flaky test. https://github.com/apache/incubator-mxnet/issues/8892")
def test_ctc_loss():
    loss = gluon.loss.CTCLoss()
    l = loss(mx.nd.ones((2,20,4)), mx.nd.array([[1,0,-1,-1],[2,1,1,-1]]))
    mx.test_utils.assert_almost_equal(l.asnumpy(), np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss(layout='TNC')
    l = loss(mx.nd.ones((20,2,4)), mx.nd.array([[1,0,-1,-1],[2,1,1,-1]]))
    mx.test_utils.assert_almost_equal(l.asnumpy(), np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss(layout='TNC', label_layout='TN')
    l = loss(mx.nd.ones((20,2,4)), mx.nd.array([[1,0,-1,-1],[2,1,1,-1]]).T)
    mx.test_utils.assert_almost_equal(l.asnumpy(), np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss()
    l = loss(mx.nd.ones((2,20,4)), mx.nd.array([[2,1,2,2],[3,2,2,2]]), None, mx.nd.array([2,3]))
    mx.test_utils.assert_almost_equal(l.asnumpy(), np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss()
    l = loss(mx.nd.ones((2,25,4)), mx.nd.array([[2,1,-1,-1],[3,2,2,-1]]), mx.nd.array([20,20]))
    mx.test_utils.assert_almost_equal(l.asnumpy(), np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss()
    l = loss(mx.nd.ones((2,25,4)), mx.nd.array([[2,1,3,3],[3,2,2,3]]), mx.nd.array([20,20]), mx.nd.array([2,3]))
    mx.test_utils.assert_almost_equal(l.asnumpy(), np.array([18.82820702, 16.50581741]))

@unittest.skip("flaky test. https://github.com/apache/incubator-mxnet/issues/8892")
def test_ctc_loss_train():
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 20, 10))
    label = mx.nd.arange(4, repeat=N).reshape((N, 4))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label', shuffle=True)
    output = get_net(5, False)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.CTCLoss(layout='NTC', label_layout='NT')
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 1.},
            initializer=mx.init.Xavier(magnitude=2), eval_metric=mx.metric.Loss(),
            optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 10


def test_sample_weight_loss():
    np.random.seed(1234)
    nclass = 10
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, nclass))
    label = mx.nd.array(np.random.randint(0, nclass, size=(N,)), dtype='int32')
    weight = mx.nd.array([1 for i in range(10)] + [0 for i in range(10)])
    data_iter = mx.io.NDArrayIter(data, {'label': label, 'w': weight}, batch_size=10)
    output = get_net(nclass)
    l = mx.symbol.Variable('label')
    w = mx.symbol.Variable('w')
    Loss = gluon.loss.SoftmaxCrossEntropyLoss()
    loss = Loss(output, l, w)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label', 'w'))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            eval_metric=mx.metric.Loss(), optimizer='adam')
    data_iter = mx.io.NDArrayIter(data[10:], {'label': label, 'w': weight}, batch_size=10)
    score =  mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1]
    assert score > 1
    data_iter = mx.io.NDArrayIter(data[:10], {'label': label, 'w': weight}, batch_size=10)
    score =  mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1]
    assert score < 0.05


def test_saveload():
    mx.random.seed(1234)
    np.random.seed(1234)
    nclass = 10
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, nclass))
    label = mx.nd.array(np.random.randint(0, nclass, size=(N,)), dtype='int32')
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(nclass)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.SoftmaxCrossEntropyLoss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=100, optimizer_params={'learning_rate': 1.},
            eval_metric=mx.metric.Loss())
    mod.save_checkpoint('test', 100, save_optimizer_states=True)
    mod = mx.mod.Module.load('test', 100, load_optimizer_states=True,
                             data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=100, optimizer_params={'learning_rate': 1.},
            eval_metric=mx.metric.Loss())
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05

def test_huber_loss():
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label', shuffle=True)
    output = get_net(1)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.HuberLoss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(magnitude=2), eval_metric=mx.metric.Loss(),
            optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05


def test_hinge_loss():
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.nd.sign(mx.random.uniform(-1, 1, shape=(N, 1)))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label', shuffle=True)
    output = get_net(1)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.HingeLoss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(magnitude=2), eval_metric=mx.metric.Loss(),
            optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05


def test_squared_hinge_loss():
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.nd.sign(mx.random.uniform(-1, 1, shape=(N, 1)))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label', shuffle=True)
    output = get_net(1)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.SquaredHingeLoss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(magnitude=2), eval_metric=mx.metric.Loss(),
            optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05


def test_triplet_loss():
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    pos = mx.random.uniform(-1, 1, shape=(N, 10))
    neg = mx.random.uniform(-1, 1, shape=(N, 10))
    data_iter = mx.io.NDArrayIter(data, {'pos': pos, 'neg': neg}, batch_size=10,
                                  label_name='label', shuffle=True)
    output = get_net(10)
    pos = mx.symbol.Variable('pos')
    neg = mx.symbol.Variable('neg')
    Loss = gluon.loss.TripletLoss()
    loss = Loss(output, pos, neg)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('pos','neg'))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(magnitude=2), eval_metric=mx.metric.Loss(),
            optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05


if __name__ == '__main__':
    import nose
    nose.runmodule()
