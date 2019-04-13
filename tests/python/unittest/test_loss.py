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
from common import setup_module, with_seed, teardown
import unittest


@with_seed()
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

# tracked at: https://github.com/apache/incubator-mxnet/issues/11692
@with_seed()
def test_ce_loss():
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
            eval_metric=mx.metric.Loss(), optimizer='adam',
            initializer=mx.init.Xavier(magnitude=2))
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05

# tracked at: https://github.com/apache/incubator-mxnet/issues/11691
@with_seed()
def test_bce_loss():
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
    assert_almost_equal(mx_bce_loss, npy_bce_loss, rtol=1e-4, atol=1e-5)

@with_seed()
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
    assert_almost_equal(loss_binary(data, label).asnumpy(), loss_bce(data, label).asnumpy(), atol=1e-6)
    assert_almost_equal(loss_signed(data, 2 * label - 1).asnumpy(), loss_bce(data, label).asnumpy(), atol=1e-6)

@with_seed()
def test_kl_loss():
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


@with_seed()
def test_l2_loss():
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


@with_seed()
def test_l1_loss():
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


@with_seed()
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


@with_seed()
def test_ctc_loss_train():
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
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(magnitude=2), eval_metric=mx.metric.Loss(),
            optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 10


@with_seed()
def test_sample_weight_loss():
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


@with_seed(1234)
def test_saveload():
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

@with_seed()
def test_huber_loss():
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


@with_seed()
def test_hinge_loss():
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
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.06


@with_seed()
def test_squared_hinge_loss():
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


@with_seed()
def test_triplet_loss():
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

@with_seed()
def test_cosine_loss():
    #Generating samples
    input1 = mx.nd.random.randn(3, 2)
    input2 = mx.nd.random.randn(3, 2)
    label = mx.nd.sign(mx.nd.random.randn(input1.shape[0]))
    #Calculating loss from cosine embedding loss function in Gluon
    Loss = gluon.loss.CosineEmbeddingLoss()
    loss = Loss(input1, input2, label)

    # Calculating the loss Numpy way
    numerator = mx.nd.sum(input1 * input2, keepdims=True, axis=1)
    denominator = mx.nd.sqrt(mx.nd.sum(input1**2, axis=1, keepdims=True)) \
    * mx.nd.sqrt(mx.nd.sum(input2**2, axis=1, keepdims=True))
    numpy_loss = mx.nd.where(label == 1, 1-numerator/denominator, \
    mx.nd.broadcast_maximum(mx.nd.array([0]), numerator/denominator, axis=1))
    assert_almost_equal(loss.asnumpy(), numpy_loss.asnumpy(), rtol=1e-3, atol=1e-5)

def test_poisson_nllloss():
    pred = mx.nd.random.normal(shape=(3, 4))
    min_pred = mx.nd.min(pred)
    #This is necessary to ensure only positive random values are generated for prediction,
    # to avoid ivalid log calculation
    pred[:] = pred + mx.nd.abs(min_pred)
    target = mx.nd.random.normal(shape=(3, 4))
    min_target = mx.nd.min(target)
    #This is necessary to ensure only positive random values are generated for prediction,
    # to avoid ivalid log calculation
    target[:] += mx.nd.abs(min_target)

    Loss = gluon.loss.PoissonNLLLoss(from_logits=True)
    Loss_no_logits = gluon.loss.PoissonNLLLoss(from_logits=False)
    #Calculating by brute formula for default value of from_logits = True

    # 1) Testing for flag logits = True
    brute_loss = np.mean(np.exp(pred.asnumpy()) - target.asnumpy() * pred.asnumpy())
    loss_withlogits = Loss(pred, target)
    assert_almost_equal(brute_loss, loss_withlogits.asscalar())

    #2) Testing for flag logits = False
    loss_no_logits = Loss_no_logits(pred, target)
    np_loss_no_logits = np.mean(pred.asnumpy() - target.asnumpy() * np.log(pred.asnumpy() + 1e-08))
    if np.isnan(loss_no_logits.asscalar()):
        assert_almost_equal(np.isnan(np_loss_no_logits), np.isnan(loss_no_logits.asscalar()))
    else:
        assert_almost_equal(np_loss_no_logits, loss_no_logits.asscalar())

    #3) Testing for Sterling approximation
    np_pred = np.random.uniform(1, 5, (2, 3))
    np_target = np.random.uniform(1, 5, (2, 3))
    np_compute_full = np.mean((np_pred - np_target * np.log(np_pred + 1e-08)) + ((np_target * np.log(np_target)-\
     np_target + 0.5 * np.log(2 * np_target * np.pi))*(np_target > 1)))
    Loss_compute_full = gluon.loss.PoissonNLLLoss(from_logits=False, compute_full=True)
    loss_compute_full = Loss_compute_full(mx.nd.array(np_pred), mx.nd.array(np_target))
    assert_almost_equal(np_compute_full, loss_compute_full.asscalar())

@with_seed()
def test_poisson_nllloss_mod():
    N = 1000
    data = mx.random.poisson(shape=(N, 2))
    label = mx.random.poisson(lam=4, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=20, label_name='label', shuffle=True)
    output = mx.sym.exp(get_net(1))
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.PoissonNLLLoss(from_logits=False)
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=20, optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Normal(sigma=0.1), eval_metric=mx.metric.Loss(),
            optimizer='adam')
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05

@with_seed()
def test_bce_loss_with_pos_weight():
    # Suppose it's a multi-label classification
    N = np.random.randint(5, 30)
    data = mx.nd.random.uniform(-1, 1, shape=(N, 20))
    label = mx.nd.array(np.random.randint(2, size=(N, 5)), dtype='float32')
    pos_weight = mx.nd.random.uniform(0, 10, shape=(1, 5))
    pos_weight = mx.nd.repeat(pos_weight, repeats=N, axis=0)
    data_iter = mx.io.NDArrayIter(data, {'label': label, 'pos_w': pos_weight}, batch_size=10, label_name='label')
    output = get_net(5)
    l = mx.symbol.Variable('label')
    pos_w = mx.symbol.Variable('pos_w')
    Loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    loss = Loss(output, l, None, pos_w)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label', 'pos_w'))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.01},
            eval_metric=mx.metric.Loss(), optimizer='adam',
            initializer=mx.init.Xavier(magnitude=2))
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.01
    # Test against npy
    data = mx.nd.random.uniform(-5, 5, shape=(N, 5))
    label = mx.nd.array(np.random.randint(2, size=(N, 5)), dtype='float32')
    pos_weight = mx.nd.random.uniform(0, 10, shape=(1, 5))
    mx_bce_loss = Loss(data, label, None, pos_weight).asnumpy()
    prob_npy = 1.0 / (1.0 + np.exp(-data.asnumpy()))
    label_npy = label.asnumpy()
    pos_weight_npy = pos_weight.asnumpy()
    npy_bce_loss = (- label_npy * np.log(prob_npy)*pos_weight_npy - (1 - label_npy) * np.log(1 - prob_npy)).mean(axis=1)
    assert_almost_equal(mx_bce_loss, npy_bce_loss, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    import nose
    nose.runmodule()
