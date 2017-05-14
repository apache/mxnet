import mxnet as mx
import numpy as np
from mxnet.contrib import nn


def test_loss_ndarray():
    output = mx.nd.array([1, 2, 3, 4])
    label = mx.nd.array([1, 3, 5, 7])
    weighting = mx.nd.array([0.5, 1, 0.5, 1])

    assert mx.nd.sum(nn.loss.l1_loss(output, label)).asscalar() == 6.
    assert mx.nd.sum(nn.loss.l1_loss(output, label, weight=0.5)).asscalar() == 3.
    assert mx.nd.sum(nn.loss.l1_loss(output, label, sample_weight=weighting)).asscalar() == 5.

    assert mx.nd.sum(nn.loss.l2_loss(output, label)).asscalar() == 7.
    assert mx.nd.sum(nn.loss.l2_loss(output, label, weight=0.25)).asscalar() == 1.75
    assert mx.nd.sum(nn.loss.l2_loss(output, label, sample_weight=weighting)).asscalar() == 6

    output = mx.nd.array([[0, 2], [1, 4]])
    label = mx.nd.array([0, 1])
    weighting = mx.nd.array([[0.5, 1.0]])

    loss = nn.loss.softmax_cross_entropy_loss(output, label).asnumpy()
    mx.test_utils.assert_almost_equal(loss, np.array([ 2.12692809,  0.04858733]))

    loss = nn.loss.softmax_cross_entropy_loss(output, label, sample_weight=weighting).asnumpy()
    mx.test_utils.assert_almost_equal(loss, np.array([ 2.12692809,  0.04858733])*weighting.asnumpy())


def check_loss(loss):
    output = mx.sym.var('data')
    pred1 = mx.sym.var('data1')
    pred2 = mx.sym.var('data2')
    label = mx.sym.var('label')

    sym = loss(output, label, name='loss1')
    assert sym.list_outputs()[1] == 'loss1_loss'
    assert sym.list_arguments() == ['data', 'label']
    assert sym[0].list_arguments() == ['data']
    assert sym[1].list_attr()['__output__'] == 'loss'

    sym = loss(output, label, sample_weight=pred1, name='loss1')
    assert sym.list_outputs()[1] == 'loss1_loss'
    assert sym.list_arguments() == ['data', 'label', 'data1']
    assert sym[0].list_arguments() == ['data']

    sym = loss(output, label, extra_outputs=(pred1, pred2), name='loss2')
    assert sym.list_outputs()[1:] == ['data1_out_output', 'data2_out_output', 'loss2_loss']


def test_loss_symbol():
    check_loss(nn.loss.l1_loss)
    check_loss(nn.loss.l2_loss)
    check_loss(nn.loss.softmax_cross_entropy_loss)


def get_net(num_hidden):
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=num_hidden)
    return fc3


def test_ce_loss():
    mx.random.seed(1234)
    np.random.seed(1234)
    nclass = 10
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, nclass))
    label = mx.nd.array(np.random.randint(0, nclass, size=(N,)), dtype='int32')
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(nclass)
    fc2 = output.get_internals()['fc2_output']
    l = mx.symbol.Variable('label')
    loss = nn.loss.softmax_cross_entropy_loss(output, l, extra_outputs=(fc2,))
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter)[0][1] == 1.0


def test_l2_loss():
    mx.random.seed(1234)
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    loss = nn.loss.l2_loss(output, l)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter)[0][1] < 0.05


def test_l1_loss():
    mx.random.seed(1234)
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    loss = nn.loss.l1_loss(output, l)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.1},
            initializer=mx.init.Uniform(0.5))
    assert mod.score(data_iter)[0][1] < 0.1


def test_custom_loss():
    mx.random.seed(1234)
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    loss = mx.sym.square(output - l)
    loss = nn.loss.custom_loss(loss, output, l, weight=0.5, metrics='mse')
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200,
            optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter)[0][1] < 0.05


def test_sample_weight_loss():
    mx.random.seed(1234)
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
    loss = nn.loss.softmax_cross_entropy_loss(output, l, sample_weight=w)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label', 'w'))
    mod.fit(data_iter, num_epoch=200,
            optimizer_params={'learning_rate': 1.})
    score =  mod.score(data_iter)[0][1]
    assert score >= 0.5 and score <= 0.75


def test_multi_loss():
    mx.random.seed(1234)
    np.random.seed(1234)
    nclass = 10
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, nclass))
    label1 = mx.nd.array(np.random.randint(0, nclass, size=(N,)), dtype='int32')
    label2 = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, {'label1': label1, 'label2': label2},
                                  batch_size=10, label_name='label')
    fc3 = get_net(64)
    act3 = mx.symbol.Activation(fc3, name='relu3', act_type="relu")
    output1 = mx.symbol.FullyConnected(act3, name='output1', num_hidden=10)
    output2 = mx.symbol.FullyConnected(act3, name='output2', num_hidden=1)
    l1 = mx.symbol.Variable('label1')
    l2 = mx.symbol.Variable('label2')
    loss1 = nn.loss.softmax_cross_entropy_loss(output1, l1)
    loss2 = nn.loss.l2_loss(output2, l2)
    loss = nn.loss.multitask_loss([loss1, loss2])
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label1', 'label2'))

    mod.fit(data_iter, num_epoch=200,
            optimizer_params={'learning_rate': 0.5},
            initializer=mx.init.Uniform(0.1))
    score = mod.score(data_iter)
    assert score[0][1] == 1.0
    assert score[2][1] < 0.2


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
    loss = nn.loss.softmax_cross_entropy_loss(output, l)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=100, optimizer_params={'learning_rate': 1.})
    mod.save_checkpoint('test', 100, save_optimizer_states=True)
    mod = mx.mod.Module.load('test', 100, load_optimizer_states=True,
                             data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=100, optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter)[0][1] == 1.0


if __name__ == '__main__':
    import nose
    nose.runmodule()
