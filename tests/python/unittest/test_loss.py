import mxnet as mx
import numpy as np
from mxnet import gluon


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
    Loss = gluon.loss.SoftmaxCrossEntropyLoss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 1.},
            eval_metric=mx.metric.Loss())
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.01


def test_l2_loss():
    mx.random.seed(1234)
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.L2Loss()
    Loss(label, label)
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 1.},
            eval_metric=mx.metric.Loss())
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.05

def test_l1_loss():
    mx.random.seed(1234)
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    Loss = gluon.loss.L1Loss()
    loss = Loss(output, l)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label',))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.1},
            initializer=mx.init.Uniform(0.5), eval_metric=mx.metric.Loss())
    assert mod.score(data_iter, eval_metric=mx.metric.Loss())[0][1] < 0.1


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
    Loss = gluon.loss.SoftmaxCrossEntropyLoss()
    loss = Loss(output, l, w)
    loss = mx.sym.make_loss(loss)
    mod = mx.mod.Module(loss, data_names=('data',), label_names=('label', 'w'))
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 1.},
            eval_metric=mx.metric.Loss())
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


if __name__ == '__main__':
    import nose
    nose.runmodule()
