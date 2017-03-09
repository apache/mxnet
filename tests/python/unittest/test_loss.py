import mxnet as mx
import numpy as np

def get_net(num_hidden):
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=num_hidden)
    return fc3


def test_ce_loss():
    nclass = 10
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, nclass))
    label = mx.nd.array(np.random.randint(0, nclass, size=(N,)), dtype='int32')
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(nclass)
    l = mx.symbol.Variable('label')
    loss = mx.loss.cross_entropy_loss(output, l)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter, 'acc')[0][1] == 1.0


def test_l2_loss():
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    loss = mx.loss.l2_loss(output, l)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter, 'mse')[0][1] < 0.05


def test_l1_loss():
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    loss = mx.loss.l1_loss(output, l)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 0.1},
            initializer=mx.init.Uniform(0.5))
    assert mod.score(data_iter, 'mse')[0][1] < 0.05


def test_custom_loss():
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    loss = mx.sym.square(output - l)
    loss = mx.loss.custom_loss(loss, output, ['label'], weight=0.5)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, num_epoch=200, optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter, 'mse')[0][1] < 0.05


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
    loss = mx.loss.cross_entropy_loss(output, l, sample_weight=w)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, eval_metric=None, num_epoch=200, optimizer_params={'learning_rate': 1.})
    #assert mod.score(data_iter, 'acc')[0][1] == 1.0


if __name__ == '__main__':
    test_sample_weight_loss()
    test_custom_loss()
    test_l2_loss()
    test_l1_loss()
    test_ce_loss()
