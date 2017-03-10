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
    mx.random.seed(1234)
    np.random.seed(1234)
    nclass = 10
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, nclass))
    label = mx.nd.array(np.random.randint(0, nclass, size=(N,)), dtype='int32')
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(nclass)
    l = mx.symbol.Variable('label')
    loss = mx.loss.cross_entropy_loss(output, l)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, eval_metric=loss.metric, num_epoch=200, optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter, loss.metric)[0][1] == 1.0


def test_l2_loss():
    mx.random.seed(1234)
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    loss = mx.loss.l2_loss(output, l)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, eval_metric=loss.metric, num_epoch=200, optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter, loss.metric)[0][1] < 0.05


def test_l1_loss():
    mx.random.seed(1234)
    np.random.seed(1234)
    N = 20
    data = mx.random.uniform(-1, 1, shape=(N, 10))
    label = mx.random.uniform(-1, 1, shape=(N, 1))
    data_iter = mx.io.NDArrayIter(data, label, batch_size=10, label_name='label')
    output = get_net(1)
    l = mx.symbol.Variable('label')
    loss = mx.loss.l1_loss(output, l)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, eval_metric=loss.metric, num_epoch=200, optimizer_params={'learning_rate': 0.1},
            initializer=mx.init.Uniform(0.5))
    assert mod.score(data_iter, loss.metric)[0][1] < 0.1


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
    loss = mx.loss.custom_loss(loss, output, ['label'], weight=0.5)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, eval_metric=loss.metric, num_epoch=200,
            optimizer_params={'learning_rate': 1.})
    assert mod.score(data_iter, loss.metric)[0][1] < 0.05


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
    loss = mx.loss.cross_entropy_loss(output, l, sample_weight=w)
    mod = mx.mod.Module(loss)
    mod.fit(data_iter, eval_metric=loss.metric, num_epoch=200,
            optimizer_params={'learning_rate': 1.})
    score =  mod.score(data_iter, loss.metric)[0][1]
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
    loss1 = mx.loss.cross_entropy_loss(output1, l1)
    loss2 = mx.loss.l2_loss(output2, l2)
    loss = mx.loss.multi_loss([loss1, loss2])
    mod = mx.mod.Module(loss)

    mod.fit(data_iter, eval_metric=loss.metric, num_epoch=200,
            optimizer_params={'learning_rate': 0.5},
            initializer=mx.init.Uniform(0.1))
    score = mod.score(data_iter, loss.metric)
    assert score[0][1] == 1.0
    assert score[2][1] < 0.1


if __name__ == '__main__':
    test_multi_loss()
    test_sample_weight_loss()
    test_custom_loss()
    test_l2_loss()
    test_l1_loss()
    test_ce_loss()
