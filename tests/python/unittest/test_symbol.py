import mxnet as mx
from common import models

def test_symbol_basic():
    mlist = []
    mlist.append(models.mlp2())
    for m in mlist:
        m.list_arguments()
        m.list_returns()


def test_compose():
    data = mx.symbol.Variable('data')
    net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
    net1 = mx.symbol.FullyConnected(data=net1, name='fc2', num_hidden=100)
    net1.list_arguments() == ['data',
                              'fc1_weight', 'fc1_bias',
                              'fc2_weight', 'fc2_bias']

    net2 = mx.symbol.FullyConnected(name='fc3', num_hidden=10)
    net2 = mx.symbol.Activation(data=net2)
    net2 = mx.symbol.FullyConnected(data=net2, name='fc4', num_hidden=20)
    print(net2.debug_str())

    composed = net2(fc3_data=net1, name='composed')
    print(composed.debug_str())
    multi_out = mx.symbol.Group([composed, net1])
    assert len(multi_out.list_returns()) == 2
