import os
import mxnet as mx
from common import models
import pickle as pkl

def test_attr_basic():
    with mx.AttrScope(group='4', data='great'):
        data = mx.symbol.Variable('data',
                                  attr={'dtype':'data',
                                        'group': '1'})
        gdata = mx.symbol.Variable('data2')
    assert gdata.attr('group') == '4'
    assert data.attr('group') == '1'
    data2 = pkl.loads(pkl.dumps(data))
    assert data.attr('dtype') == data2.attr('dtype')

def test_operator():
    data = mx.symbol.Variable('data')
    with mx.AttrScope(group='4', data='great'):
        fc1 = mx.symbol.Activation(data, act_type='relu')
        with mx.AttrScope(init_bias='0.0'):
            fc2 = mx.symbol.FullyConnected(fc1, num_hidden=10, name='fc2')
    assert fc1.attr('data') == 'great'
    fc2copy = pkl.loads(pkl.dumps(fc2))
    assert fc2copy.tojson() == fc2.tojson()
    fc2weight = fc2.get_internals()['fc2_weight']


if __name__ == '__main__':
    test_attr_basic()
    test_operator()


