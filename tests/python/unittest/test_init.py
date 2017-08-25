import mxnet as mx
import numpy as np

def test_default_init():
    data = mx.sym.Variable('data')
    sym = mx.sym.LeakyReLU(data=data, act_type='prelu')
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (10,10))])
    mod.init_params()
    assert (list(mod.get_params()[0].values())[0].asnumpy() == 0.25).all()

def test_variable_init():
    data = mx.sym.Variable('data')
    gamma = mx.sym.Variable('gamma', init=mx.init.One())
    sym = mx.sym.LeakyReLU(data=data, gamma=gamma, act_type='prelu')
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (10,10))])
    mod.init_params()
    assert (list(mod.get_params()[0].values())[0].asnumpy() == 1).all()

def test_aux_init():
    data = mx.sym.Variable('data')
    sym = mx.sym.BatchNorm(data=data, name='bn')
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (10, 10, 3, 3))])
    mod.init_params()
    assert (mod.get_params()[1]['bn_moving_var'].asnumpy() == 1).all()
    assert (mod.get_params()[1]['bn_moving_mean'].asnumpy() == 0).all()


if __name__ == '__main__':
    test_variable_init()
    test_default_init()
    test_aux_init()
