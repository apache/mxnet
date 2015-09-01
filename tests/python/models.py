"""This file defines various models used in the test"""
import mxnet as mx

def mlp2():
    data = mx.symbol.Variable('data')
    out = mx.symbol.FullyConnected(data=data, name='fc1', nb_hidden=1000)
    out = mx.symbol.Activation(data=out, act_type='relu')
    out = mx.symbol.FullyConnected(data=out, name='fc2', nb_hidden=10)
    return out

