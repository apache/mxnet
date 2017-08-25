"""This file defines various models used in the test"""
import mxnet as mx

def mlp2():
    data = mx.symbol.Variable('data')
    out = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=1000)
    out = mx.symbol.Activation(data=out, act_type='relu')
    out = mx.symbol.FullyConnected(data=out, name='fc2', num_hidden=10)
    return out



def conv():
    data = mx.symbol.Variable('data')
    conv1= mx.symbol.Convolution(data = data, name='conv1', num_filter=32, kernel=(3,3), stride=(2,2))
    bn1 = mx.symbol.BatchNorm(data = conv1, name="bn1")
    act1 = mx.symbol.Activation(data = bn1, name='relu1', act_type="relu")
    mp1 = mx.symbol.Pooling(data = act1, name = 'mp1', kernel=(2,2), stride=(2,2), pool_type='max')

    conv2= mx.symbol.Convolution(data = mp1, name='conv2', num_filter=32, kernel=(3,3), stride=(2,2))
    bn2 = mx.symbol.BatchNorm(data = conv2, name="bn2")
    act2 = mx.symbol.Activation(data = bn2, name='relu2', act_type="relu")
    mp2 = mx.symbol.Pooling(data = act2, name = 'mp2', kernel=(2,2), stride=(2,2), pool_type='max')

    fl = mx.symbol.Flatten(data = mp2, name="flatten")
    fc2 = mx.symbol.FullyConnected(data = fl, name='fc2', num_hidden=10)
    softmax = mx.symbol.SoftmaxOutput(data = fc2, name = 'sm')
    return softmax

