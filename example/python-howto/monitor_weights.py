# pylint: skip-file
from data import mnist_iterator
import mxnet as mx
import numpy as np
import logging

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
mlp = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

# data

train, val = mnist_iterator(batch_size=100, input_shape = (784,))

# train

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(
    ctx = mx.cpu(), symbol = mlp, num_epoch = 20,
    learning_rate = 0.1, momentum = 0.9, wd = 0.00001)

def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)
mon = mx.mon.Monitor(100, norm_stat)
model.fit(X=train, eval_data=val, monitor=mon, 
          batch_end_callback = mx.callback.Speedometer(100, 100))

