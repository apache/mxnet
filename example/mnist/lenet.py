# pylint: skip-file
from data import mnist_iterator
import mxnet as mx
import logging

## define lenet
# input
data = mx.symbol.Variable('data')
# first conv
conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                          kernel=(2,2), stride=(2,2))
# second conv
conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                          kernel=(2,2), stride=(2,2))
# first fullc
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet = mx.symbol.Softmax(data=fc2)

## data
train, val = mnist_iterator(batch_size=100, input_shape=(1,28,28))

## train
logging.basicConfig(level=logging.DEBUG)
# dev = [mx.gpu(i) for i in range(2)]
dev = mx.gpu()
model = mx.model.FeedForward(
    ctx = dev, symbol = lenet, num_round = 20,
    learning_rate = 0.05, momentum = 0.9, wd = 0.00001)
model.fit(X=train, eval_data=val,
          epoch_end_callback=mx.callback.Speedometer(100))
