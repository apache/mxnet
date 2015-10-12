# pylint: skip-file
import mxnet as mx
from common import mnist, accuracy
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

def test_lenet(devs, kv_type):
    # guarantee the same weight init for each run
    mx.random.seed(0)
    logging.basicConfig(level=logging.DEBUG)

    (train, val) = mnist(batch_size = 100, input_shape=(1,28,28))

    model = mx.model.FeedForward.create(
        ctx           = devs,
        symbol        = lenet,
        X             = train,
        num_round     = 5,
        learning_rate = 0.05,
        momentum      = 0.9,
        wd            = 0.00001)

    return accuracy(model, val)

if __name__ == "__main__":

    base = test_lenet(mx.gpu(), 'none')
    print base

    cpus = [mx.gpu(i) for i in range(2)]
    acc =  test_mlp(cpus, 'local_update_cpu')
    print acc
