# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
sys.path.append("../../tests/python/common")
import mxnet as mx
import logging
import numpy as np
import get_data

# define mlp

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
mlp = mx.symbol.Softmax(data = fc3, name = 'mlp')

# data

batch_size = 100

get_data.GetMNIST_ubyte()
train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        input_shape=(784,),
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        input_shape=(784,),
        batch_size=batch_size, shuffle=True, flat=True, silent=False)


# train

logging.basicConfig(level=logging.DEBUG)

model = mx.model.FeedForward(ctx = mx.cpu(),
                             symbol = mlp,
                             num_round = 10,
                             learning_rate = 0.1,
                             momentum = 0.9,
                             wd = 0.00001)

model.fit(X=train_dataiter, eval_data=val_dataiter)
