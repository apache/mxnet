# pylint: skip-file
import mxnet as mx
import numpy as np
import os, gzip
import pickle as pickle
from common import get_data

# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
softmax = mx.symbol.Softmax(data = fc3, name = 'sm')

# infer shape
data_shape = (batch_size, 784)

model = mx.model.FeedForward(softmax, mx.cpu(), data_shape,
                             num_round=9, learning_rate=0.1, wd=0.0004,
                             momentum=0)
#check data
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

def test_mlp():
    model.fit(X=train_dataiter,
              eval_data=val_dataiter)

if __name__ == "__main__":
    test_mlp()
