# pylint: skip-file
import mxnet as mx
import numpy as np
import os, gzip
import pickle as pickle
import sys
def CalAcc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label) * 1.0 / out.shape[0]

# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
softmax = mx.symbol.Softmax(data = fc3, name = 'sm')
args_list = softmax.list_arguments()
# infer shape
data_shape = (batch_size, 784)
arg_shapes, out_shapes = softmax.infer_shape(data=data_shape)
arg_narrays = [mx.narray.create(shape) for shape in arg_shapes]
grad_narrays = [mx.narray.create(shape) for shape in arg_shapes]
inputs = dict(zip(args_list, arg_narrays))
np.random.seed(0)
# set random weight
for name, narray in inputs.items():
    if "weight" in name:
        narray.numpy[:, :] = np.random.uniform(-0.07, 0.07, narray.numpy.shape)
    if "bias" in name:
        narray.numpy[:] = 0.0

req = ['write_to' for i in range(len(arg_narrays))]
# bind executer
# TODO(bing): think of a better bind interface
executor = softmax.bind(mx.Context('cpu'), arg_narrays, grad_narrays, req)
# update

out_narray = executor.heads()[0]
grad_narray = mx.narray.create(out_narray.shape)

epoch = 9
lr = 0.1
wd = 0.0004
def Update(grad, weight):
    weight.numpy[:] -= lr * grad.numpy[:]  / batch_size

block = zip(grad_narrays, arg_narrays)

train_dataiter = mx.io.MNISTIter(
        image="/home/tianjun/data/mnist/train-images-idx3-ubyte",
        label="/home/tianjun/data/mnist/train-labels-idx1-ubyte",
        batch_size=batch_size, shuffle=1, flat=1, silent=0, seed=10)
val_dataiter = mx.io.MNISTIter(
        image="/home/tianjun/data/mnist/t10k-images-idx3-ubyte",
        label="/home/tianjun/data/mnist/t10k-labels-idx1-ubyte",
        batch_size=batch_size, shuffle=1, flat=1, silent=0)

def test_mlp():
    acc_train = 0.
    acc_val = 0.
    for i in range(epoch):
        # train
        print("Epoch %d" % i)
        train_acc = 0.0
        val_acc = 0.0
        train_nbatch = 0
        val_nbatch = 0
        for data, label in train_dataiter:
            data = data.numpy
            label = label.numpy.flatten()
            inputs["data"].numpy[:] = data
            inputs["sm_label"].numpy[:] = label
            executor.forward()
            train_acc += CalAcc(out_narray.numpy, label)
            train_nbatch += 1
            grad_narray.numpy[:] = out_narray.numpy
            executor.backward([grad_narray])

            for grad, weight in block:
                Update(grad, weight)

        # evaluate
        for data, label in val_dataiter:
            data = data.numpy
            label = label.numpy.flatten()
            inputs["data"].numpy[:] = data
            executor.forward()
            val_acc += CalAcc(out_narray.numpy, label)
            val_nbatch += 1
        acc_train = train_acc / train_nbatch
        acc_val = val_acc / val_nbatch
        print("Train Acc: ", train_acc / train_nbatch)
        print("Valid Acc: ", val_acc / val_nbatch)
    assert(acc_train > 0.98)
    assert(acc_val > 0.97)

