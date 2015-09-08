# pylint: skip-file
import mxnet as mx
import numpy as np
import os, gzip
import pickle as pickle
import sys
import get_data

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
arg_shapes, out_shapes, aux_shapes = softmax.infer_shape(data=data_shape)

arg_narrays = [mx.narray.create(shape, ctx=mx.Context("gpu")) for shape in arg_shapes]
grad_narrays = [mx.narray.create(shape, ctx=mx.Context("gpu")) for shape in arg_shapes]

inputs = dict(zip(args_list, arg_narrays))

name2shape = dict(zip(args_list, arg_shapes))
pred = mx.narray.create(out_shapes[0])

np.random.seed(0)
# set random weight
for name, narray in inputs.items():
    if "weight" in name:
        tmp = mx.narray.create(name2shape[name])
        tmp.numpy[:] = np.random.uniform(-0.07, 0.07, name2shape[name])
        tmp.copyto(narray)
    if "bias" in name:
        narray[:] = 0.0

# bind executer
# TODO(bing): think of a better bind interface
executor = softmax.bind(mx.Context('gpu'), arg_narrays, grad_narrays)
# update

out_narray = executor.heads()[0]
grad_narray = mx.narray.create(out_narray.shape)

epoch = 9
lr = 0.1
wd = 0.0004

def Update(grad, weight):
    weight[:] -= lr * grad  / batch_size

block = list(zip(grad_narrays, arg_narrays))

#check data
get_data.GetMNIST_ubyte()

train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        batch_size=batch_size, shuffle=True, flat=True, silent=False)

tmp_label = mx.narray.create(name2shape["sm_label"])

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
            data = data
            tmp_label.numpy[:] = label.numpy.reshape(tmp_label.shape)
            data.copyto(inputs["data"])
            tmp_label.copyto(inputs["sm_label"])
            executor.forward()
            out_narray.copyto(pred)
            train_acc += CalAcc(pred.numpy, label.numpy.flatten())
            train_nbatch += 1
            out_narray.copyto(grad_narray)
            executor.backward([grad_narray])

            for grad, weight in block:
                Update(grad, weight)

        # evaluate
        for data, label in val_dataiter:
            data = data
            label = label.numpy.flatten()
            data.copyto(inputs["data"])
            executor.forward()
            out_narray.copyto(pred)
            val_acc += CalAcc(pred.numpy, label)
            val_nbatch += 1
        acc_train = train_acc / train_nbatch
        acc_val = val_acc / val_nbatch
        print("Train Acc: ", train_acc / train_nbatch)
        print("Valid Acc: ", val_acc / val_nbatch)
        train_dataiter.reset()
        val_dataiter.reset()
    assert(acc_train > 0.98)
    assert(acc_val > 0.97)

if __name__ == "__main__":
    test_mlp()
