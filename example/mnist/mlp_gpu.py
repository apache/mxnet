# pylint: skip-file

import mxnet as mx
import numpy as np
import os, gzip
import pickle as pickle
import sys
sys.path.append("../../tests/python")
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

# create GPU NArray for data
arg_narrays = [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]
grad_narrays = [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]
inputs = dict(zip(args_list, arg_narrays))

# create CPU NArray for result stat
name2shape = dict(zip(args_list, arg_shapes))
pred = mx.nd.zeros(out_shapes[0])


# set random weight
np.random.seed(0)
for name, narray in inputs.items():
    if "weight" in name:
        tmp = mx.nd.array(np.random.uniform(-0.07, 0.07, name2shape[name]))
        tmp.copyto(narray)

# bind executer
# TODO(bing): think of a better bind interface
executor = softmax.bind(mx.Context('gpu'), arg_narrays, grad_narrays)
# create gradient NArray
out_narray = executor.outputs[0]
grad_narray = mx.nd.zeros(out_narray.shape, ctx=mx.gpu())


# update
epoch = 9
lr = 0.1
wd = 0.0004

# SGD Update rule
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

tmp_label = mx.nd.zeros(name2shape["sm_label"])

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
            label = label.asnumpy().reshape(tmp_label.shape)
            tmp_label[:] = label
            inputs["data"][:] = data
            inputs["sm_label"][:] = tmp_label
            executor.forward()
            pred[:] = out_narray
            train_acc += CalAcc(pred.asnumpy(), label)
            train_nbatch += 1
            grad_narray[:] = out_narray
            executor.backward([grad_narray])

            for grad, weight in block:
                Update(grad, weight)

        # evaluate
        for data, label in val_dataiter:
            label = label.asnumpy().flatten()
            inputs["data"][:] = data
            executor.forward()
            pred[:] = out_narray
            val_acc += CalAcc(pred.asnumpy(), label)
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
