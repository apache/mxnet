# pylint: skip-file
import mxnet as mx
import numpy as np
import os, pickle, gzip
from common import get_data

def CalAcc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label) * 1.0 / out.shape[0]

# symbol net
batch_size = 100
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
softmax = mx.symbol.Softmax(data = fc2, name = 'sm')
args_list = softmax.list_arguments()
# infer shape
#data_shape = (batch_size, 784)

data_shape = (batch_size, 1, 28, 28)
arg_shapes, out_shapes, aux_shapes = softmax.infer_shape(data=data_shape)
arg_narrays = [mx.nd.empty(shape) for shape in arg_shapes]
grad_narrays = [mx.nd.empty(shape) for shape in arg_shapes]
aux_narrays = [mx.nd.empty(shape) for shape in aux_shapes]

inputs = dict(zip(args_list, arg_narrays))
np.random.seed(0)
# set random weight
for name, narray in inputs.items():
    if "weight" in name:
        narray[:] = np.random.uniform(-0.07, 0.07, narray.shape)
    if "bias" in name:
        narray[:] = 0.0
    if "gamma" in name:
        narray[:] = 1.0
    if "beta" in name:
        narray[:] = 0.0

# bind executer
# TODO(bing): think of a better bind interface

executor = softmax.bind(mx.cpu(), arg_narrays, grad_narrays, 'write', aux_narrays)
# update
print(executor.debug_str())
out_narray = executor.outputs[0]
grad_narray = mx.nd.empty(out_narray.shape)

epoch = 1
momentum = 0.9
lr = 0.1
wd = 0.0004

def Update(grad, weight):
    weight[:] -= lr * grad / batch_size

block = list(zip(grad_narrays, arg_narrays))

# check data
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

def test_mnist():
    acc_train = 0.0
    acc_val = 0.0
    for i in range(epoch):
        # train
        print("Epoch %d" % i)
        train_acc = 0.0
        val_acc = 0.0
        train_nbatch = 0
        val_nbatch = 0
        for data, label in train_dataiter:
            label = label.asnumpy().flatten()
            inputs["data"][:] = data
            inputs["sm_label"][:] = label
            executor.forward(is_train = True)
            train_acc += CalAcc(out_narray.asnumpy(), label)
            train_nbatch += 1
            grad_narray[:] = out_narray
            executor.backward([grad_narray])

            for grad, weight in block:
                Update(grad, weight)

        # evaluate
        for data, label in val_dataiter:
            label = label.asnumpy().flatten()
            inputs["data"][:] = data
            executor.forward(is_train = False)
            val_acc += CalAcc(out_narray.asnumpy(), label)
            val_nbatch += 1
        print("Train Acc: ", train_acc / train_nbatch)
        print("Valid Acc: ", val_acc / val_nbatch)
        acc_train = train_acc / train_nbatch
        acc_val = val_acc / val_nbatch
        train_dataiter.reset()
        val_dataiter.reset()
    assert(acc_train > 0.84)
    assert(acc_val > 0.96)


if __name__ == "__main__":
    test_mnist()

