# pylint: skip-file
import mxnet as mx
import numpy as np
import os, cPickle, gzip

def Softmax(x):
    batch, nidden = x.shape
    maxes = np.max(x, axis=1)
    x -= maxes.reshape(batch, 1)
    x = np.exp(x)
    norm = np.sum(x, axis=1)
    prob = x / norm.reshape((batch, 1))
    return prob

def CalAcc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label.transpose()) * 1.0 / out.shape[0]

def SetGradient(out_grad, label):
    assert(out_grad.shape[0] == label.shape[0])
    for i in xrange(label.shape[0]):
        k = label[i]
        out_grad[i][k] -= 1.0

# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=160)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=10)
args_list = fc2.list_arguments()
# infer shape
data_shape = (batch_size, 1, 1, 784)
arg_shapes, out_shapes = fc2.infer_shape(data=data_shape)
arg_narrays = [mx.narray.create(shape) for shape in arg_shapes]
grad_narrays = [mx.narray.create(shape) for shape in arg_shapes]
mom_narrays = [mx.narray.create(shape) for shape in arg_shapes]
inputs = dict(zip(args_list, arg_narrays))

np.random.seed(0)
# set random weight
for name, narray in inputs.items():
    if "weight" in name:
        narray.numpy[:, :] = np.random.uniform(-0.001, 0.001, narray.numpy.shape)
    if "bias" in name:
        narray.numpy[:] = 0.0

req = ['write_to' for i in range(len(arg_narrays))]
# bind executer
# TODO(bing): think of a better bind interface
executor = fc2.bind(mx.Context('cpu'), arg_narrays, grad_narrays, req)
# update

out_narray = executor.heads()[0]
grad_narray = mx.narray.create(out_narray.shape)

epoch = 10
momentum = 0.9
lr = 0.001
wd = 0.0004

def Update(mom, grad, weight):
    weight.numpy[:] -= lr * grad.numpy[:]

block = zip(mom_narrays, grad_narrays, arg_narrays)


train_dataiter = mx.io.MNISTIter(
        image="/home/tianjun/data/mnist/train-images-idx3-ubyte",
        label="/home/tianjun/data/mnist/train-labels-idx1-ubyte",
        batch_size=100, shuffle=1, silent=0, flat=1, seed=1)
val_dataiter = mx.io.MNISTIter(
        image="/home/tianjun/data/mnist/t10k-images-idx3-ubyte",
        label="/home/tianjun/data/mnist/t10k-labels-idx1-ubyte",
        batch_size=100, shuffle=1, silent=0, flat=1)

for i in xrange(epoch):
    # train
    print "Epoch %d" % i
    train_acc = 0.0
    val_acc = 0.0
    train_nbatch = 0
    val_nbatch = 0
    
    for data, label in train_dataiter:
        data = data.numpy
        label = label.numpy.astype(np.int32)
        inputs["data"].numpy[:] = data
        executor.forward()
        out_narray.numpy[:] = Softmax(out_narray.numpy)
        train_acc += CalAcc(out_narray.numpy, label)
        train_nbatch += 1
        grad_narray.numpy[:] = out_narray.numpy
        SetGradient(grad_narray.numpy, label)
        executor.backward([grad_narray])

        for mom, grad, weight in block:
            Update(mom, grad, weight)

    # evaluate
    for data, label in val_dataiter:
        data = data.numpy
        label = label.numpy.astype(np.int32)
        inputs["data"].numpy[:] = data
        executor.forward()
        val_acc += CalAcc(out_narray.numpy, label)
        val_nbatch += 1
    print "Train Acc: ", train_acc / train_nbatch
    print "Valid Acc: ", val_acc / val_nbatch
