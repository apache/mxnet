# pylint: skip-file
import mxnet as mx
import numpy as np
import os, cPickle, gzip

def CalAcc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label.transpose()) * 1.0 / out.shape[0]

# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.Convolution(data = data, name='conv1', nb_filter=32, kernel=(7,7), stride=(2,2), nstep=10, no_bias=1)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
mp = mx.symbol.Pooling(data = act1, name = 'mp', kernel=(2,2), stride=(2,2), pool_type='avg')
fl = mx.symbol.Flatten(data = mp, name="flatten")
fc2 = mx.symbol.FullyConnected(data = fl, name='fc2', num_hidden=10)
softmax = mx.symbol.Softmax(data = fc2, name = 'sm')
args_list = softmax.list_arguments()
# infer shape
#data_shape = (batch_size, 784)

data_shape = (batch_size, 1, 28, 28)
arg_shapes, out_shapes = softmax.infer_shape(data=data_shape)
arg_narrays = [mx.narray.create(shape) for shape in arg_shapes]
grad_narrays = [mx.narray.create(shape) for shape in arg_shapes]
mom_narrays = [mx.narray.create(shape) for shape in arg_shapes]
inputs = dict(zip(args_list, arg_narrays))
print zip(args_list, arg_shapes)
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
executor = softmax.bind(mx.Context('cpu'), arg_narrays, grad_narrays, req)
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


train_dataiter = mx.io.MNISTIterator(path_img="/home/tianjun/data/mnist/train-images-idx3-ubyte",
        path_label="/home/tianjun/data/mnist/train-labels-idx1-ubyte",
        batch_size=100, shuffle=1, silent=1, input_flat="flat")
train_dataiter.beforefirst()
val_dataiter = mx.io.MNISTIterator(path_img="/home/tianjun/data/mnist/t10k-images-idx3-ubyte",
        path_label="/home/tianjun/data/mnist/t10k-labels-idx1-ubyte",
        batch_size=100, shuffle=1, silent=1, input_flat="flat")
val_dataiter.beforefirst()

for i in xrange(epoch):
    # train
    print "Epoch %d" % i
    train_acc = 0.0
    val_acc = 0.0
    train_nbatch = 0
    val_nbatch = 0
    while train_dataiter.next():
        data = train_dataiter.getdata()
        label = train_dataiter.getlabel().numpy.astype(np.int32)
        inputs["data"].numpy[:] = data.numpy
        executor.forward()
        train_acc += CalAcc(out_narray.numpy, label)
        train_nbatch += 1
        grad_narray.numpy[:] = out_narray.numpy
        executor.backward([grad_narray])

        for mom, grad, weight in block:
            Update(mom, grad, weight)

    # evaluate
    while val_dataiter.next():
        data = val_dataiter.getdata()
        label = val_dataiter.getlabel().numpy.astype(np.int32)
        inputs["data"].numpy[:] = data.numpy
        executor.forward()
        val_acc += CalAcc(out_narray.numpy, label)
        val_nbatch += 1
    print "Train Acc: ", train_acc / train_nbatch
    print "Valid Acc: ", val_acc / val_nbatch
    train_dataiter.beforefirst()
    val_dataiter.beforefirst()



