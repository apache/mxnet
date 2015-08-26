# pylint: skip-file
import mxnet as mx
import numpy as np
import os, pickle, gzip
import sys


def CalAcc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label) * 1.0 / out.shape[0]

def IgnorePython3():
    if sys.version_info[0] >= 3:
        # TODO(tianjun): use IO instead of pickle
        # Python3 pickle is not able to load data correctly
        sys.exit(0)


# load data
class MNISTIter(object):
    def __init__(self, which_set, batch_size=100, flatten=True):
        if not os.path.exists('mnist.pkl.gz'):
            os.system("wget http://deeplearning.net/data/mnist/mnist.pkl.gz")
        f = gzip.open('mnist.pkl.gz', 'rb')
        IgnorePython3()
        train_set, valid_set, test_set = pickle.load(f)
        f.close()
        if which_set == 'train':
            self.data = train_set[0]
            self.label = np.asarray(train_set[1])
        elif which_set == 'valid':
            self.data = valid_set[0]
            self.label = np.asarray(valid_set[1])
        else:
            self.data = test_set[0]
            self.data = np.asarray(test_set[1])
        self.flatten = flatten
        self.batch_size = batch_size
        self.nbatch = self.data.shape[0] / batch_size
        assert(self.data.shape[0] % batch_size == 0) # I am lazy
        self.now_idx = -1
    def BeforeFirst(self):
        self.now_idx = -1
    def Next(self):
        self.now_idx += 1
        if self.now_idx == self.nbatch:
            return False
        return True
    def Get(self):
        if self.now_idx < 0:
            raise Exception("Iterator is at head")
        elif self.now_idx >= self.nbatch:
            raise Exception("Iterator is at end")
        start = self.now_idx * self.batch_size
        end = (self.now_idx + 1) * self.batch_size
        if self.flatten:
            return (self.data[start:end, :], self.label[start:end])
        else:
            return (self.data[start:end, :].reshape(batch_size, 1, 28, 28),
                    self.label[start:end])



# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
conv1= mx.symbol.Convolution(data = data, name='conv1', nb_filter=32, kernel=(3,3), stride=(1,1), nstep=10)
act1 = mx.symbol.Activation(data = conv1, name='relu1', act_type="relu")
mp1 = mx.symbol.Pooling(data = act1, name = 'mp1', kernel=(2,2), stride=(2,2), pool_type='max')

conv2= mx.symbol.Convolution(data = mp1, name='conv2', nb_filter=32, kernel=(3,3), stride=(1,1), nstep=10)
act2 = mx.symbol.Activation(data = conv2, name='relu2', act_type="relu")
mp2 = mx.symbol.Pooling(data = act2, name = 'mp2', kernel=(2,2), stride=(2,2), pool_type='max')


fl = mx.symbol.Flatten(data = mp2, name="flatten")
fc2 = mx.symbol.FullyConnected(data = fl, name='fc2', num_hidden=10)
softmax = mx.symbol.Softmax(data = fc2, name = 'sm')
args_list = softmax.list_arguments()
# infer shape
#data_shape = (batch_size, 784)

data_shape = (batch_size, 1, 28, 28)
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

epoch = 1
momentum = 0.9
lr = 0.1
wd = 0.0004

def Update(grad, weight):
    weight.numpy[:] -= lr * grad.numpy[:] / batch_size

block = zip(grad_narrays, arg_narrays)


train = MNISTIter("train", batch_size, False)
valid = MNISTIter("valid", batch_size, False)

def test_mnist():
    acc_train = 0.0
    acc_val = 0.0
    for i in xrange(epoch):
        # train
        print("Epoch %d" % i)
        train_acc = 0.0
        val_acc = 0.0
        while train.Next():
            data, label = train.Get()
            inputs["data"].numpy[:] = data
            inputs["sm_label"].numpy[:] = label
            executor.forward()
            train_acc += CalAcc(out_narray.numpy, label)
            grad_narray.numpy[:] = out_narray.numpy
            executor.backward([grad_narray])

            for grad, weight in block:
                Update(grad, weight)

        # evaluate
        while valid.Next():
            data, label = valid.Get()
            inputs["data"].numpy[:] = data
            executor.forward()
            val_acc += CalAcc(out_narray.numpy, label)
        print("Train Acc: ", train_acc / train.nbatch)
        print("Valid Acc: ", val_acc / valid.nbatch)
        acc_train = train_acc / train.nbatch
        acc_val = val_acc / valid.nbatch
        train.BeforeFirst()
        valid.BeforeFirst()
    assert(acc_train > 0.84)
    assert(acc_val > 0.96)

