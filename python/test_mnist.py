# pylint: skip-file
import mxnet as mx
import numpy as np
import os, cPickle, gzip

def Softmax(x):
    maxes = np.max(x, axis=1)
    x -= maxes.reshape(maxes.shape[0], 1)
    e = np.exp(x)
    return e / np.sum(e, axis=1)

def CalAcc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label) * 1.0 / out.shape[0]

def SetGradient(out_grad, label):
    assert(out_grad.shape[0] == label.shape[0])
    for i in xrange(label.shape[0]):
        k = label[i]
        out_grad[i][k] -= 1.0

# load data
class MNISTIter(object):
    def __init__(self, which_set, batch_size=100):
        if not os.path.exists('mnist.pkl.gz'):
            os.system("wget http://deeplearning.net/data/mnist/mnist.pkl.gz")
        f = gzip.open('mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
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
        return (self.data[start:end, :], self.label[start:end])



# symbol net
batch_size = 100
data = mx.sym.Variable('data')
fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=160)
act1 = mx.sym.Activation(data = fc1, name='relu1', type="relu")
fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=10)
args_list = fc2.list_arguments()

# infer shape
data_shape = (batch_size, 784)
arg_shapes, out_shapes = fc2.infer_shape(data=data_shape)
arg_narrays = [mx.narray.create(shape) for shape in arg_shapes]
grad_narrays = [mx.narray.create(shape) for shape in arg_shapes]
mom_narrays = [mx.narray.create(shape) for shape in arg_shapes]
out_narray = mx.narray.create(out_shapes[0])
inputs = dict(zip(args_list, arg_narrays))

# set random weight
for name, narray in inputs.items():
    if "weight" in name:
        narray.numpy[:, :] = np.random.uniform(-0.01, 0.01, narray.numpy.shape)


# bind executer
# exec = bind(fc2, args_narray, grad_narray, req)
# update

epoch = 10
momentum = 0.9
lr = 0.01
wd = 0.0004

def Update(mom, grad, weight):
    if len(mom.numpy.shape) == 1:
        mom.numpy[:] = mom.numpy * momentum - lr * (grad.numpy + wd * weight.numpy)
    else:
        mom.numpy[:, :] = mom.numpy * momentum - lr * (grad.numpy + wd * weight.numpy)
    weight += mom

block = zip(mom_narrays, grad_narrays, arg_narrays)


train = MNISTIter("train")
valid = MNISTIter("valid")
for i in xrange(epoch):
    # train
    print "Epoch %d" % i
    train_acc = 0.0
    val_acc = 0.0
    while train.Next():
        data, label = train.Get()
        inputs["data"].numpy[:,:] = data
        # exec.Forward(args_narray)
        train_acc += CalAcc(out_narray.numpy, label)
        SetGradient(out_narray.numpy, label)
        # exec.Backward(out_narray)
        for mom, grad, weight in block:
            Update(mom, grad, weight)
    # evaluate
    while valid.Next():
        data, label = valid.Get()
        inputs["data"].numpy[:,:] = data
        # exec.Forward([ inputs["data"] ])
        val_acc += CalAcc(out_narray.numpy, label)
    print "Train Acc: ", train_acc / train.nbatch
    print "Valid Acc: ", val_acc / valid.nbatch
    train.BeforeFirst()
    valid.BeforeFirst()



