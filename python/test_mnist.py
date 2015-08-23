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
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=160)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=10)
args_list = fc2.list_arguments()
# infer shape
data_shape = (batch_size, 784)
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


train = MNISTIter("train", batch_size)
valid = MNISTIter("valid", batch_size)

for i in xrange(epoch):
    # train
    print "Epoch %d" % i
    train_acc = 0.0
    val_acc = 0.0
    while train.Next():
        data, label = train.Get()
        inputs["data"].numpy[:] = data
        executor.forward()
        out_narray.numpy[:] = Softmax(out_narray.numpy)
        train_acc += CalAcc(out_narray.numpy, label)
        grad_narray.numpy[:] = out_narray.numpy
        SetGradient(grad_narray.numpy, label)
        executor.backward([grad_narray])

        for mom, grad, weight in block:
            Update(mom, grad, weight)

    # evaluate
    while valid.Next():
        data, label = valid.Get()
        inputs["data"].numpy[:] = data
        executor.forward()
        val_acc += CalAcc(out_narray.numpy, label)
    print "Train Acc: ", train_acc / train.nbatch
    print "Valid Acc: ", val_acc / valid.nbatch
    train.BeforeFirst()
    valid.BeforeFirst()



