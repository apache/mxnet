# pylint: skip-file
import mxnet as mx
import numpy as np
import os, cPickle, gzip

def CalAcc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label) * 1.0 / out.shape[0]


# load data
class MNISTIter(object):
    def __init__(self, which_set, batch_size=100, flatten=True):
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


train = MNISTIter("train", batch_size, False)
valid = MNISTIter("valid", batch_size, False)

for i in xrange(epoch):
    # train
    print "Epoch %d" % i
    train_acc = 0.0
    val_acc = 0.0
    while train.Next():
        data, label = train.Get()
        print np.shape(data)
        print np.shape(label)
        exit(0)
        inputs["data"].numpy[:] = data
        inputs["sm_label"].numpy[:] = label
        executor.forward()
        train_acc += CalAcc(out_narray.numpy, label)
        grad_narray.numpy[:] = out_narray.numpy
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



