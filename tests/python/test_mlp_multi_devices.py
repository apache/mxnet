# pylint: skip-file
import sys
sys.path.append('../../python/')

import mxnet as mx
import numpy as np
import os, gzip
import pickle as pickle
import get_data

# symbol net
data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', nb_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', nb_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', nb_hidden=10)
mlp  = mx.symbol.Softmax(data = fc3, name = 'mlp')

# use multiple devices
num_devs = 2
devs = [mx.Context('cpu', i) for i in range(num_devs)]

# infer shape
batch_size = 100
input_shape = (batch_size / num_devs, 784)
param_shapes, out_shapes, aux_shapes  = mlp.infer_shape(data=input_shape)
param_names = mlp.list_arguments()

# allocate memory
params = [[mx.narray.create(s, d) for s in param_shapes] for d in devs];
grads = [[mx.narray.create(s, d) for s in param_shapes] for d in devs];

# only need to init param on device 0
mx.kvstore.init(devs)

np.random.seed(0)
for i, v in enumerate(params[0]):
    if "weight" in param_names[i]:
        v.numpy[:, :] = np.random.uniform(-0.07, 0.07, v.numpy.shape)
        mx.kvstore.insert(i, v)
    if "bias" in param_names[i]:
        v.numpy[:] = 0.0
        mx.kvstore.insert(i, v)

# register param updater
def make_updater(env):
    def updater(grad, weight):
        eta = env['lr'] / sqrt(env['iter']) / env['batch_size']
        env['iter'] += 1
        weight[:] -= eta * grad
    return updater

mx.kvstore.register(make_updater(
    {'lr' : 0.1, 'batch_size' : batch_size, 'wd' : .00004}))

# create exector for each device

req         = ['write_to' for i in range(len(param_names))]
executors   = [mlp.bind(devs[i], params[i], grads[i], req) for i in range(num_devs)]
forward_out = [mx.narray.create(e.heads()[0].shape) for e in executors]

# data reader
get_data.GetMNIST_ubyte()
train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        batch_size=batch_size, shuffle=True, flat=True, silent=False)

def cal_acc(out, label):
    pred = np.argmax(out, axis=1)
    return np.sum(pred == label) * 1.0 / out.shape[0]

def test_mlp():
    epoch = 9
    acc_train = 0.
    acc_val = 0.
    for i in range(epoch):
        # train
        print("Epoch %d" % i)
        train_acc = 0.0
        for data, label in train_dataiter:
            data = data.numpy
            label = label.numpy.flatten()

            for d in range(num_devs):
                # feed input
                k = batch_size / num_devs
                idx = range(d*k, (d+1)*k)
                params[d][param_names.index('data')].numpy[:] = data[idx,:]
                params[d][param_names.index('mlp_label')].numpy[:] = label[idx]

                # pull weight
                for j, m in enumerate(param_names):
                    if 'weight' in m or 'bias' in m:
                        mx.kvstore.pull(i, params[d][j])

                # forward and backward
                executors[d].forward()
                # TODO copyto should not block execution?
                executors[d].heads()[0].copyto(forward_out[d])
                executors[d].backward([forward_out[d]])

                # push gradient
                for j, m in enumerate(param_names):
                    if 'weight' in m or 'bias' in m:
                        mx.kvstore.pull(i, grads[d][j])

            # TODO should evalute accuray here? otherwise the above forloop will not
            # be paralleled?

        train_acc /= train_nbatch
        train_nbatch += 1
        print("Train Acc: ", train_acc)
        train_dataiter.reset()

    assert(acc_train > 0.98)

if __name__ == "__main__":
    test_mlp()
