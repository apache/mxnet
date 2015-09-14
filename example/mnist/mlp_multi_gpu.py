# pylint: skip-file
import mxnet as mx
import numpy as np
import os, gzip
import sys
sys.path.append("../../tests/python")
import get_data
import time

# use multiple devices
num_devs = 4
devs = [mx.Context('gpu', i) for i in range(num_devs)]
mx.kvstore.start()

# symbol net
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
mlp = mx.symbol.Softmax(data = fc3, name = 'mlp')

# define model updater
lr = .1
def updater(key, grad, weight):
    weight -= lr * grad / batch_size

mx.kvstore.set_updater(updater)

# find the params needed to be synchronized between devices
param_names = mlp.list_arguments()
sync_indices = [index for index, name in enumerate(param_names)
                if "weight" in name or "bias" in name]

# infer shape
batch_size = 100
batch_size -= (batch_size % num_devs)
input_shape = (batch_size / num_devs, 784)
param_shapes, out_shapes, aux_shapes  = mlp.infer_shape(data=input_shape)

# init param in the kvstore
np.random.seed(0)
for idx in sync_indices:
    shape = param_shapes[idx]
    val = mx.nd.zeros(shape)
    if "weight" in param_names[idx]:
        val[:] = np.random.uniform(-0.07, 0.07, shape)
    mx.kvstore.init(idx, val)

# allocate device's memory
params = [[mx.nd.zeros(s, d) for s in param_shapes] for d in devs]
grads = [[mx.nd.zeros(s, d) for s in param_shapes] for d in devs]

# create executors for devices
executors = [mlp.bind(devs[d], params[d], grads[d]) for d in range(num_devs)]

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

def run_sgd():
    k = batch_size / num_devs
    batch_splits = [range(d*k, (d+1)*k) for d in range(num_devs)]

    num_epochs = 9
    for epoch in range(num_epochs):
        start = time.time()
        print "Epoch %d" % epoch
        train_count = 0.0
        train_acc = 0.0
        val_count = 0.0
        val_acc = 0.0
        # train
        for data, label in train_dataiter:
            # pull weight
            for idx in sync_indices:
                mx.kvstore.pull(idx, out = [p[idx] for p in params])

            # forward and backward
            data = data.asnumpy()
            label = label.asnumpy().flatten()
            for d in range(num_devs):
                rows = batch_splits[d]
                params[d][param_names.index('data')][:] = data[rows,:]
                params[d][param_names.index('mlp_label')][:] = label[rows]

                executors[d].forward()
                executors[d].backward()
            # push gradient
            for idx in sync_indices:
                mx.kvstore.push(idx, [g[idx] for g in grads])

            # eval
            for d in range(num_devs):
                train_acc += cal_acc(executors[d].outputs[0].asnumpy(),
                                     label[batch_splits[d]])
                train_count += 1

        # validation
        for data, label in val_dataiter:

            # forward
            data = data.asnumpy()
            label = label.asnumpy().flatten()
            for d in range(num_devs):
                rows = batch_splits[d]
                params[d][param_names.index('data')][:] = data[rows,:]
                executors[d].forward()

            # eval
            for d in range(num_devs):
                val_acc += cal_acc(executors[d].outputs[0].asnumpy(),
                                   label[batch_splits[d]])
                val_count += 1

        print("Train Acc: %g, Valid Acc: %g, time: %g" % (
            train_acc / train_count,
            val_acc / val_count,
            time.time() - start))
        train_dataiter.reset()
        val_dataiter.reset()

if __name__ == "__main__":
    run_sgd()
