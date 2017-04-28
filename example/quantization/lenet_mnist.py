import mxnet as mx
import numpy as np
import logging
import os
from sklearn.datasets import fetch_mldata
from mxnet.quantization import *
import mxnet.ndarray as nd

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

no_bias = True
batch_size = 32
name = 'lenet_mnist'

data = mx.symbol.Variable('data')
conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=20, no_bias=True)
relu1 = mx.symbol.relu(data=conv1)
pool1 = mx.symbol.max_pool(data=relu1, kernel=(2, 2), stride=(2, 2))

conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=48, no_bias=True)
relu2 = mx.symbol.relu(data=conv2)
pool2 = mx.symbol.max_pool(data=relu2, kernel=(2, 2), stride=(2, 2))

flatten = mx.symbol.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500, no_bias=True)
relu3 = mx.symbol.relu(data=fc1)

fc2 = mx.symbol.FullyConnected(data=relu3, num_hidden=10, no_bias=True)
lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')


# prepare data
mnist = fetch_mldata('MNIST original')
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist.data.shape[0])
X = mnist.data[p].reshape(70000, 1, 28, 28)
pad = np.zeros(shape=(70000, 3, 28, 28))
X = np.concatenate([X, pad], axis=1)
Y = mnist.target[p]

X = X.astype(np.float32)/255
X_train = X[:60000]
X_test = X[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]

train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)
val_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size)

# create a trainable module on GPU 0
lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu(0))
# train with the same
# lenet_model.fit(train_iter,
#                 eval_data=val_iter,
#                 optimizer='sgd',
#                 optimizer_params={'learning_rate':0.1},
#                 eval_metric='acc',
#                 batch_end_callback = mx.callback.Speedometer(batch_size, 100),
#                 num_epoch=10)
# lenet_model.save_checkpoint(name, 10)
sym, arg_params, aux_params = mx.model.load_checkpoint(name, 10)
lenet_model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
lenet_model.set_params(arg_params=arg_params, aux_params=aux_params)


test_iter = val_iter
# predict accuracy for lenet
acc = mx.metric.Accuracy()
print('Accuracy: {}%'.format(lenet_model.score(test_iter, acc)[0][1]*100))

quantized_lenet = quantize_graph(lenet)
print(quantized_lenet.debug_str())
params = lenet_model.get_params()[0]

def test(symbol):
    model = mx.model.FeedForward(
        symbol,
        ctx=mx.gpu(0),
        arg_params=params)
    print 'Accuracy:', model.score(test_iter)*100, '%'

print('origin:')
test(lenet)
print('after quantization:')
test(quantized_lenet)
