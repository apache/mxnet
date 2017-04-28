import mxnet as mx
import numpy as np
import logging
import os
from sklearn.datasets import fetch_mldata
from mxnet.quantization import *
import mxnet.ndarray as nd

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

INFERENCE = False
no_bias = True
batch_size = 32
name = "conv_mnist"

data = mx.symbol.Variable('data')
conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5),
    num_filter=20, no_bias=True, layout='NHWC')
relu1 = mx.symbol.relu(data=conv1)
flatten = mx.symbol.flatten(data=relu1)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=10)
conv_net = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

# prepare data
mnist = fetch_mldata('MNIST original')
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist.data.shape[0])
X = mnist.data[p].reshape(70000, 28, 28, 1)
pad = np.zeros(shape=(70000, 28, 28, 3))
X = np.concatenate([X, pad], axis=3)
Y = mnist.target[p]

X = X.astype(np.float32)/255
X_train = X[:60000]
X_test = X[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]

train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)
val_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size)

# create a trainable module on GPU 0
model = mx.mod.Module(symbol=conv_net, context=mx.gpu(0))
if not INFERENCE:
# train with the same
    model.fit(train_iter,
                    eval_data=val_iter,
                    optimizer='sgd',
                    optimizer_params={'learning_rate':0.1},
                    eval_metric='acc',
                    batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                    num_epoch=10)
    model.save_checkpoint(name, 10)
else:
    _, arg_params, aux_params = mx.model.load_checkpoint(name, 10)
    model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    model.set_params(arg_params=arg_params, aux_params=aux_params)


test_iter = val_iter
# predict accuracy for conv net
acc = mx.metric.Accuracy()
print('Accuracy: {}%'.format(model.score(test_iter, acc)[0][1]*100))

quantized_conv_net = quantize_graph(conv_net)
print(quantized_conv_net.debug_str())
params = model.get_params()[0]

def test(symbol):
    model = mx.model.FeedForward(
        symbol,
        ctx=mx.gpu(0),
        arg_params=params)
    print 'Accuracy:', model.score(test_iter)*100, '%'

print('origin:')
test(conv_net)
print('after quantization:')
test(quantized_conv_net)
