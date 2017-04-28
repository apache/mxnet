import mxnet as mx
import numpy as np
import logging
from sklearn.datasets import fetch_mldata
from mxnet.quantization import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

name = 'mlp_mnist'
no_bias = True
batch_size = 32
INFERENCE = True

data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=32, no_bias=no_bias)
act1 = mx.symbol.relu(data = fc1, name='act1')
fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64, no_bias=no_bias)
act2 = mx.symbol.relu(data = fc2)
fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10, no_bias=no_bias)
mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

print(mlp.list_arguments())


# prepare data
mnist = fetch_mldata('MNIST original')
np.random.seed(1234) # set seed for deterministic ordering
p = np.random.permutation(mnist.data.shape[0])
X = mnist.data[p]
Y = mnist.target[p]

X = X.astype(np.float32)/255
X_train = X[:60000]
X_test = X[60000:]
Y_train = Y[:60000]
Y_test = Y[60000:]

train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)
test_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size)
val_iter = test_iter

model = mx.mod.Module(symbol=mlp, context=mx.gpu(0))
if not INFERENCE:
    model.fit(train_iter,
              eval_data=val_iter,
              optimizer='sgd',
              optimizer_params={'learning_rate':0.1},
              eval_metric='acc',
              batch_end_callback = mx.callback.Speedometer(batch_size, 200),
              num_epoch=10)
    model.save_checkpoint(name, 10)
else:
    _, arg_params, aux_params = mx.model.load_checkpoint(name, 10)
    model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    model.set_params(arg_params=arg_params, aux_params=aux_params)

acc = mx.metric.Accuracy()
print('Accuracy: {}%'.format(model.score(test_iter, acc)[0][1]*100))


quantized_mlp = quantize_graph(mlp)
print(quantized_mlp.debug_str())
params = model.get_params()[0]

def test(symbol):
    model = mx.model.FeedForward(
        symbol,
        ctx=mx.gpu(0),
        arg_params=params)
    print 'Accuracy:', model.score(test_iter)*100, '%'

print('origin:')
test(mlp)
print('after quantization:')
test(quantized_mlp)
