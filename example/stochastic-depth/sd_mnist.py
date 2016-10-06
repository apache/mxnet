################################################################################
# A sanity check mainly for debugging purpose. See sd_cifar10.py for a non-trivial
# example of stochastic depth on cifar10.
################################################################################

import os
import sys
import mxnet as mx
import logging

import sd_module

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "image-classification")))
from train_mnist import get_iterator
from symbol_resnet import get_conv

death_rates = [0.3]
contexts = [mx.context.cpu()]

data = mx.symbol.Variable('data')
conv = get_conv(
    name='conv0',
    data=data,
    num_filter=16,
    kernel=(3, 3),
    stride=(1, 1),
    pad=(1, 1),
    with_relu=True,
    bn_momentum=0.9
)

base_mod = mx.mod.Module(conv, label_names=None, context=contexts)
mod_seq = mx.mod.SequentialModule()
mod_seq.add(base_mod)

for i in range(len(death_rates)):
    conv = get_conv(
        name='conv0_%d' % i,
        data=mx.sym.Variable('data_%d' % i),
        num_filter=16,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        with_relu=True,
        bn_momentum=0.9
    )
    conv = get_conv(
        name='conv1_%d' % i,
        data=conv,
        num_filter=16,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        with_relu=False,
        bn_momentum=0.9
    )
    mod = sd_module.StochasticDepthModule(conv, data_names=['data_%d' % i],
                                          context=contexts, death_rate=death_rates[i])
    mod_seq.add(mod, auto_wiring=True)

act = mx.sym.Activation(mx.sym.Variable('data_final'), act_type='relu')
flat = mx.sym.Flatten(act)
pred = mx.sym.FullyConnected(flat, num_hidden=10)
softmax = mx.sym.SoftmaxOutput(pred, name='softmax')
mod_seq.add(mx.mod.Module(softmax, context=contexts, data_names=['data_final']),
            auto_wiring=True, take_labels=True)


n_epoch = 2
batch_size = 100


train = mx.io.MNISTIter(
        image="../image-classification/mnist/train-images-idx3-ubyte",
        label="../image-classification/mnist/train-labels-idx1-ubyte",
        input_shape=(1, 28, 28), flat=False,
        batch_size=batch_size, shuffle=True, silent=False, seed=10)
val = mx.io.MNISTIter(
        image="../image-classification/mnist/t10k-images-idx3-ubyte",
        label="../image-classification/mnist/t10k-labels-idx1-ubyte",
        input_shape=(1, 28, 28), flat=False,
        batch_size=batch_size, shuffle=True, silent=False)

logging.basicConfig(level=logging.DEBUG)
mod_seq.fit(train, val, optimizer_params={'learning_rate': 0.01, 'momentum': 0.9},
            num_epoch=n_epoch, batch_end_callback=mx.callback.Speedometer(batch_size, 10))
