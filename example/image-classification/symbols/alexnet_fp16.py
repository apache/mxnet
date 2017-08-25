"""
Reference:

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
"""
import mxnet as mx
import numpy as np

def get_symbol(num_classes, **kwargs):
    input_data = mx.symbol.Variable(name="data")
    input_data = mx.symbol.Cast(data=input_data, dtype=np.float16)
    # stage 1
    weight = mx.symbol.Variable(name='conv1_weight', dtype=np.float16)
    bias = mx.symbol.Variable(name='conv1_bias', dtype=np.float16)
    conv1 = mx.symbol.Convolution(name='conv1',
        data=input_data, weight=weight, bias=bias, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    lrn1 = mx.symbol.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool1 = mx.symbol.Pooling(
        data=lrn1, pool_type="max", kernel=(3, 3), stride=(2,2))
    # stage 2
    weight = mx.symbol.Variable(name='conv2_weight', dtype=np.float16)
    bias = mx.symbol.Variable(name='conv2_bias', dtype=np.float16)
    conv2 = mx.symbol.Convolution(name='conv2',
        data=pool1, weight=weight, bias=bias, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    lrn2 = mx.symbol.LRN(data=relu2, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool2 = mx.symbol.Pooling(data=lrn2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 3
    weight = mx.symbol.Variable(name='conv3_weight', dtype=np.float16)
    bias = mx.symbol.Variable(name='conv3_bias', dtype=np.float16)
    conv3 = mx.symbol.Convolution(name='conv3',
        data=pool2, weight=weight, bias=bias, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    weight = mx.symbol.Variable(name='conv4_weight', dtype=np.float16)
    bias = mx.symbol.Variable(name='conv4_bias', dtype=np.float16)
    conv4 = mx.symbol.Convolution(name='conv4',
        data=relu3, weight=weight, bias=bias, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    weight = mx.symbol.Variable(name='conv5_weight', dtype=np.float16)
    bias = mx.symbol.Variable(name='conv5_bias', dtype=np.float16)
    conv5 = mx.symbol.Convolution(name='conv5',
        data=relu4, weight=weight, bias=bias, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    weight = mx.symbol.Variable(name='fc1_weight', dtype=np.float16)
    bias = mx.symbol.Variable(name='fc1_bias', dtype=np.float16)
    fc1 = mx.symbol.FullyConnected(name='fc1', data=flatten, weight=weight, bias=bias,
        num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    # stage 5
    weight = mx.symbol.Variable(name='fc2_weight', dtype=np.float16)
    bias = mx.symbol.Variable(name='fc2_bias', dtype=np.float16)
    fc2 = mx.symbol.FullyConnected(name='fc2', data=dropout1, weight=weight, bias=bias,
        num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
    # stage 6
    weight = mx.symbol.Variable(name='fc3_weight', dtype=np.float16)
    bias = mx.symbol.Variable(name='fc3_bias', dtype=np.float16)
    fc3 = mx.symbol.FullyConnected(name='fc3', data=dropout2, weight=weight, bias=bias,
        num_hidden=num_classes)
    label = mx.symbol.Variable(name='softmax_label')
    label = mx.symbol.Cast(data=label, dtype=np.float16)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax', label=label)
    return softmax
