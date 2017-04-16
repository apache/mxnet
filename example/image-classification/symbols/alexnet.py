"""
Reference:

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
"""
import mxnet as mx

def get_symbol(num_classes, **kwargs):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(name='conv1',
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    lrn1 = mx.symbol.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool1 = mx.symbol.Pooling(
        data=lrn1, pool_type="max", kernel=(3, 3), stride=(2,2))
    # stage 2
    conv2 = mx.symbol.Convolution(name='conv2',
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    lrn2 = mx.symbol.LRN(data=relu2, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool2 = mx.symbol.Pooling(data=lrn2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 3
    conv3 = mx.symbol.Convolution(name='conv3',
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(name='conv4',
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(name='conv5',
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(name='fc1', data=flatten, num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    # stage 5
    fc2 = mx.symbol.FullyConnected(name='fc2', data=dropout1, num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
    # stage 6
    fc3 = mx.symbol.FullyConnected(name='fc3', data=dropout2, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax
