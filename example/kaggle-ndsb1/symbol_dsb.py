import find_mxnet
import mxnet as mx

def get_symbol(num_classes = 121):
    net = mx.sym.Variable("data")
    net = mx.sym.Convolution(data=net, kernel=(5, 5), num_filter=32, pad=(2, 2))
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Convolution(data=net, kernel=(5, 5), num_filter=64, pad=(2, 2))
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Pooling(data=net, pool_type="max", kernel=(3, 3), stride=(2, 2))
    # stage 2
    net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=64, pad=(1, 1))
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=64, pad=(1, 1))
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=128, pad=(1, 1))
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Pooling(data=net, pool_type="max", kernel=(3, 3), stride=(2, 2))
    # stage 3
    net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=256, pad=(1, 1))
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), num_filter=256, pad=(1, 1))
    net = mx.sym.Activation(data=net, act_type="relu")
    net = mx.sym.Pooling(data=net, pool_type="avg", kernel=(9, 9), stride=(1, 1))
    # stage 4
    net = mx.sym.Flatten(data=net)
    net = mx.sym.Dropout(data=net, p=0.25)
    net = mx.sym.FullyConnected(data=net, num_hidden=121)
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')

    return net

