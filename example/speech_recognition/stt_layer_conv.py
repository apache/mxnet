import mxnet as mx


def conv(net,
         channels,
         filter_dimension,
         stride,
         weight=None,
         bias=None,
         act_type="relu",
         no_bias=False
         ):
    # 2d convolution's input should have the shape of 4D (batch_size,1,seq_len,feat_dim)
    if weight is None or bias is None:
        # ex) filter_dimension = (41,11) , stride=(2,2)
        net = mx.sym.Convolution(data=net, num_filter=channels, kernel=filter_dimension, stride=stride, no_bias=no_bias)
    elif weight is None or bias is not None:
        net = mx.sym.Convolution(data=net, num_filter=channels, kernel=filter_dimension, stride=stride, bias=bias,
                                 no_bias=no_bias)
    elif weight is not None or bias is None:
        net = mx.sym.Convolution(data=net, num_filter=channels, kernel=filter_dimension, stride=stride, weight=weight,
                                 no_bias=no_bias)
    else:
        net = mx.sym.Convolution(data=net, num_filter=channels, kernel=filter_dimension, stride=stride, weight=weight,
                                 bias=bias, no_bias=no_bias)
    net = mx.sym.Activation(data=net, act_type=act_type)
    return net
