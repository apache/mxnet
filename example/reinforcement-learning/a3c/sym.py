import mxnet as mx

def get_symbol_atari(act_dim):
    net = mx.symbol.Variable('data')
    net = mx.symbol.Cast(data=net, dtype='float32')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu4', act_type="relu")
    fc_policy = mx.symbol.FullyConnected(data=net, name='fc_policy', num_hidden=act_dim)
    policy = mx.symbol.SoftmaxOutput(data=fc_policy, name='policy', out_grad=True)
    entropy = mx.symbol.SoftmaxActivation(data=fc_policy, name='entropy')
    value = mx.symbol.FullyConnected(data=net, name='fc_value', num_hidden=1)
    value = mx.symbol.LinearRegressionOutput(data=value, name='value')
    return mx.symbol.Group([policy, entropy, value])
