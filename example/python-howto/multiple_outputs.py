"""Create a Multiple output configuration.

This example shows how to create a multiple output configuration.
"""
import mxnet as mx

net = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=64)
out = mx.symbol.SoftmaxOutput(data=net, name='softmax')
# group fc1 and out together
group = mx.symbol.Group([fc1, out])
print group.list_outputs()

# You can go ahead and bind on the group
# executor = group.simple_bind(data=data_shape)
# executor.forward()
# executor.output[0] will be value of fc1
# executor.output[1] will be value of softmax
