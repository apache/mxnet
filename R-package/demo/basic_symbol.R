require(mxnet)

# TODO(KK) all the functions under mx.symbol.fun should be re-exposed
# by taking ... and redirect to list

data = mx.symbol.Variable("data")
d2 = mx.symbol.Variable("d2")
cat(data$debug.str())

group = mx.symbol.fun.Group(list(data, d2))
cat(group$debug.str())
xx = mx.symbol.fun.internal.Plus(list(data, data))
cat(xx$debug.str())

net1 = mx.symbol.fun.FullyConnected(list(data=data, name="fc1", num_hidden=10))
cat(net1$debug.str())
net1 = mx.symbol.fun.FullyConnected(list(data=net1, num_hidden=100))
cat(net1$debug.str())


