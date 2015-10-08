require(mxnet)

# TODO(KK) all the functions under mx.varg.* should be re-exposed
# by taking ... and redirect to list
# TODO(KK, tong) expose the internal functions and members as formal R method

data = mx.symbol.Variable("data")
d2 = mx.symbol.Variable("d2")

group = mx.varg.symbol.Group(list(data, d2))
xx = data + data

net1 = mx.varg.symbol.FullyConnected(list(data=data, name="fc1", num_hidden=10))
net1 = mx.varg.symbol.FullyConnected(list(data=net1, num_hidden=100))

json = net1$as.json()
net2 = mx.symbol.load.json(json)
print(net2$arguments())

slist = net1$infer.shape(list(data=c(100, 10)))
print(slist)



