import mxnet as mx
import ctypes

x = mx.sym.Variable('x')
x2 = mx.sym.Activation(x, name='relu', act_type="relu")
x3 = mx.sym.Activation(x2, name='relu2', act_type="relu")

X = mx.nd.ones((2,2))
exe = x3.bind(mx.cpu(0), args={'x':X})
a = None
def p(name, d):
	global a
	print name
	d = mx.nd.NDArray(ctypes.cast(d, mx.base.NDArrayHandle), writable=False)
	a = mx.nd.norm(d)
	#print name
exe.set_monitor_callback(p)
exe.forward()
print 'out', exe.outputs[0].asnumpy()
print a.asscalar()
# dot = mx.viz.plot_network(z)
# dot.format = 'png'
# dot.render('a')

# z._compose(x3)

# dot = mx.viz.plot_network(z)
# dot.format = 'png'
# dot.render('b')