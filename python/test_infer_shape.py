# pylint: skip-file
import mxnet as mx

data = mx.sym.Variable('data')

fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=1000)
fc2 = mx.sym.FullyConnected(data=fc1, name='fc2', num_hidden=10)
fc3 = mx.sym.FullyConnected( name='fc2', num_hidden=10)

print fc2.list_arguments()

data_shape = (100, 100)
arg_shapes, out_shapes = fc2.infer_shape(data=data_shape)
print dict(zip(fc2.list_arguments(), arg_shapes))
print dict(zip(fc2.list_returns(), out_shapes))

weight_shape= (1, 100)
data_shape = (100, 100)
arg_shapes, out_shapes = fc2.infer_shape(data=data_shape, fc1_weight=weight_shape)
