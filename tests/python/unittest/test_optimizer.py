import numpy as np
import mxnet as mx



def test_lr_wd_mult():
	data = mx.sym.Variable('data')
	bias = mx.sym.Variable('fc1_bias', lr_mult=1.0)
	fc1 = mx.sym.FullyConnected(data=data, bias=bias, name='fc1', num_hidden=10, lr_mult=0)
	fc2 = mx.sym.FullyConnected(data=fc1, name='fc2', num_hidden=10, wd_mult=0.5)

	mod = mx.mod.Module(symbol=fc2, label_names=None)
	mod.bind(data_shapes=[('data', (5,10))])
	mod.init_params(initializer=mx.init.Uniform(1.0))
	mod.init_optimizer(optimizer_params={'learning_rate': 1.0})
	args1, _ = mod.get_params()
	args1 = {k: v.asnumpy() for k, v in args1.items()}
	mod.forward(mx.io.DataBatch(data=[mx.random.uniform(low=-1.0, high=1.0, shape=(5,10))], label=None), is_train=True)
	mod.backward(mod.get_outputs())
	mod.update()
	args2, _ = mod.get_params()
	args2 = {k: v.asnumpy() for k, v in args2.items()}

	assert mod._optimizer.lr_mult == {'fc1_bias': 1.0, 'fc1_weight': 0.0}
	assert mod._optimizer.wd_mult == {'fc2_bias': 0.5, 'fc2_weight': 0.5, 'fc1_bias': 0.0}
	assert mx.test_utils.almost_equal(args1['fc1_weight'], args2['fc1_weight'], 1e-10)
	assert not mx.test_utils.almost_equal(args1['fc1_bias'], args2['fc1_bias'], 1e-1)
	assert not mx.test_utils.almost_equal(args1['fc2_weight'], args2['fc2_weight'], 1e-1)



if __name__ == '__main__':
	test_lr_wd_mult()