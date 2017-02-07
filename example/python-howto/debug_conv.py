import mxnet as mx

data_shape = (1,3,5,5)
class SimpleData(object):

    def __init__(self, data):
        self.data = data

data = mx.sym.Variable('data')
conv = mx.sym.Convolution(data=data, kernel=(3,3), pad=(1,1), stride=(1,1), num_filter=1)
mon = mx.mon.Monitor(1)


mod = mx.mod.Module(conv)
mod.bind(data_shapes=[('data', data_shape)])
mod._exec_group.install_monitor(mon)
mod.init_params()

input_data = mx.nd.ones(data_shape)
mod.forward(data_batch=SimpleData([input_data]))
res = mod.get_outputs()[0].asnumpy()
print(res)