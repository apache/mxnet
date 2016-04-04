# pylint: skip-file
import mxnet as mx
import numpy as np

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(fc3, name = 'sm')

batch_size = 100
train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        data_shape=(784,),
        label_name='sm_label',
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)

mod = mx.mod.Module(softmax, ['data', 'sm_label'])
mod.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)
mod.init_params()

mod.init_optimizer(optimizer_params={'learning_rate':0.01, 'momentum': 0.9})

metric = mx.metric.create('acc')

n_epoch = 5

for i_epoch in range(n_epoch):
    for i_iter, batch in enumerate(train_dataiter):
        mod.forward(batch)
        mod.update_metric(metric, batch.label)

        mod.backward()
        mod.update()

    for name, val in metric.get_name_value():
        print('epoch %03d: %s=%f' % (i_epoch, name, val))
    metric.reset()
    train_dataiter.reset()
