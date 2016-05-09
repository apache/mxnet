# pylint: skip-file
import mxnet as mx
import numpy as np
import logging

data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(fc3, name = 'softmax')

n_epoch = 2
batch_size = 100
train_dataiter = mx.io.MNISTIter(
        image="../image-classification/mnist/train-images-idx3-ubyte",
        label="../image-classification/mnist/train-labels-idx1-ubyte",
        data_shape=(784,),
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
val_dataiter = mx.io.MNISTIter(
        image="../image-classification/mnist/t10k-images-idx3-ubyte",
        label="../image-classification/mnist/t10k-labels-idx1-ubyte",
        data_shape=(784,),
        batch_size=batch_size, shuffle=True, flat=True, silent=False)

################################################################################
# Intermediate-level API
################################################################################
mod = mx.mod.Module(softmax)
mod.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label)
mod.init_params()

mod.init_optimizer(optimizer_params={'learning_rate':0.01, 'momentum': 0.9})
metric = mx.metric.create('acc')

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


################################################################################
# High-level API
################################################################################
logging.basicConfig(level=logging.DEBUG)
train_dataiter.reset()
mod = mx.mod.Module(softmax)
mod.fit(train_dataiter, eval_data=val_dataiter,
        optimizer_params={'learning_rate':0.01, 'momentum': 0.9}, num_epoch=n_epoch)

# prediction iterator API
for preds, i_batch, batch in mod.iter_predict(val_dataiter):
    pred_label = preds[0].asnumpy().argmax(axis=1)
    label = batch.label[0].asnumpy().astype('int32')
    if i_batch % 20 == 0:
        print('batch %03d acc: %.3f' % (i_batch, (label == pred_label).sum() / float(len(pred_label))))

# a dummy call just to test if the API works for merge_batches=True
preds = mod.predict(val_dataiter)

# perform prediction and calculate accuracy manually
preds = mod.predict(val_dataiter, merge_batches=False)
val_dataiter.reset()
acc_sum = 0.0; acc_cnt = 0
for i, batch in enumerate(val_dataiter):
    pred_label = preds[i][0].asnumpy().argmax(axis=1)
    label = batch.label[0].asnumpy().astype('int32')
    acc_sum += (label == pred_label).sum()
    acc_cnt += len(pred_label)
print('validation Accuracy: %.3f' % (acc_sum / acc_cnt))

# evaluate on validation set with a evaluation metric
mod.score(val_dataiter, metric)
for name, val in metric.get_name_value():
    print('%s=%f' % (name, val))

