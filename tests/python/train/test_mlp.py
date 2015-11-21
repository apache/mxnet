# pylint: skip-file
import mxnet as mx
import numpy as np
import os, sys
import pickle as pickle
import logging
from common import get_data

# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(fc3, name = 'sm')

def accuracy(label, pred):
    py = np.argmax(pred, axis=1)
    return np.sum(py == label) / float(label.size)

num_epoch = 4
prefix = './mlp'

#check data
get_data.GetMNIST_ubyte()

train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        data_shape=(784,),
        label_name='sm_label',
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        data_shape=(784,),
        label_name='sm_label',
        batch_size=batch_size, shuffle=True, flat=True, silent=False)

def test_mlp():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    model = mx.model.FeedForward.create(
        softmax,
        X=train_dataiter,
        eval_data=val_dataiter,
        eval_metric=mx.metric.np(accuracy),
        epoch_end_callback=mx.callback.do_checkpoint(prefix),
        ctx=[mx.cpu(i) for i in range(2)],
        num_epoch=num_epoch,
        learning_rate=0.1, wd=0.0004,
        momentum=0.9)

    logging.info('Finish traning...')
    prob = model.predict(val_dataiter)
    logging.info('Finish predict...')
    val_dataiter.reset()
    y = np.concatenate([batch.label[0].asnumpy() for batch in val_dataiter]).astype('int')
    py = np.argmax(prob, axis=1)
    acc1 = float(np.sum(py == y)) / len(y)
    logging.info('final accuracy = %f', acc1)
    assert(acc1 > 0.95)

    # predict internal featuremaps
    internals = softmax.get_internals()
    fc2 = internals['fc2_output']
    mfeat = mx.model.FeedForward(symbol=fc2,
                                 arg_params=model.arg_params,
                                 aux_params=model.aux_params,
                                 allow_extra_params=True)
    feat = mfeat.predict(val_dataiter)
    assert feat.shape == (10000, 64)
    # pickle the model
    smodel = pickle.dumps(model)
    model2 = pickle.loads(smodel)
    prob2 = model2.predict(val_dataiter)
    assert np.sum(np.abs(prob - prob2)) == 0

    # load model from checkpoint
    model3 = mx.model.FeedForward.load(prefix, num_epoch)
    prob3 = model3.predict(val_dataiter)
    assert np.sum(np.abs(prob - prob3)) == 0

    # save model explicitly
    model.save(prefix, 128)
    model4 = mx.model.FeedForward.load(prefix, 128)
    prob4 = model4.predict(val_dataiter)
    assert np.sum(np.abs(prob - prob4)) == 0

    for i in range(num_epoch):
        os.remove('%s-%04d.params' % (prefix, i + 1))
    os.remove('%s-symbol.json' % prefix)
    os.remove('%s-0128.params' % prefix)


if __name__ == "__main__":
    test_mlp()
