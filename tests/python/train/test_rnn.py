# Dataset is from the 'pystruct' package

import logging

import numpy as np
from mxnet.optimizer import RMSProp
from mxnet.metric import EvalMetric
from mxnet import symbol, context, ndarray, io, model, visualization
from mxnet.rnn import SimpleRecurrece, Sequencer

class DummyBatch(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.sequence_length = len(data)

    def data_at(self, t):
        return self.data[t]
    def label_at(self, t):
        return self.label[t]

from pystruct.datasets import load_letters
letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)
label_empty = 26
seq_len = 8
for i in range(len(X)):
    if X[i].shape[0] >= seq_len:
        X[i] = X[i][:seq_len]
        y[i] = y[i][:seq_len]
    else:
        add_len = seq_len-X[i].shape[0]
        X[i] = np.vstack([X[i], np.zeros((add_len, X[i].shape[1]))])
        y[i] = np.hstack([y[i], np.zeros(add_len, int)+label_empty])

X_train, X_test = X[folds != 1], X[folds == 1]
y_train, y_test = y[folds != 1], y[folds == 1]

class LettersData(object):
    def __init__(self, X, Y, batch_size=16, n_batch=10):
        self.batch_size = batch_size
        self.n_batch = n_batch
        self.X = X
        self.Y = Y
        self.provide_data = [('data', (batch_size, X[0].shape[1]))]
        self.provide_label = [('softmax_label', (batch_size,))]

    def reset(self):
        pass

    def __iter__(self):
        for i in range(0, self.n_batch*self.batch_size, self.batch_size):
            data = self.X[i:i+self.batch_size]
            data = [[ndarray.array(np.vstack([x[t] for x in data]))] for t in range(seq_len)]
            label = self.Y[i:i+self.batch_size]
            label = [[ndarray.array(np.hstack([y[t] for y in label]))] for t in range(seq_len)]

            yield DummyBatch(data, label)


#class DummyData(object):
#    def __init__(self, seq_len=5, batch_size=1, n_batch=10):
#        self.sequence_length = seq_len
#        self.batch_size = batch_size
#        self.n_batch = n_batch
#        self.provide_data = [('data', (batch_size, 1))]
#        self.provide_label = [('softmax_label', (batch_size,))]
#
#    def reset(self):
#        pass
#
#    def __iter__(self):
#        for i in range(self.n_batch):
#            data = []
#            label = []
#            for t in range(self.sequence_length):
#                x = ndarray.array(np.zeros((self.batch_size, 1))+t)
#                y = ndarray.array(np.zeros((self.batch_size, ))+t+1)
#                data.append([x])
#                label.append([y])
#            yield DummyBatch(data, label)

class SeqAccuracy(EvalMetric):
    """Calculate accuracy"""

    def __init__(self):
        super(SeqAccuracy, self).__init__('accuracy')

    def begin_sequence(self):
        pass

    def end_sequence(self):
        pass

    def update_at(self, labels, preds, t):
        for i in range(len(labels)):
            pred = preds[i].asnumpy()
            label = labels[i].asnumpy().astype('int32')
            pred_label = np.argmax(pred, axis=1)

            self.sum_metric += (pred_label == label).sum()
            self.num_inst += pred_label.shape[0]

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

batch_size = 16
num_epoch = 10
n_batch = 20
data = symbol.Variable('data')
rec, rules = SimpleRecurrece(data=data, name='rec', num_hidden=16)
pred = symbol.FullyConnected(data=rec, name='pred', num_hidden=27)
loss = symbol.SoftmaxOutput(data=pred, name='softmax')

seq = Sequencer(loss, rules, context.cpu())
data = LettersData(n_batch=n_batch, batch_size=batch_size, X=X_train, Y=y_train)
eval_data = LettersData(n_batch=n_batch, batch_size=batch_size, X=X_test, Y=y_test)
eval_metric = SeqAccuracy()
optimizer = RMSProp()

#visualization.plot_network(loss).render('net-orig')
#visualization.plot_network(seq.sym).render('net-seq')

seq.fit(data=data, eval_data=eval_data, optimizer=optimizer,
        eval_metric=eval_metric, end_epoch=num_epoch)


# ordinary mlp, without recurrence
data = symbol.Variable('data')
hid = symbol.FullyConnected(data=data, name='rec', num_hidden=16)
pred = symbol.FullyConnected(data=hid, name='pred', num_hidden=27)
loss = symbol.SoftmaxOutput(data=pred, name='softmax')

X_train = np.vstack(X_train)[:batch_size*n_batch]
y_train = np.hstack(y_train)[:batch_size*n_batch]
X_test = np.vstack(X_test)[:batch_size*n_batch]
y_test = np.hstack(y_train)[:batch_size*n_batch]
data = io.NDArrayIter(data=X_train, label=y_train, batch_size=batch_size)
eval_data = io.NDArrayIter(data=X_test, label=y_test, batch_size=batch_size)
mlp = model.FeedForward(loss, ctx=context.cpu(), optimizer=optimizer, num_epoch=num_epoch)

mlp.fit(data, eval_data=eval_data)