#!/usr/bin/env python

import sys
logs = sys.stderr

import mxnet as mx
import data_helper
import lstm
import time

step_size = 10
context_size = 5
batch_size = 64
num_hidden = 150
num_embed = 100

print >> logs, 'context size = %d' % context_size
print >> logs, 'batch size = %d' % batch_size
print >> logs, 'step size = %d' % step_size

train_path, dev_path = 'train.conll', 'test.conll'
x, y, vocab = data_helper.load_data(train_path)
x_dev, y_dev, _ = data_helper.load_data(dev_path, vocab, False)
# save vocabulary
vocab_file = open('vocab_map', 'w')
for k, v in vocab.items():
    print >> vocab_file, '%s%s' % (k, v)
vocab_file.close()

print >> logs, 'vocabulary size=%d' % len(vocab)
num_label = len(data_helper.LabelVocab)
print >> logs, 'output labels = %d' % num_label


X_data, y_data = data_helper.reshape_data(x, y, vocab, context_size, step_size)
X_dev_data, y_dev_data = data_helper.reshape_data(x_dev, y_dev, vocab, context_size, step_size)
print >> logs, 'training data shape %s' % str(X_data.shape)

num_epoch = 100

lstm_model = lstm.setup_lstm_model(ctx=mx.gpu(1), num_lstm_layer=1,
                                   step_size=step_size,
                                   context_size=context_size,
                                   num_hidden=num_hidden,
                                   num_embed=num_embed,
                                   num_label=num_label,
                                   batch_size=batch_size,
                                   vocab_size=len(vocab),
                                   initializer=mx.initializer.Uniform(0.1),
                                   dropout=0.5)

# default optimizer is RMSProp, you can choose SGD with learning_rate=0.1
lstm.train_lstm(lstm_model, X_data, y_data, X_dev_data, y_dev_data, num_epoch=num_epoch,
		optimizer='rmsprop', learning_rate=0.001)
