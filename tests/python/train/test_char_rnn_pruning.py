import mxnet as mx
import numpy as np
import random
import bisect

import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

import sys
sys.path.append('char_rnn/')
#from char_rnn import lstm, bucket_io, rnn_model
from lstm import lstm_unroll, lstm_inference_symbol
from bucket_io import BucketSentenceIter
from rnn_model import LSTMInferenceModel

from numpy import count_nonzero as nz

# read from doc
def read_content(path):
    with open(path) as ins:
        content = ins.read()
        return content

# build a vocabulary of what char we have in the content
def build_vocab(path):
    content = read_content(path)
    content = list(content)
    idx = 1 # 0 is left for zero-padding
    the_vocab = {}
    for word in content:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab

# to assign each char with a special numerical id
def text2id(sentence, the_vocab):
    words = list(sentence)
    words = [the_vocab[w] for w in words if len(w) > 0]
    return words

# evaluation 
def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

# get data
import os
logging.info('Getting data...')
data_url = "http://data.mxnet.io/mxnet/data/char_lstm.zip"
os.system("wget %s" % data_url)
os.system("unzip -o char_lstm.zip")

# LSTM hyperparameters
batch_size = 128
buckets = [129]
num_hidden = 512
num_embed = 256
num_lstm_layer = 3
num_epoch = 2
learning_rate = 0.01
momentum = 0.0

devs = [mx.context.gpu(i) for i in range(1)]

# build char vocabluary from input
vocab = build_vocab("./obama.txt")

# generate symbol for a length
def sym_gen(seq_len):
    return lstm_unroll(num_lstm_layer, seq_len, len(vocab) + 1,
                       num_hidden=num_hidden, num_embed=num_embed,
                       num_label=len(vocab) + 1, dropout=0.2)

# initalize states for LSTM
init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h

# build an iterator for text
data_train = BucketSentenceIter("./obama.txt", vocab, buckets, batch_size,
                                init_states, seperate_char='\n',
                                text2id=text2id, read_content=read_content)

# the network symbol
symbol = sym_gen(buckets[0])

# Train a LSTM network as simple as feedforward network
model = mx.model.FeedForward(ctx=devs,
                             symbol=symbol,
                             num_epoch=num_epoch,
                             learning_rate=learning_rate,
                             momentum=momentum,
                             wd=0.0001,
                             initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
                             optimizer = 'sparsesgd',
                             pruning_switch_epoch = [1, 2],
                             weight_sparsity = [25, 50],
                             bias_sparsity = [0, 0],
                             batches_per_epoch = 65,
                            )

# fit model
logging.info('Fitting model...')
model.fit(X=data_train,
          eval_metric = mx.metric.np(Perplexity),
          batch_end_callback=mx.callback.Speedometer(batch_size, 10),
          epoch_end_callback=mx.callback.do_checkpoint("obama"))

# chack pruning
logging.info('Check pruning...')
weight_percent = [0.75, 0.5]
bias_percent = [1.0, 1.0]
for i in range(1, num_epoch + 1):
    sym, arg_params, aux_params = mx.model.load_checkpoint('obama', i)
    for key in arg_params.keys():
        idx = i - 1
        if 'weight' in key:
            assert nz(arg_params[key].asnumpy())/float(arg_params[key].size) == weight_percent[idx]
        else:
            assert nz(arg_params[key].asnumpy())/float(arg_params[key].size) == bias_percent[idx]

for i in range(num_epoch):
    os.remove('obama-000%d.params' % (i + 1))
os.remove('char_lstm.zip')
os.remove('obama-0075.params')
os.remove('obama-symbol.json')
os.remove('obama.txt')
