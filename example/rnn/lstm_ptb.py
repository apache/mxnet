# pylint:skip-file
import lstm
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np

"""
PennTreeBank Language Model
We would like to thanks Wojciech Zaremba for his Torch LSTM code

The data file can be found at:
https://github.com/dmlc/web-data/tree/master/mxnet/ptb
"""

def load_data(path, dic=None):
    fi = open(path)
    content = fi.read()
    content = content.replace('\n', '<eos>')
    content = content.split(' ')
    print("Loading %s, size of data = %d" % (path, len(content)))
    x = np.zeros(len(content))
    if dic == None:
        dic = {}
    idx = 0
    for i in range(len(content)):
        word = content[i]
        if len(word) == 0:
            continue
        if not word in dic:
            dic[word] = idx
            idx += 1
        x[i] = dic[word]
    print("Unique token: %d" % len(dic))
    return x, dic

def drop_tail(X, seq_len):
    shape = X.shape
    nstep = int(shape[0] / seq_len)
    return X[0:(nstep * seq_len), :]


def replicate_data(x, batch_size):
    nbatch = int(x.shape[0] / batch_size)
    x_cut = x[:nbatch * batch_size]
    data = x_cut.reshape((nbatch, batch_size), order='F')
    return data

batch_size = 20
seq_len = 35
num_hidden = 200
num_embed = 200
num_lstm_layer = 2
num_round = 25
learning_rate= 0.1
wd=0.
momentum=0.0
max_grad_norm = 5.0
update_period = 1

X_train, dic = load_data("./data/ptb.train.txt")
X_val, _ = load_data("./data/ptb.valid.txt", dic)
X_train_batch = replicate_data(X_train, batch_size)
X_val_batch = replicate_data(X_val, batch_size)
vocab = len(dic)
print("Vocab=%d" %vocab)

X_train_batch = drop_tail(X_train_batch, seq_len)
X_val_batch = drop_tail(X_val_batch, seq_len)


model = lstm.setup_rnn_model(mx.cpu(),
                             num_lstm_layer=num_lstm_layer,
                             seq_len=seq_len,
                             num_hidden=num_hidden,
                             num_embed=num_embed,
                             num_label=vocab,
                             batch_size=batch_size,
                             input_size=vocab,
                             initializer=mx.initializer.Uniform(0.1),dropout=0.5)

lstm.train_lstm(model, X_train_batch, X_val_batch,
                num_round=num_round,
                half_life=2,
                max_grad_norm = max_grad_norm,
                update_period=update_period,
                learning_rate=learning_rate,
                wd=wd)
#               momentum=momentum)

