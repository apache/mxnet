# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint:skip-file
import lstm
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
# reuse the bucket_io library
sys.path.insert(0, "../../rnn/old")
from bucket_io import BucketSentenceIter, default_build_vocab

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
    if dic is None:
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
num_hidden = 400
num_embed = 200
num_lstm_layer = 8
num_round = 25
learning_rate= 0.1
wd=0.
momentum=0.0
max_grad_norm = 5.0
update_period = 1

dic = default_build_vocab("./data/ptb.train.txt")
vocab = len(dic)

# static buckets
buckets = [8, 16, 24, 32, 60]

init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h

X_train_batch = BucketSentenceIter("./data/ptb.train.txt", dic,
                                        buckets, batch_size, init_states, model_parallel=True)
X_val_batch = BucketSentenceIter("./data/ptb.valid.txt", dic,
                                      buckets, batch_size, init_states, model_parallel=True)

ngpu = 2
# A simple two GPU placement plan
group2ctx = {'embed': mx.gpu(0),
             'decode': mx.gpu(ngpu - 1)}

for i in range(num_lstm_layer):
    group2ctx['layer%d' % i] = mx.gpu(i * ngpu // num_lstm_layer)

# whether do group-wise concat
concat_decode = False
use_loss=True
model = lstm.setup_rnn_model(mx.gpu(), group2ctx=group2ctx,
                             concat_decode=concat_decode,
                             use_loss=use_loss,
                             num_lstm_layer=num_lstm_layer,
                             seq_len=X_train_batch.default_bucket_key,
                             num_hidden=num_hidden,
                             num_embed=num_embed,
                             num_label=vocab,
                             batch_size=batch_size,
                             input_size=vocab,
                             initializer=mx.initializer.Uniform(0.1),dropout=0.5, buckets=buckets)

lstm.train_lstm(model, X_train_batch, X_val_batch,
                num_round=num_round,
                concat_decode=concat_decode,
                use_loss=use_loss,
                half_life=2,
                max_grad_norm = max_grad_norm,
                update_period=update_period,
                learning_rate=learning_rate,
                batch_size = batch_size,
                wd=wd)
