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

"""A simple demo of new RNN cell with sherlockholmes language model."""

import os

import numpy as np
import mxnet as mx

from bucket_io import BucketSentenceIter, default_build_vocab


data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


def Perplexity(label, pred):
    # TODO(tofix): we make a transpose of label here, because when
    # using the RNN cell, we called swap axis to the data.
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


if __name__ == '__main__':
    batch_size = 128
    buckets = [10, 20, 30, 40, 50, 60]
    num_hidden = 200
    num_embed = 200
    num_lstm_layer = 2

    num_epoch = 2
    learning_rate = 0.01
    momentum = 0.0

    contexts = [mx.context.gpu(i) for i in range(4)]
    vocab = default_build_vocab(os.path.join(data_dir, 'sherlockholmes.train.txt'))

    init_h = [('LSTM_init_h', (batch_size, num_lstm_layer, num_hidden))]
    init_c = [('LSTM_init_c', (batch_size, num_lstm_layer, num_hidden))]
    init_states = init_c + init_h

    data_train = BucketSentenceIter(os.path.join(data_dir, 'sherlockholmes.train.txt'),
                                    vocab, buckets, batch_size, init_states)
    data_val = BucketSentenceIter(os.path.join(data_dir, 'sherlockholmes.valid.txt'),
                                  vocab, buckets, batch_size, init_states)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=num_embed, name='embed')

        # TODO(tofix)
        # The inputs and labels from IO are all in batch-major.
        # We need to transform them into time-major to use RNN cells.
        embed_tm = mx.sym.SwapAxis(embed, dim1=0, dim2=1)
        label_tm = mx.sym.SwapAxis(label, dim1=0, dim2=1)

        # TODO(tofix)
        # Create transformed RNN initial states. Normally we do
        # no need to do this. But the RNN symbol expects the state
        # to be time-major shape layout, while the current mxnet
        # IO and high-level training logic assume everything from
        # the data iter have batch_size as the first dimension.
        # So until we have extended our IO and training logic to
        # support this more general case, this dummy axis swap is
        # needed.
        rnn_h_init = mx.sym.SwapAxis(mx.sym.Variable('LSTM_init_h'),
                                     dim1=0, dim2=1)
        rnn_c_init = mx.sym.SwapAxis(mx.sym.Variable('LSTM_init_c'),
                                     dim1=0, dim2=1)

        # TODO(tofix)
        # currently all the LSTM parameters are concatenated as
        # a huge vector, and named '<name>_parameters'. By default
        # mxnet initializer does not know how to initilize this
        # guy because its name does not ends with _weight or _bias
        # or anything familiar. Here we just use a temp workaround
        # to create a variable and name it as LSTM_bias to get
        # this demo running. Note by default bias is initialized
        # as zeros, so this is not a good scheme. But calling it
        # LSTM_weight is not good, as this is 1D vector, while
        # the initialization scheme of a weight parameter needs
        # at least two dimensions.
        rnn_params = mx.sym.Variable('LSTM_bias')

        # RNN cell takes input of shape (time, batch, feature)
        rnn = mx.sym.RNN(data=embed_tm, state_size=num_hidden,
                         num_layers=num_lstm_layer, mode='lstm',
                         name='LSTM',
                         # The following params can be omitted
                         # provided we do not need to apply the
                         # workarounds mentioned above
                         state=rnn_h_init,
                         state_cell=rnn_c_init,
                         parameters=rnn_params)

        # the RNN cell output is of shape (time, batch, dim)
        # if we need the states and cell states in the last time
        # step (e.g. when building encoder-decoder models), we
        # can set state_outputs=True, and the RNN cell will have
        # extra outputs: rnn['LSTM_output'], rnn['LSTM_state']
        # and for LSTM, also rnn['LSTM_state_cell']

        # now we collapse the time and batch dimension to do the
        # final linear logistic regression prediction
        hidden = mx.sym.Reshape(data=rnn, shape=(-1, num_hidden))
        label_cl = mx.sym.Reshape(data=label_tm, shape=(-1,))

        pred = mx.sym.FullyConnected(data=hidden, num_hidden=len(vocab),
                                     name='pred')
        sm = mx.sym.SoftmaxOutput(data=pred, label=label_cl, name='softmax')

        data_names = ['data', 'LSTM_init_h', 'LSTM_init_c']
        label_names = ['softmax_label']

        return (sm, data_names, label_names)

    if len(buckets) == 1:
        mod = mx.mod.Module(*sym_gen(buckets[0]), context=contexts)
    else:
        mod = mx.mod.BucketingModule(sym_gen, default_bucket_key=data_train.default_bucket_key,
                                     context=contexts)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    mod.fit(data_train, eval_data=data_val, num_epoch=num_epoch,
            eval_metric=mx.metric.np(Perplexity),
            batch_end_callback=mx.callback.Speedometer(batch_size, 50),
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            optimizer='sgd',
            optimizer_params={'learning_rate': learning_rate,
                              'momentum': momentum, 'wd': 0.00001})
