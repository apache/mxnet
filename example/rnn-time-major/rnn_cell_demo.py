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

"""A simple demo of new RNN cell with PTB language model."""

################################################################################
# Speed test (time major is 1.5~2 times faster than batch major).
#
# -- This script (time major) -----
# 2016-10-10 18:43:21,890 Epoch[0] Batch [50]     Speed: 1717.76 samples/sec      Train-Perplexity=4311.345018
# 2016-10-10 18:43:25,959 Epoch[0] Batch [100]    Speed: 1573.17 samples/sec      Train-Perplexity=844.092421
# 2016-10-10 18:43:29,807 Epoch[0] Batch [150]    Speed: 1663.17 samples/sec      Train-Perplexity=498.080716
# 2016-10-10 18:43:33,871 Epoch[0] Batch [200]    Speed: 1574.84 samples/sec      Train-Perplexity=455.051252
# 2016-10-10 18:43:37,720 Epoch[0] Batch [250]    Speed: 1662.87 samples/sec      Train-Perplexity=410.500066
# 2016-10-10 18:43:40,766 Epoch[0] Batch [300]    Speed: 2100.81 samples/sec      Train-Perplexity=274.317460
# 2016-10-10 18:43:44,571 Epoch[0] Batch [350]    Speed: 1682.45 samples/sec      Train-Perplexity=350.132577
# 2016-10-10 18:43:48,377 Epoch[0] Batch [400]    Speed: 1681.41 samples/sec      Train-Perplexity=320.674884
# 2016-10-10 18:43:51,253 Epoch[0] Train-Perplexity=336.210212
# 2016-10-10 18:43:51,253 Epoch[0] Time cost=33.529
# 2016-10-10 18:43:53,373 Epoch[0] Validation-Perplexity=282.453883
#
# -- ../rnn/rnn_cell_demo.py (batch major) -----
# 2016-10-10 18:44:34,133 Epoch[0] Batch [50]     Speed: 1004.50 samples/sec      Train-Perplexity=4398.428571
# 2016-10-10 18:44:39,874 Epoch[0] Batch [100]    Speed: 1114.85 samples/sec      Train-Perplexity=771.401960
# 2016-10-10 18:44:45,528 Epoch[0] Batch [150]    Speed: 1132.03 samples/sec      Train-Perplexity=525.207444
# 2016-10-10 18:44:51,564 Epoch[0] Batch [200]    Speed: 1060.37 samples/sec      Train-Perplexity=453.741140
# 2016-10-10 18:44:57,865 Epoch[0] Batch [250]    Speed: 1015.78 samples/sec      Train-Perplexity=411.914237
# 2016-10-10 18:45:04,032 Epoch[0] Batch [300]    Speed: 1037.92 samples/sec      Train-Perplexity=381.302188
# 2016-10-10 18:45:10,153 Epoch[0] Batch [350]    Speed: 1045.49 samples/sec      Train-Perplexity=363.326871
# 2016-10-10 18:45:16,062 Epoch[0] Batch [400]    Speed: 1083.21 samples/sec      Train-Perplexity=377.929014
# 2016-10-10 18:45:19,993 Epoch[0] Train-Perplexity=294.675899
# 2016-10-10 18:45:19,993 Epoch[0] Time cost=52.604
# 2016-10-10 18:45:21,401 Epoch[0] Validation-Perplexity=294.345659
################################################################################

import os
import numpy as np
import mxnet as mx

from bucket_io import BucketSentenceIter, default_build_vocab

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


def Perplexity(label, pred):
    """ Calculates prediction perplexity

    Args:
        label (mx.nd.array): labels array
        pred (mx.nd.array): prediction array

    Returns:
        float: calculated perplexity

    """

    # collapse the time, batch dimension
    label = label.reshape((-1,))
    pred = pred.reshape((-1, pred.shape[-1]))

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

    # Update count per available GPUs
    gpu_count = 1
    contexts = [mx.context.gpu(i) for i in range(gpu_count)]

    vocab = default_build_vocab(os.path.join(data_dir, 'ptb.train.txt'))

    init_h = [mx.io.DataDesc('LSTM_state', (num_lstm_layer, batch_size, num_hidden), layout='TNC')]
    init_c = [mx.io.DataDesc('LSTM_state_cell', (num_lstm_layer, batch_size, num_hidden), layout='TNC')]
    init_states = init_c + init_h

    data_train = BucketSentenceIter(os.path.join(data_dir, 'ptb.train.txt'),
                                    vocab, buckets, batch_size, init_states,
                                    time_major=True)
    data_val = BucketSentenceIter(os.path.join(data_dir, 'ptb.valid.txt'),
                                  vocab, buckets, batch_size, init_states,
                                  time_major=True)

    def sym_gen(seq_len):
        """ Generates the MXNet symbol for the RNN

        Args:
            seq_len (int): input sequence length

        Returns:
            tuple: tuple containing symbol, data_names, label_names

        """
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=num_embed, name='embed')

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
        rnn = mx.sym.RNN(data=embed, state_size=num_hidden,
                         num_layers=num_lstm_layer, mode='lstm',
                         name='LSTM',
                         # The following params can be omitted
                         # provided we do not need to apply the
                         # workarounds mentioned above
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

        pred = mx.sym.FullyConnected(data=hidden, num_hidden=len(vocab),
                                     name='pred')

        # reshape to be of compatible shape as labels
        pred_tm = mx.sym.Reshape(data=pred, shape=(seq_len, -1, len(vocab)))

        sm = mx.sym.SoftmaxOutput(data=pred_tm, label=label, preserve_shape=True,
                                  name='softmax')

        data_names = ['data', 'LSTM_state', 'LSTM_state_cell']
        label_names = ['softmax_label']

        return sm, data_names, label_names

    if len(buckets) == 1:
        mod = mx.mod.Module(*sym_gen(buckets[0]), context=contexts)
    else:
        mod = mx.mod.BucketingModule(sym_gen,
                                     default_bucket_key=data_train.default_bucket_key,
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
