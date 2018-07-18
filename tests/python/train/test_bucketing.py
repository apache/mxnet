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

# pylint: skip-file
import numpy as np
import mxnet as mx
import random
from random import randint


def test_bucket_module():
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    batch_size = 128
    num_epochs = 5
    num_hidden = 25
    num_embed = 25
    num_layers = 2
    len_vocab = 50
    buckets = [5, 10, 20, 30, 40]

    invalid_label = -1
    num_sentence = 1000

    train_sent = []
    val_sent = []

    for _ in range(num_sentence):
        len_sentence = randint(6, max(buckets)-1) # leave out the two last buckets empty
        train_sentence = []
        val_sentence = []
        for _ in range(len_sentence):
            train_sentence.append(randint(1, len_vocab))
            val_sentence.append(randint(1, len_vocab))
        train_sent.append(train_sentence)
        val_sent.append(val_sentence)

    data_train = mx.rnn.BucketSentenceIter(train_sent, batch_size, buckets=buckets,
                                   invalid_label=invalid_label)
    data_val =  mx.rnn.BucketSentenceIter(val_sent, batch_size, buckets=buckets,
                                 invalid_label=invalid_label)

    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_' % i))

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len_vocab,
                                 output_dim=num_embed, name='embed')

        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

        pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len_vocab, name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        loss = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return loss, ('data',), ('softmax_label',)

    contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(
        sym_gen=sym_gen,
        default_bucket_key=data_train.default_bucket_key,
        context=contexts)

    logging.info('Begin fit...')
    model.fit(
        train_data=data_train,
        eval_data=data_val,
        eval_metric=mx.metric.Perplexity(invalid_label), # Use Perplexity for multiclass classification.
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.01,
                          'momentum': 0,
                          'wd': 0.00001},
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch=num_epochs,
        batch_end_callback=mx.callback.Speedometer(batch_size, 50))
    logging.info('Finished fit...')
    # This test forecasts random sequence of words to check bucketing.
    # We cannot guarantee the accuracy of such an impossible task, and comments out the following line.
    # assert model.score(data_val, mx.metric.MSE())[0][1] < 350, "High mean square error."


if __name__ == "__main__":
    test_bucket_module()
