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

    class DummySentenceIter(mx.rnn.BucketSentenceIter):
        """Dummy sentence iterator to output sentences the same as input.
        """

        def __init__(self, sentences, batch_size, buckets=None, invalid_label=-1,
                     data_name='data', label_name='l2_label', dtype='float32',
                     layout='NTC'):
            super(DummySentenceIter, self).__init__(sentences, batch_size,
                                                    buckets=buckets, invalid_label=invalid_label,
                                                    data_name=data_name, label_name=label_name,
                                                    dtype=dtype, layout=layout)

        def reset(self):
            """Resets the iterator to the beginning of the data."""
            self.curr_idx = 0
            random.shuffle(self.idx)
            for buck in self.data:
                np.random.shuffle(buck)

            self.nddata = []
            self.ndlabel = []
            for buck in self.data:
                self.nddata.append(mx.nd.array(buck, dtype=self.dtype))
                self.ndlabel.append(mx.nd.array(buck, dtype=self.dtype))

    batch_size = 128
    num_epochs = 5
    num_hidden = 25
    num_embed = 25
    num_layers = 2
    len_vocab = 50
    buckets = [10, 20, 30, 40]

    invalid_label = 0
    num_sentence = 1000

    train_sent = []
    val_sent = []

    for _ in range(num_sentence):
        len_sentence = randint(1, max(buckets) + 10)
        train_sentence = []
        val_sentence = []
        for _ in range(len_sentence):
            train_sentence.append(randint(1, len_vocab))
            val_sentence.append(randint(1, len_vocab))
        train_sent.append(train_sentence)
        val_sent.append(val_sentence)

    data_train = DummySentenceIter(train_sent, batch_size, buckets=buckets,
                                   invalid_label=invalid_label)
    data_val = DummySentenceIter(val_sent, batch_size, buckets=buckets,
                                 invalid_label=invalid_label)

    stack = mx.rnn.SequentialRNNCell()
    for i in range(num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=num_hidden, prefix='lstm_l%d_' % i))

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('l2_label')
        embed = mx.sym.Embedding(data=data, input_dim=len_vocab,
                                 output_dim=num_embed, name='embed')

        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

        pred = mx.sym.Reshape(outputs, shape=(-1, num_hidden))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=1, name='pred')
        pred = mx.sym.reshape(pred, shape=(batch_size, -1))
        loss = mx.sym.LinearRegressionOutput(pred, label, name='l2_loss')

        return loss, ('data',), ('l2_label',)

    contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(
        sym_gen=sym_gen,
        default_bucket_key=data_train.default_bucket_key,
        context=contexts)

    logging.info('Begin fit...')
    model.fit(
        train_data=data_train,
        eval_data=data_val,
        eval_metric=mx.metric.MSE(),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.01,
                          'momentum': 0,
                          'wd': 0.00001},
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch=num_epochs,
        batch_end_callback=mx.callback.Speedometer(batch_size, 50))
    logging.info('Finished fit...')
    assert model.score(data_val, mx.metric.MSE())[0][1] < 350, "High mean square error."


if __name__ == "__main__":
    test_bucket_module()
