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

"""Gluon Text Sentiment Classification Example using RNN
Example modified from below link to demostrate using fit() API:
https://github.com/d2l-ai/d2l-en/blob/master/chapter_natural-language-processing/sentiment-analysis-rnn.md"""

import argparse
from mxnet import gluon, init, nd, cpu, gpu
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn
from mxnet.gluon.estimator import estimator as est
from data_utils import download_imdb, read_imdb, get_vocab_imdb, preprocess_imdb


# Model
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # Set Bidirectional to True to get a bidirectional recurrent neural
        # network
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of inputs is (batch size, number of words). Because LSTM
        # needs to use sequence as the first dimension, the input is
        # transformed and the word feature is then extracted. The output shape
        # is (number of words, batch size, word vector dimension).
        embeddings = self.embedding(inputs.T)
        # The shape of states is (number of words, batch size, 2 * number of
        # hidden units).
        states = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * number of hidden units)
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs


if __name__ == '__main__':
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='MXNet Gluon Text Sentiment '
                                                 'Classification Example using RNN')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training and testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='whether to use GPU (default: False)')
    opt = parser.parse_args()

    ctx = gpu(0) if opt.use_gpu else cpu()

    # data
    download_imdb()
    train_data, test_data = read_imdb('train'), read_imdb('test')
    vocab = get_vocab_imdb(train_data)

    train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
    test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
    train_iter = gdata.DataLoader(train_set, opt.batch_size, shuffle=True)
    test_iter = gdata.DataLoader(test_set, opt.batch_size)

    embed_size, num_hiddens, num_layers = 100, 100, 2

    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    net.initialize(init.Xavier(), ctx=ctx)

    glove_embedding = text.embedding.create(
        'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)

    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.embedding.collect_params().setattr('grad_req', 'null')

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': opt.lr})
    loss = gloss.SoftmaxCrossEntropyLoss()

    # train
    e = est.Estimator(net, loss=loss, trainers=trainer, context=ctx)
    e.fit(train_iter, test_iter, epochs=opt.epochs, batch_size=opt.batch_size)
