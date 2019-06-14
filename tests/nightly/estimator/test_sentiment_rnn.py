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

"""Gluon Text Sentiment Classification Example using RNN/CNN
Example modified from below link:
https://github.com/d2l-ai/d2l-en/blob/master/chapter_natural-language-processing/sentiment-analysis-rnn.md
https://github.com/d2l-ai/d2l-en/blob/master/chapter_natural-language-processing/sentiment-analysis-cnn.md"""

import argparse
import os
import tarfile
import random
import collections
import mxnet as mx
from mxnet import nd, gluon
from mxnet.contrib import text
from mxnet.gluon import nn, rnn
from mxnet.gluon.contrib.estimator import estimator


class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # The embedding layer does not participate in training
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no weight, so it can share an
        # instance
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate the output of two embedding layers with shape of
        # (batch size, number of words, word vector dimension) by word vector
        embeddings = nd.concat(
            self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # According to the input format required by Conv1D, the word vector
        # dimension, that is, the channel dimension of the one-dimensional
        # convolutional layer, is transformed into the previous dimension
        embeddings = embeddings.transpose((0, 2, 1))
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, an NDArray with the shape of (batch size, channel size, 1)
        # can be obtained. Use the flatten function to remove the last
        # dimension and then concatenate on the channel dimension
        encoding = nd.concat(*[nd.flatten(
            self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # After applying the dropout method, use a fully connected layer to
        # obtain the output
        outputs = self.decoder(self.dropout(encoding))
        return outputs


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


def download_imdb(data_dir='/tmp/data'):
    '''
    Download and extract the IMDB dataset
    '''
    # Large Movie Review Dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    # Note this dataset is copyright to Andrew Maas and Stanford AI Lab
    # @InProceedings{maas-EtAl:2011:ACL-HLT2011,
    #   author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
    #   title     = {Learning Word Vectors for Sentiment Analysis},
    #   booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
    #   month     = {June},
    #   year      = {2011},
    #   address   = {Portland, Oregon, USA},
    #   publisher = {Association for Computational Linguistics},
    #   pages     = {142--150},
    #   url       = {http://www.aclweb.org/anthology/P11-1015}
    # }
    url = ('https://aws-ml-platform-datasets.s3.amazonaws.com/imdb/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')
    if not os.path.isfile(file_path):
        file_path = gluon.utils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(file_path, 'r') as f:
        f.extractall(data_dir)


def read_imdb(folder='train'):
    '''
    Read the IMDB dataset
    '''
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('/tmp/data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


def get_tokenized_imdb(data):
    '''
    Tokenized the words
    '''

    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    '''
    Get the indexed tokens
    '''
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)


def preprocess_imdb(data, vocab):
    '''
    Make the length of each comment 500 by truncating or adding 0s
    '''
    max_l = 500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels


def run(net, train_dataloader, test_dataloader, **kwargs):
    '''
    Train a test sentiment model
    '''
    num_epochs = kwargs['epochs']
    ctx = kwargs['ctx']
    batch_size = kwargs['batch_size']
    lr = kwargs['lr']

    # Define trainer
    trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    # Define loss and evaluation metrics
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()

    # Define estimator
    est = estimator.Estimator(net=net, loss=loss, metrics=acc,
                              trainer=trainer, context=ctx)
    # Begin training
    est.fit(train_data=train_dataloader, val_data=test_dataloader,
            epochs=num_epochs)
    return acc


def test_estimator_cpu(**kwargs):
    '''
    Test estimator by doing one pass over each model with synthetic data
    '''
    models = ['TextCNN', 'BiRNN']
    ctx = kwargs['ctx']
    batch_size = kwargs['batch_size']
    embed_size = kwargs['embed_size']

    train_data = mx.nd.random.randint(low=0, high=100, shape=(2 * batch_size, 500))
    train_label = mx.nd.random.randint(low=0, high=2, shape=(2 * batch_size,))
    val_data = mx.nd.random.randint(low=0, high=100, shape=(batch_size, 500))
    val_label = mx.nd.random.randint(low=0, high=2, shape=(batch_size,))

    train_dataloader = gluon.data.DataLoader(dataset=gluon.data.ArrayDataset(train_data, train_label),
                                             batch_size=batch_size, shuffle=True)
    val_dataloader = gluon.data.DataLoader(dataset=gluon.data.ArrayDataset(val_data, val_label),
                                           batch_size=batch_size)
    vocab_list = mx.nd.zeros(shape=(100,))

    # Get the model
    for model in models:
        if model == 'TextCNN':
            kernel_sizes, nums_channels = [3, 4, 5], [100, 100, 100]
            net = TextCNN(vocab_list, embed_size, kernel_sizes, nums_channels)
        else:
            num_hiddens, num_layers = 100, 2
            net = BiRNN(vocab_list, embed_size, num_hiddens, num_layers)
        net.initialize(mx.init.Xavier(), ctx=ctx)

        run(net, train_dataloader, val_dataloader, **kwargs)


def test_estimator_gpu(**kwargs):
    '''
    Test estimator by training Bidirectional RNN for 5 epochs on the IMDB dataset
    and verify accuracy
    '''
    ctx = kwargs['ctx']
    batch_size = kwargs['batch_size']
    num_epochs = kwargs['epochs']
    embed_size = kwargs['embed_size']

    # data
    download_imdb()
    train_data, test_data = read_imdb('train'), read_imdb('test')
    vocab = get_vocab_imdb(train_data)

    train_set = gluon.data.ArrayDataset(*preprocess_imdb(train_data, vocab))
    test_set = gluon.data.ArrayDataset(*preprocess_imdb(test_data, vocab))
    train_dataloader = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
    test_dataloader = gluon.data.DataLoader(test_set, batch_size)

    # Model
    num_hiddens, num_layers = 100, 2
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    net.initialize(mx.init.Xavier(), ctx=ctx)

    glove_embedding = text.embedding.create(
        'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)

    net.embedding.weight.set_data(glove_embedding.idx_to_vec)
    net.embedding.collect_params().setattr('grad_req', 'null')

    acc = run(net, train_dataloader, test_dataloader, **kwargs)

    assert acc.get()[1] > 0.70


parser = argparse.ArgumentParser(description='test gluon estimator')
parser.add_argument('--type', type=str, default='cpu')
opt = parser.parse_args()
kwargs = {
    'batch_size': 64,
    'lr': 0.01,
    'embed_size': 100
}

if opt.type == 'cpu':
    kwargs['ctx'] = mx.cpu()
    kwargs['epochs'] = 1
    test_estimator_cpu(**kwargs)
elif opt.type == 'gpu':
    kwargs['ctx'] = mx.gpu()
    kwargs['epochs'] = 5
    test_estimator_gpu(**kwargs)
else:
    raise RuntimeError("Unknown test type")
