#!/usr/bin/env python

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


import boto3
import mxnet as mx
import json
import os
import numpy as np
import logging
from mxnet import nd, autograd, gluon
import mxnet.ndarray as nd
from mxnet.gluon.data.vision import transforms, datasets
from mxnet import autograd as ag
import mxnet.ndarray as F
from mxnet.gluon import nn, rnn
import re
import time
import sys

# Set fixed random seeds.
mx.random.seed(7)
np.random.seed(7)
logging.getLogger().setLevel(logging.DEBUG)

# get the current mxnet version we are running on
mxnet_version = mx.__version__
bucket_name = 'mxnet-model-backwards-compatibility'
backslash = '/'
s3 = boto3.resource('s3')
num_epoch = 2

def prepare_mnist_data(mnist_raw_data):
    
    #shuffle the indices
    indices = np.random.permutation(mnist_raw_data['train_label'].shape[0])

    #print indices[0:10]
    train_idx , val_idx = indices[:50000], indices[50000:]

    train_data = mnist_raw_data['train_data'][train_idx,:]
    train_label = mnist_raw_data['train_label'][train_idx]
    
    val_data = mnist_raw_data['train_data'][val_idx,:]
    val_label = mnist_raw_data['train_label'][val_idx]
    
    test_data = mnist_raw_data['test_data']
    test_label = mnist_raw_data['test_label']

    #print len(train_data)
    #print len(val_data)
    
    train = {'train_X' : train_data, 'train_Y' : train_label}
    test = {'test_X' : test_data, 'test_Y' : test_label}
    val = {'val_X' : val_data, 'val_Y' : val_label}
    
    data = dict()
    data['train'] = train
    data['test'] = test
    data['val'] = val
    
    return data

def get_top_level_folders_in_bucket(s3client, bucket_name):
    '''This function returns the top level folders in the S3Bucket. These folders help us to navigate to the trained model files stored for different MXNet versions. '''
    bucket = s3client.Bucket(bucket_name)
    result = bucket.meta.client.list_objects(Bucket=bucket.name,
                                         Delimiter=backslash)
    folder_list = list()
    for obj in result['CommonPrefixes']:
        folder_list.append(obj['Prefix'].strip(backslash))

    return folder_list

def clean_mnist_data():
    if os.path.isfile('train-images-idx3-ubyte.gz'):
        os.remove('train-images-idx3-ubyte.gz')
    if os.path.isfile('t10k-labels-idx1-ubyte.gz'):
        os.remove('t10k-labels-idx1-ubyte.gz')
    if os.path.isfile('train-labels-idx1-ubyte.gz'):
        os.remove('train-labels-idx1-ubyte.gz')
    if os.path.isfile('t10k-images-idx3-ubyte.gz'):
        os.remove('t10k-images-idx3-ubyte.gz')

def clean_model_files(model_files):
    for file in model_files:
        if os.path.isfile(file):
            os.remove(file)

def upload_model_files_to_s3(bucket_name, files, folder_name):
    s3 = boto3.client('s3')
    for file in files:
        s3.upload_file(file, bucket_name, folder_name + file)
    print ('model successfully uploaded to s3')

def save_inference_results(inference_results_file, inference_results):
    # Write the inference results to local json file. This will be cleaned up later
    with open(inference_results_file, 'w') as file:
        json.dump(inference_results, file)


def compare_versions(version1, version2):
    '''
    https://stackoverflow.com/questions/1714027/version-number-comparison-in-python
    '''
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$','', v).split(".")]
    return cmp(normalize(version1), normalize(version2))

def get_val_test_iter():
    data = prepare_mnist_data(mx.test_utils.get_mnist())
    val = data['val']
    test = data['test']
    batch_size = 100
    val_iter = mx.io.NDArrayIter(val['val_X'], val['val_Y'], batch_size, shuffle=True)
    test_iter = mx.io.NDArrayIter(test['test_X'], test['test_Y'])
    return val_iter, test_iter

class HybridNet(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(20, kernel_size=(5,5))
            self.pool1 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.conv2 = nn.Conv2D(50, kernel_size=(5,5))
            self.pool2 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.fc1 = nn.Dense(500)
            self.fc2 = nn.Dense(10)

    def hybrid_forward(self, F, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(20, kernel_size=(5,5))
            self.pool1 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.conv2 = nn.Conv2D(50, kernel_size=(5,5))
            self.pool2 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.fc1 = nn.Dense(500)
            self.fc2 = nn.Dense(10)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.download_data_from_s3()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def download_data_from_s3(self, ):
        print ('Downloading files from bucket : ptb-small-dataset' )
        bucket = s3.Bucket('ptb-small-dataset')
        files = ['test.txt', 'train.txt', 'valid.txt']
        for file in files:
            if os.path.exists(args_data + file) :
                print ('File %s'%(args_data + file), 'already exists. Skipping download')
                continue
            file_path = args_data + file
            bucket.download_file(file_path, args_data + file) 

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return mx.nd.array(ids, dtype='int32')



#### Common utilies for lm_rnn_gluon_train & inference files
args_data = 'ptb.'
args_model = 'rnn_relu'
args_emsize = 100
args_nhid = 100
args_nlayers = 2
args_lr = 1.0
args_clip = 0.2
args_epochs = 2
args_batch_size = 32
args_bptt = 5
args_dropout = 0.2
args_tied = True
args_cuda = 'store_true'
args_log_interval = 500

class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru"%mode)
            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden,
                                        params = self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def get_batch(source, i):
    seq_len = min(args_bptt, source.shape[0] - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def eval(data_source, model):
    total_L = 0.0
    ntotal = 0
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = args_batch_size, ctx=mx.cpu(0))
    for i in range(0, data_source.shape[0] - 1, args_bptt):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def clean_ptb_data():
    files = ['test.txt', 'train.txt', 'valid.txt']
    for file in files: 
        if os.path.isfile(args_data + file):
            os.remove(args_data + file)

# This function is added so that if a download gets interrupted in between, one can clean the corrupted files
clean_mnist_data()
clean_ptb_data()
