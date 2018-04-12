# !/usr/bin/env python

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

# -*- coding: utf-8 -*-

from collections import Counter
import itertools
import iterators
import os
import numpy as np
import pandas as pd
import mxnet as mx
import argparse
import pickle
import logging

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Deep neural network for multivariate time series forecasting",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', type=str, default='../data',
                    help='relative path to input data')
parser.add_argument('--output-dir', type=str, default='../results',
                    help='directory to save model files to')
parser.add_argument('--max-records', type=int, default=None,
                    help='total records before data split')
parser.add_argument('--train_fraction', type=float, default=0.8,
                    help='fraction of data to use for training. remainder used for testing.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size.')
parser.add_argument('--buckets', type=str, default="",
                    help='unique bucket sizes')
parser.add_argument('--char-embed', type=int, default=25,
                    help='Embedding size for each unique character.')
parser.add_argument('--char-filter-list', type=str, default="3,4,5",
                    help='unique filter sizes for char level cnn')
parser.add_argument('--char-filters', type=int, default=20,
                    help='number of each filter size')
parser.add_argument('--word-embed', type=int, default=500,
                    help='Embedding size for each unique character.')
parser.add_argument('--word-filter-list', type=str, default="3,4,5",
                    help='unique filter sizes for char level cnn')
parser.add_argument('--word-filters', type=int, default=200,
                    help='number of each filter size')
parser.add_argument('--lstm-state-size', type=int, default=100,
                    help='number of hidden units in each unrolled recurrent cell')
parser.add_argument('--lstm-layers', type=int, default=1,
                    help='number of recurrent layers')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer type')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout rate for network')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='max num of epochs')
parser.add_argument('--save-period', type=int, default=20,
                    help='save checkpoint for every n epochs')
parser.add_argument('--model_prefix', type=str, default='electricity_model',
                    help='prefix for saving model params')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def save_model():
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    return mx.callback.do_checkpoint(os.path.join(args.output_dir, "checkpoint"), args.save_period)

def build_vocab(nested_list):
    """
    :param nested_list: list of list of string
    :return: dictionary mapping from string to int, inverse of that dictionary
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*nested_list))

    # Mapping from index to label
    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    # Mapping from label to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv

def build_iters(data_dir, max_records, train_fraction, batch_size, buckets=None):
    """
    Reads a csv of sentences/tag sequences into a pandas dataframe.
    Converts into X = array(list(int)) & Y = array(list(int))
    Splits into training and test sets
    Builds dictionaries mapping from index labels to labels/ indexed features to features
    :param data_dir: directory to read in csv data from
    :param max_records: total number of records to randomly select from input data
    :param train_fraction: fraction of the data to use for training
    :param batch_size: records in mini-batches during training
    :param buckets: size of each bucket in the iterators
    :return: train_iter, val_iter, word_to_index, index_to_word, pos_to_index, index_to_pos
    """
    # Read in data as numpy array
    df = pd.read_pickle(os.path.join(data_dir, "ner_data.pkl"))[:max_records]

    # Get feature lists
    entities=[list(array) for array in df["BILOU_tag"].values]
    sentences = [list(array) for array in df["token"].values]
    chars=[[[c for c in word] for word in sentence] for sentence in sentences]

    # Build vocabularies
    entity_to_index, index_to_entity = build_vocab(entities)
    word_to_index, index_to_word = build_vocab(sentences)
    char_to_index, index_to_char = build_vocab([np.array([c for c in word]) for word in index_to_word])
    save_obj(entity_to_index, os.path.join(args.data_dir, "tag_to_index"))

    # Map strings to integer values
    indexed_entities=[list(map(entity_to_index.get, l)) for l in entities]
    indexed_tokens=[list(map(word_to_index.get, l)) for l in sentences]
    indexed_chars=[[list(map(char_to_index.get, word)) for word in sentence] for sentence in chars]

    # Split into training and testing data
    idx=int(len(indexed_tokens)*train_fraction)
    X_token_train, X_char_train, Y_train = indexed_tokens[:idx], indexed_chars[:idx], indexed_entities[:idx]
    X_token_test, X_char_test, Y_test = indexed_tokens[idx:], indexed_chars[idx:], indexed_entities[idx:]

    # build iterators to feed batches to network
    train_iter = iterators.BucketNerIter(sentences=X_token_train, characters=X_char_train, label=Y_train,
                                         max_token_chars=5, batch_size=batch_size, buckets=buckets)
    val_iter = iterators.BucketNerIter(sentences=X_token_test, characters=X_char_test, label=Y_test,
                                         max_token_chars=train_iter.max_token_chars, batch_size=batch_size, buckets=train_iter.buckets)
    return train_iter, val_iter, word_to_index, char_to_index, entity_to_index

def sym_gen(seq_len):
    """
    Build NN symbol depending on the length of the input sequence
    """
    sentence_shape = train_iter.provide_data[0][1]
    char_sentence_shape = train_iter.provide_data[1][1]
    entities_shape = train_iter.provide_label[0][1]

    X_sent = mx.symbol.Variable(train_iter.provide_data[0].name)
    X_char_sent = mx.symbol.Variable(train_iter.provide_data[1].name)
    Y = mx.sym.Variable(train_iter.provide_label[0].name)

    ###############################
    # Character embedding component
    ###############################
    char_embeddings = mx.sym.Embedding(data=X_char_sent, input_dim=len(char_to_index), output_dim=args.char_embed, name='char_embed')
    char_embeddings = mx.sym.reshape(data=char_embeddings, shape=(0,1,seq_len,-1,args.char_embed), name='char_embed2')

    char_cnn_outputs = []
    for i, filter_size in enumerate(args.char_filter_list):
        # Kernel that slides over entire words resulting in a 1d output
        convi = mx.sym.Convolution(data=char_embeddings, kernel=(1, filter_size, args.char_embed), stride=(1, 1, 1),
                                   num_filter=args.char_filters, name="char_conv_layer_" + str(i))
        acti = mx.sym.Activation(data=convi, act_type='tanh')
        pooli = mx.sym.Pooling(data=acti, pool_type='max', kernel=(1, char_sentence_shape[2] - filter_size + 1, 1),
                               stride=(1, 1, 1), name="char_pool_layer_" + str(i))
        pooli = mx.sym.transpose(mx.sym.Reshape(pooli, shape=(0, 0, 0)), axes=(0, 2, 1), name="cchar_conv_layer_" + str(i))
        char_cnn_outputs.append(pooli)

    # combine features from all filters & apply dropout
    cnn_char_features = mx.sym.Concat(*char_cnn_outputs, dim=2, name="cnn_char_features")
    regularized_cnn_char_features = mx.sym.Dropout(data=cnn_char_features, p=args.dropout, mode='training',
                                                   name='regularized charCnn features')

    ##################################
    # Combine char and word embeddings
    ##################################
    word_embeddings = mx.sym.Embedding(data=X_sent, input_dim=len(word_to_index), output_dim=args.word_embed, name='word_embed')
    rnn_features = mx.sym.Concat(*[word_embeddings, regularized_cnn_char_features], dim=2, name='rnn input')

    ##############################
    # Bidirectional LSTM component
    ##############################

    # unroll the lstm cell in time, merging outputs
    bi_cell.reset()
    output, states = bi_cell.unroll(length=seq_len, inputs=rnn_features, merge_outputs=True)

    # Map to num entity classes
    rnn_output = mx.sym.Reshape(output, shape=(-1, args.lstm_state_size * 2), name='r_output')
    fc = mx.sym.FullyConnected(data=rnn_output, num_hidden=len(entity_to_index), name='fc_layer')

    # reshape back to same shape as loss will be
    reshaped_fc = mx.sym.transpose(mx.sym.reshape(fc, shape=(-1, seq_len, len(entity_to_index))), axes=(0, 2, 1))
    sm = mx.sym.SoftmaxOutput(data=reshaped_fc, label=Y, ignore_label=-1, use_ignore=True, multi_output=True, name='softmax')
    return sm, [v.name for v in train_iter.provide_data], [v.name for v in train_iter.provide_label]

def train(train_iter, val_iter):
    import metrics
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    module = mx.mod.BucketingModule(sym_gen, train_iter.default_bucket_key, context=devs)
    module.fit(train_data=train_iter,
               eval_data=val_iter,
               eval_metric=metrics.composite_classifier_metrics(),
               optimizer=args.optimizer,
               optimizer_params={'learning_rate': args.lr },
               initializer=mx.initializer.Uniform(0.1),
               num_epoch=args.num_epochs,
               epoch_end_callback=save_model())

if __name__ == '__main__':
    # parse args
    args = parser.parse_args()
    args.buckets = list(map(int, args.buckets.split(','))) if len(args.buckets) > 0 else None
    args.char_filter_list = list(map(int, args.char_filter_list.split(',')))

    # Build data iterators
    train_iter, val_iter, word_to_index, char_to_index, entity_to_index = build_iters(args.data_dir, args.max_records,
                                                                     args.train_fraction, args.batch_size, args.buckets)

    # Define the recurrent layer
    bi_cell = mx.rnn.SequentialRNNCell()
    for layer_num in range(args.lstm_layers):
        bi_cell.add(mx.rnn.BidirectionalCell(
            mx.rnn.LSTMCell(num_hidden=args.lstm_state_size, prefix="forward_layer_" + str(layer_num)),
            mx.rnn.LSTMCell(num_hidden=args.lstm_state_size, prefix="backward_layer_" + str(layer_num))))
        bi_cell.add(mx.rnn.DropoutCell(args.dropout))

    train(train_iter, val_iter)