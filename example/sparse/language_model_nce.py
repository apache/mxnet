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

import numpy as np
import mxnet as mx
import argparse
from get_data import Corpus, CorpusIter
from nce_model import rnn_model, nce_loss
import os

'''
parser = argparse.ArgumentParser(description='PennTreeBank RNN/LSTM Language Model with Noice Contrastive Estimation')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--cuda', action='store_true',
                    help='Whether to use gpu')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
args = parser.parse_args()
'''
parser = argparse.ArgumentParser(description="Train a language model with RNN on Penn Tree Bank",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, default='./data/ptb.',
                    help='location of the data corpus')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of stacked RNN layers')
parser.add_argument('--num-hidden', type=int, default=200,
                    help='hidden layer size')
parser.add_argument('--num-embed', type=int, default=200,
                    help='embedding layer size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--num-epochs', type=int, default=25,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the optimizer type')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay for sgd')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size.')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
parser.add_argument('--use-dense', action='store_true',
                    help='use dense embedding instead of sparse embedding')

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    args = parser.parse_args()

    # TODO where is this used?
    invalid_label = 0

    # data
    corpus = Corpus(args.data)
    ntokens = len(corpus.dictionary)

    train_data = mx.io.PrefetchingIter(CorpusIter(corpus.train, args.batch_size, args.bptt))
    val_data = mx.io.PrefetchingIter(CorpusIter(corpus.valid, args.batch_size, args.bptt))

    model = rnn_model(args.bptt, "lstm", ntokens, args.num_embed, args.num_hidden,
                      args.num_layers, args.use_dense)

    module = mx.mod.Module(symbol=model, context=mx.cpu(), label_names=['label'])
    logging.info("Training started ... ")

    module.fit(
        train_data          = train_data,
        eval_data           = val_data,
        eval_metric         = mx.metric.Perplexity(invalid_label),
        optimizer           = args.optimizer,
        optimizer_params    = { 'learning_rate': args.lr,
                                'momentum': args.mom,
                                'wd': args.wd },
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = args.num_epochs,
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches, auto_reset=False))
