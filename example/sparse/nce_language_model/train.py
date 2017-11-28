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
from data import Corpus, CorpusIter, DummyIter
from model import rnn_model, nce_loss, ce_loss
import os

parser = argparse.ArgumentParser(description='PennTreeBank RNN/LSTM Language Model with Noice Contrastive Estimation')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--data', type=str, default='./data/ptb.',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay for sgd')
# TODO perform clip global norm
parser.add_argument('--clip', type=float, default=15,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--scale', type=int, default=1,
                    help='scaling factor for vocab size')
parser.add_argument('--k', type=int, default=32,
                    help='number of noise samples to estimate')
parser.add_argument('--use-dense', action='store_true',
                    help='use dense embedding instead of sparse embedding')
parser.add_argument('--use-full-softmax', action='store_true',
                    help='use full softmax ce loss instead of noise contrastive estimation')
parser.add_argument('--cuda', action='store_true',
                    help='whether to use gpu')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
#parser.add_argument('--save', type=str, default='model.params',
#                    help='path to save the final model')
args = parser.parse_args()

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    args = parser.parse_args()
    ctx = mx.gpu(0) if args.cuda else mx.cpu()
    full_softmax = args.use_full_softmax

    # data
    corpus = Corpus(args.data)
    ntokens = len(corpus.dictionary) * args.scale
    # TODO dict should be the unigram for train?
    train_data = mx.io.PrefetchingIter(CorpusIter(corpus.train, args.batch_size, args.bptt, args.k, corpus.dictionary.unigram()))
    # val_data = mx.io.PrefetchingIter(CorpusIter(corpus.valid, args.batch_size, args.bptt, args.k))
    # model
    rnn_out = rnn_model(args.bptt, "lstm", ntokens, args.emsize, args.nhid,
                        args.nlayers, args.dropout, args.use_dense)
    model = nce_loss(rnn_out, ntokens, args.nhid, args.k) if not full_softmax else ce_loss(rnn_out, ntokens)
    state_names = ['sample'] if not full_softmax else None
    # module
    module = mx.mod.Module(symbol=model, context=ctx, state_names=state_names,
                           data_names=['data'], label_names=['label'])
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    module.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    optimizer_params=(('learning_rate', args.lr), ('wd', args.wd), ('momentum', args.mom),
                      ('clip_gradient', args.clip))
    module.init_optimizer(optimizer='sgd', optimizer_params=optimizer_params)
    # use accuracy as the metric
    metric = mx.metric.Perplexity(ignore_label=None)
    speedometer = mx.callback.Speedometer(args.batch_size, args.log_interval)

    # train
    logging.info("Training started ... ")
    for epoch in range(args.epochs):
        nbatch = 0
        metric.reset()
        for batch in train_data:
            nbatch += 1
            if not full_softmax:
                samples = batch.data[1]
                module.set_states(value=samples)
            module.forward_backward(batch)
            # update all parameters (including the weight parameter)
            module.update()
            # update training metric
            module.update_metric(metric, batch.label)
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=metric, locals=locals())
            speedometer(speedometer_param)
        train_data.reset()
    logging.info("Training completed. ")
