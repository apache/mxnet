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

import argparse
import time
import math
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import contrib
import model
import data

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Language Model on Wikitext-2.')
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
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
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
parser.add_argument('--gctype', type=str, default='none',
                    help='type of gradient compression to use, \
                          takes `2bit` or `none` for now.')
parser.add_argument('--gcthreshold', type=float, default=0.5,
                    help='threshold for 2bit gradient compression')
args = parser.parse_args()


###############################################################################
# Load data
###############################################################################


if args.cuda:
    context = mx.gpu(0)
else:
    context = mx.cpu(0)

train_dataset = contrib.data.text.WikiText2('./data', 'train', seq_len=args.bptt)
vocab = train_dataset.vocabulary
val_dataset, test_dataset = [contrib.data.text.WikiText2('./data', segment,
                                                         vocab=vocab,
                                                         seq_len=args.bptt)
                             for segment in ['validation', 'test']]

nbatch_train = len(train_dataset) / args.batch_size
train_data = gluon.data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   sampler=contrib.data.IntervalSampler(len(train_dataset),
                                                                        nbatch_train),
                                   last_batch='discard')

nbatch_val = len(val_dataset) / args.batch_size
val_data = gluon.data.DataLoader(val_dataset,
                                 batch_size=args.batch_size,
                                 sampler=contrib.data.IntervalSampler(len(val_dataset),
                                                                      nbatch_val),
                                 last_batch='discard')

nbatch_test = len(test_dataset) / args.batch_size
test_data = gluon.data.DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  sampler=contrib.data.IntervalSampler(len(test_dataset),
                                                                       nbatch_test),
                                  last_batch='discard')


###############################################################################
# Build the model
###############################################################################


ntokens = len(vocab)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                       args.nlayers, args.dropout, args.tied)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)

compression_params = None if args.gctype == 'none' else {'type': args.gctype, 'threshold': args.gcthreshold}
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args.lr,
                         'momentum': 0,
                         'wd': 0},
                        compression_params=compression_params)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=context)
    for i, (data, target) in enumerate(data_source):
        data = data.as_in_context(context).T
        target = target.as_in_context(context).T.reshape((-1, 1))
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def train():
    best_val = float("Inf")
    for epoch in range(args.epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=context)
        for i, (data, target) in enumerate(train_data):
            data = data.as_in_context(context).T
            target = target.as_in_context(context).T.reshape((-1, 1))
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [p.grad(context) for p in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, args.clip * args.bptt * args.batch_size)

            trainer.step(args.batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if i % args.log_interval == 0 and i > 0:
                cur_L = total_L / args.bptt / args.batch_size / args.log_interval
                print('[Epoch %d Batch %d] loss %.2f, ppl %.2f'%(
                    epoch, i, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = eval(val_data)

        print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f'%(
            epoch, time.time()-start_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = eval(test_data)
            model.collect_params().save(args.save)
            print('test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
        else:
            args.lr = args.lr*0.25
            trainer._init_optimizer('sgd',
                                    {'learning_rate': args.lr,
                                     'momentum': 0,
                                     'wd': 0})
            model.collect_params().load(args.save, context)

if __name__ == '__main__':
    train()
    model.collect_params().load(args.save, context)
    test_L = eval(test_data)
    print('Best test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
