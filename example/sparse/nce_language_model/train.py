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
from model import rnn, nce_loss, ce_loss
import os

parser = argparse.ArgumentParser(description='PennTreeBank RNN/LSTM Language Model with Noice Contrastive Estimation')
parser.add_argument('--data', type=str, default='./data/ptb.',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1500,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.1,
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
parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--scale', type=int, default=1,
                    help='scaling factor for vocab size')
parser.add_argument('--k', type=int, default=15,
                    help='number of noise samples to estimate')
parser.add_argument('--use-gpu', type=int, default=0,
                    help='which gpu to use')
parser.add_argument('--use-dense', action='store_true',
                    help='use dense embedding instead of sparse embedding')
parser.add_argument('--use-full-softmax', action='store_true',
                    help='use full softmax ce loss instead of noise contrastive estimation')
parser.add_argument('--cuda', action='store_true',
                    help='whether to use gpu')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--profile', action='store_true',
                    help='whether to use profiler')
parser.add_argument('--gpu-only', action='store_true',
                    help='whether to not use cpu')
parser.add_argument('--kvstore', type=str, default=None,
                    help='type of kvstore to use')
parser.add_argument('--dummy-iter', action='store_true',
                    help='whether to dummy data iterator')
#parser.add_argument('--save', type=str, default='model.params',
#                    help='path to save the final model')
args = parser.parse_args()

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    args = parser.parse_args()
    logging.info(args)
    ctx = mx.gpu(args.use_gpu) if args.cuda else mx.cpu()
    full_softmax = args.use_full_softmax

    # data
    corpus = Corpus(args.data)
    ntokens = len(corpus.dictionary) * args.scale
    # TODO dict should be the unigram for train?
    train_data = CorpusIter(corpus.train, args.batch_size, args.bptt, args.k, corpus.dictionary.unigram())
    if args.dummy_iter:
        train_data = DummyIter(train_data)
    #val_data = mx.io.PrefetchingIter(CorpusIter(corpus.valid, args.batch_size, args.bptt, args.k))
    # model
    on_cpu = True
    group2ctxs={'cpu_dev':mx.cpu(0), 'gpu_dev':mx.gpu(0)} if on_cpu else None
    rnn_out = rnn(args.bptt, ntokens, args.emsize, args.nhid,
                  args.nlayers, args.dropout, args.use_dense, on_cpu)
    model = nce_loss(rnn_out, ntokens, args.nhid, args.k, on_cpu) if not full_softmax else ce_loss(rnn_out, ntokens)
    state_names = ['sample'] if not full_softmax else None
    # module
    module = mx.mod.Module(symbol=model, context=ctx, state_names=state_names,
                           data_names=['data'], label_names=['label'], group2ctxs=group2ctxs)
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    module.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    kvstore = None if args.kvstore is None else mx.kv.create(kvstore)
    optimizer_params=(('learning_rate', args.lr), ('wd', args.wd), ('momentum', args.mom),
                      ('clip_gradient', args.clip))
    module.init_optimizer(optimizer='sgd', optimizer_params=optimizer_params, kvstore=kvstore)
    # use accuracy as the metric
    metric = mx.metric.Perplexity(ignore_label=None)
    speedometer = mx.callback.Speedometer(args.batch_size, args.log_interval)

    # get the sparse weight parameter
    encoder_weight_index = module._exec_group.param_names.index('encoder_weight')
    encoder_weight_param = module._exec_group.param_arrays[encoder_weight_index]
    if not full_softmax:
        decoder_w_index = module._exec_group.param_names.index('decoder_weight')
        decoder_w_param = module._exec_group.param_arrays[decoder_w_index]

    if args.profile:
        config = ['scale', args.scale, 'nhid', args.nhid, 'k', args.k, 'nlayers', args.nlayers,
                  'use_dense', args.use_dense, 'use_full_softmax', args.use_full_softmax]
        config_str = map(lambda x: str(x), config)
        filename = '-'.join(config_str) + '.json'
        mx.profiler.profiler_set_config(mode='all', filename=filename)
        mx.profiler.profiler_set_state('run')

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
            if kvstore:
                # TODO use kvstore
                #row_ids = batch.data[0].indices
                #kv.row_sparse_pull('weight', weight_param, row_ids=[row_ids],
                #                   priority=-weight_index)
                pass
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
    if args.profile:
        mx.profiler.profiler_set_state('stop')
