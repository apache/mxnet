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

# This file compares different RNN implementation on the PTB benchmark using the word-level language modeling.
# Note that this file is not exactly the same with the original source file.
# Specifically, the following changes have been made:
#     (1) fixed the UserWarning on inconsistent batch_size
#     (2) ported MXNet Default and OpenLSTMRNN implementation
#     (3) replaced MXNet Speedometer with TensorboardSpeedometer
#     (4) added MXNet profiler for detailed performance analysis

import math
import argparse

import mxnet as mx

from model import *
from module import *
from data import Corpus, CorpusIter
from mxnet.model import BatchEndParam

parser = argparse.ArgumentParser(description='PennTreeBank LSTM Language Model')
parser.add_argument('--dataset-dir', type=str, default='./dataset/',
                    help='location of the data corpus')
parser.add_argument('--dataset-name', type=str, default='ptb',
                    help='name of the data corpus (ptb/wikitext-2)')
parser.add_argument('--emsize', type=int, default=650,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping by global norm')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--backend', type=str, default='open',
                    help='select the RNN backend implementation')
parser.add_argument('--profiling', action='store_true',
                    help='whether to enable the MXNet profiling')
parser.add_argument('--profiling-start', type=int, default=200,
                    help='the start iteration of profiling')
parser.add_argument('--profiling-end', type=int, default=201,
                    help='the end iteration of profiling')
parser.add_argument('--profiler-output-fname', type=str, default='profiler_output.json',
                    help='filename of profiler output')
args = parser.parse_args()

best_loss = float('inf')


def evaluate(valid_module, data_iter,
             epoch, mode,
             bptt, batch_size,
             summary_writer):
    total_loss = 0.0
    nbatch = 0
    for batch in data_iter:
        valid_module.forward(batch, is_train=False)
        outputs = valid_module.get_loss()
        total_loss += mx.nd.sum(outputs[0]).asscalar()
        nbatch += 1
    data_iter.reset()
    loss = total_loss / bptt / batch_size / nbatch
    perplexity = math.exp(loss)
    logging.info('Iter[%d] %5s Loss:\t%.7f, Perplexity: %.7f' % \
                 (epoch, mode, loss, perplexity))
    summary_writer.add_scalar(tag="%s-Loss" % mode,
                              value=loss,
                              global_step=epoch)
    summary_writer.add_scalar(tag="%s-Perplexity" % mode,
                              value=perplexity,
                              global_step=epoch)

    return loss


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-15s %(message)s')

    args = parser.parse_args()

    logging.info(args)
    
    batch_size = args.batch_size
    bptt = args.bptt
    backend = args.backend

    # ==================================================================================================================
    # Data Preparation
    # ==================================================================================================================

    corpus = Corpus(args.dataset_dir, args.dataset_name)
    ntokens = len(corpus.dictionary)

    data_layout = 'TN' if backend == 'cudnn' else 'NT'

    train_data = CorpusIter(source=corpus.train, batch_size=batch_size, bptt=bptt, layout=data_layout)
    valid_data = CorpusIter(source=corpus.valid, batch_size=batch_size, bptt=bptt, layout=data_layout)
    test_data  = CorpusIter(source=corpus.test , batch_size=batch_size, bptt=bptt, layout=data_layout)

    # ==================================================================================================================
    # Training Model
    # ==================================================================================================================

    pred, states, state_names = rnn(bptt=bptt, vocab_size=ntokens, num_embed=args.emsize, nhid=args.nhid,
                                    num_layers=args.nlayers, dropout=args.dropout,
                                    batch_size=batch_size, tied=args.tied, backend=backend)
    loss = softmax_ce_loss(pred)

    # ==================================================================================================================
    # Training Module
    # ==================================================================================================================

    module = CustomStatefulModule(loss, states, state_names=state_names, context=mx.gpu())
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    module.init_params(initializer=mx.init.Xavier())
    optimizer = mx.optimizer.create('sgd',
                                    learning_rate=args.lr,
                                    rescale_grad=1.0/batch_size)
    module.init_optimizer(optimizer=optimizer)

    # ==================================================================================================================
    # Monitor
    # ==================================================================================================================

    try:
        import mxboard
    except ImportError:
        logging.error("Please install mxboard using `sudo -H pip install mxboard`.")

    summary_writer = mxboard.SummaryWriter('./log')
    speedometer = mx.callback.TensorboardSpeedometer(summary_writer=summary_writer,
                                                     batch_size=batch_size,
                                                     frequent=args.log_interval)
    logging.info("MXNet will be training using the RNN backend: %s.", backend)

    # ==================================================================================================================
    # Profiling
    # ==================================================================================================================

    if args.profiling:
        mx.profiler.profiler_set_config(mode='symbolic', filename='./log/%s' % args.profiler_output_fname)
        logging.info("Profiling has been enabled. MXNet will be profiling from iteration %s to %s." %
                     (args.profiling_start, args.profiling_end))
        assert args.profiling_end > args.profiling_start, \
            "Profiling start iteration must precede profiling end iteration."

    # ==================================================================================================================
    # Training
    # ==================================================================================================================

    logging.info("Training started ... ")

    global_step = 0

    for epoch in range(args.epochs):
        total_loss = 0.0
        nbatch = 0
        for batch in train_data:
            # profiling
            if args.profiling:
                if global_step == args.profiling_start:
                    mx.profiler.profiler_set_state('run')
                if global_step == args.profiling_end:
                    mx.profiler.profiler_set_state('stop')
            # train
            module.forward(batch)
            module.backward()
            module.update(max_norm=args.clip * bptt * batch_size)
            # update metric
            outputs = module.get_loss()
            total_loss += mx.nd.sum(outputs[0]).asscalar()
            speedometer_param = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                              eval_metric=None, locals=locals())
            speedometer(speedometer_param)
            if nbatch % args.log_interval == 0 and nbatch > 0:
                loss = total_loss / bptt / batch_size / args.log_interval
                perplexity = math.exp(loss)
                logging.info('Iter[%d] Batch [%d]\tLoss:  %.7f,\tPerplexity:\t%.7f' % \
                             (epoch, nbatch, loss, perplexity))
                summary_writer.add_scalar(tag="Loss",
                                          value=loss,
                                          global_step=global_step)
                summary_writer.add_scalar(tag="Perplexity",
                                          value=perplexity,
                                          global_step=global_step)
                total_loss = 0.0
            nbatch += 1
            global_step += 1
        # validation
        valid_loss = evaluate(valid_module=module, data_iter=valid_data,
                              epoch=epoch, mode='Valid',
                              bptt=bptt, batch_size=batch_size,
                              summary_writer=summary_writer)
        if valid_loss < best_loss:
            best_loss = valid_loss
            # test
            test_loss = evaluate(valid_module=module, data_iter=test_data,
                                 epoch=epoch, mode='Test',
                                 bptt=bptt, batch_size=batch_size,
                                 summary_writer=summary_writer)
        else:
            optimizer.lr *= 0.25
        train_data.reset()
    logging.info("Training completed.")
