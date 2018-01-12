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
import run_utils
from data import Corpus, CorpusIter, DummyIter, MultiSentenceIter
from model import *
from sampler import *
from sparse_module import SparseModule
import os, math, logging, time, pickle
import data_utils


def evaluate(mod, data_iter, epoch, log_interval, early_stop=None):
    import time
    start = time.time()
    total_L = 0.0
    nbatch = 0
    mod.set_states(value=0)
    for batch in data_iter:
        mod.forward(batch, is_train=False)
        outputs = mod.get_outputs(merge_multi_context=False)
        states = outputs[:-1]
        total_L += outputs[-1][0].asscalar()
        mod.set_states(states=states)
        nbatch += 1
        if (nbatch + 1) % log_interval == 0:
            logging.info("eval batch %d : %.7f" % (nbatch, total_L / nbatch))
        if (nbatch + 1) == early_stop:
            break
    data_iter.reset()
    loss = total_L / nbatch
    try:
        ppl = math.exp(loss) if loss < 100 else -1
    except Exception:
        ppl = 1e37
    end = time.time()
    logging.info('Iter[%d]\t\t CE loss %.7f, ppl %.7f. Time cost = %.2f seconds'%(epoch, loss, ppl, end - start))
    return loss

if __name__ == '__main__':
    parser = run_utils.get_parser(is_train=False)
    args = parser.parse_args()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info(args)

    # data
    vocab = data_utils.Vocabulary.from_file(args.vocab)
    unigram = vocab.unigram()
    ntokens = unigram.size

    train_data = mx.io.PrefetchingIter(MultiSentenceIter(args.data if not args.bench else "./data/ptb.tiny.txt", vocab,
                                      args.batch_size, args.bptt))
    eval_data = mx.io.PrefetchingIter(MultiSentenceIter(args.data if not args.bench else "./data/ptb.tiny.txt", vocab,
                                      args.eval_size, args.bptt))

    rnn_module = RNNModel(args.bptt, ntokens, args.emsize, args.nhid, args.nlayers,
                          args.dropout, args.num_proj)

    extra_states = ['sample', 'p_noise_sample', 'p_noise_target']
    state_names = rnn_module.state_names
    sparse_params=['encoder_weight', 'decoder_weight', 'decoder_bias']
    data_names = ['data', 'mask']
    label_names = ['label']
    epoch = 0
    while True:
        nce_mod = SparseModule.load(args.checkpoint_dir, 0, context=mx.cpu(), state_names=(state_names + extra_states),
                                    data_names=data_names, label_names=label_names, sparse_params=sparse_params)
        nce_mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)

        ############### eval model ####################
        eval_rnn_out, eval_last_states = rnn_module.forward(args.eval_size)
        eval_model = ce_loss(eval_rnn_out, ntokens, args.dense)
        eval_last_states.append(eval_model)
        ############### eval module ####################
        eval_module = SparseModule(symbol=mx.sym.Group(eval_last_states), context=mx.cpu(), data_names=data_names,
                                   label_names=label_names, state_names=state_names, sparse_params=sparse_params)
        eval_module.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label, shared_module=nce_mod, for_training=False)
        val_L = evaluate(eval_module, eval_data, epoch, args.log_interval)
