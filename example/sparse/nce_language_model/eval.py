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
from data import Corpus, CorpusIter, DummyIter, MultiSentenceIter
from model import *
from sampler import *
from sparse_module import SparseModule
import os, math, logging, time, pickle
import data_utils

parser = argparse.ArgumentParser(description='PennTreeBank LSTM Language Model with Noice Contrastive Estimation')
parser.add_argument('--train-data', type=str, default='./data/ptb.train.txt',
                    help='location of the data corpus')
parser.add_argument('--eval-data', type=str, default='./data/ptb.valid.txt',
                    help='location of the data corpus')
parser.add_argument('--vocab', type=str, default='./data/ptb_vocab.txt',
                    help='location of the corpus vocab')
parser.add_argument('--emsize', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1500,
                    help='number of hidden units per layer')
parser.add_argument('--num_proj', type=int, default=0,
                    help='number of projection units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--mom', type=float, default=0.0,
                    help='mom')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1')
parser.add_argument('--wd', type=float, default=0.0,
                    help='wd')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping by global norm')
parser.add_argument('--epochs', type=int, default=6000,
                    help='upper epoch limit')
parser.add_argument('--eval-every', type=int, default=1,
                    help='evalutaion every x epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size per gpu')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--k', type=int, default=15,
                    help='number of noise samples to estimate')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--dense', action='store_true',
                    help='use dense embedding instead of sparse embedding')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--lr-decay', type=float, default=0.25,
                    help='learning rate decay')
parser.add_argument('--minlr', type=float, default=0.00001,
                    help='min learning rate')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='which optimizer to use')
parser.add_argument('--profile', action='store_true',
                    help='whether to use profiler')
parser.add_argument('--kvstore', type=str, default='device',
                    help='type of kv-store to use')
parser.add_argument('--init', type=str, default='uniform',
                    help='type of initialization for embed and softmax weight')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoint/',
                    help='dir for checkpoint')
parser.add_argument('--bench', action='store_true',
                    help='whether to use tiny data')
parser.add_argument('--clip-lstm', action='store_true',
                    help='only clip lstm layers')
parser.add_argument('--tf-nce', action='store_true',
                    help='use tf nce impl')
args = parser.parse_args()

def evaluate(mod, data_iter, epoch, mode, args):
    import time
    start = time.time()
    total_L = 0.0
    nbatch = 0
    mod.set_states(value=0)
    for batch in data_iter:
        mod.forward(batch, is_train=False)
        outputs = mod.get_outputs(merge_multi_context=False)
        state_cache = outputs[:-1]
        # (args.batch_size * args.bptt)
        for g in range(1):
            total_L += mx.nd.sum(outputs[-1][g]).asscalar()
        mod.set_states(states=state_cache)
        nbatch += 1
        logging.info("eval batch %d : %.7f" % (nbatch, total_L / args.bptt / 1 / nbatch))
    data_iter.reset()
    loss = total_L / args.bptt / 1 / nbatch
    try:
        ppl = math.exp(loss)
    except Exception:
        ppl = -1
    end = time.time()
    logging.info('Iter[%d] %s\t\tloss %.7f, ppl %.7f. Cost = %.2f'%(epoch, mode, loss, ppl, end - start))
    return loss


if __name__ == '__main__':
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    args = parser.parse_args()
    logging.info(args)
    if args.init == 'uniform':
        init = mx.init.Uniform(0.1)
    elif args.init == 'uniform_unit':
        init = mx.init.UniformUnitScaling()
    else:
        raise NotImplementedError()

    # data
    vocab = data_utils.Vocabulary.from_file(args.vocab)
    unigram = vocab.unigram()
    ntokens = unigram.size

    train_data = mx.io.PrefetchingIter(MultiSentenceIter(args.train_data if not args.bench else "./data/ptb.tiny.txt", vocab,
                                      args.batch_size, args.bptt))
    eval_data = mx.io.PrefetchingIter(MultiSentenceIter(args.eval_data if not args.bench else "./data/ptb.tiny.txt", vocab,
                                      1, args.bptt))

    extra_states = ['sample', 'p_noise_sample', 'p_noise_target']
    state_names = ['lstm_l0_0', 'lstm_l0_1', 'lstm_l1_0', 'lstm_l1_1'] if args.nlayers == 2 else ['lstm_l0_0', 'lstm_l0_1']
    sparse_params=['encoder_weight', 'decoder_weight', 'decoder_bias']
    data_names = ['data', 'mask']
    label_names = ['label']
    epoch = 0
    while True:
        nce_mod = SparseModule.load(args.checkpoint_dir, 0, context=mx.cpu(), state_names=(state_names + extra_states),
                                    data_names=data_names, label_names=label_names, sparse_params=sparse_params)
        nce_mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)

        ############### eval model ####################
        eval_rnn_out, eval_last_states = rnn(args.bptt, ntokens, args.emsize, args.nhid, # 0 for dropout at test
                                             args.nlayers, 0, args.dense, 1, init, args.num_proj)
        eval_model = ce_loss(eval_rnn_out, ntokens, args.dense)
        eval_last_states.append(eval_model)
        ############### eval module ####################
        eval_module = SparseModule(symbol=mx.sym.Group(eval_last_states), context=mx.cpu(), data_names=data_names,
                                   label_names=label_names, state_names=state_names, sparse_params=sparse_params)
        eval_module.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label, shared_module=nce_mod, for_training=False)
        val_L = evaluate(eval_module, eval_data, epoch, 'Valid', args)
        eval_data.reset()
