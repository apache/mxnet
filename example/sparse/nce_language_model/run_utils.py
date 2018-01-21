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

def _add_train_args(parser):
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='wd')
    parser.add_argument('--clip', type=float, default=0.2,
                        help='gradient clipping by global norm')
    parser.add_argument('--init', type=float, default=1,
                        help='init value for adagrad')
    parser.add_argument('--clip-lstm', action='store_true',
                        help='only clip lstm layers')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='checkpoint every x epochs')
    # TODO change default value
    parser.add_argument('--load-epoch', type=int, default=-1,
                        help='load epoch')
    parser.add_argument('--py-sampler', action='store_true',
                        help='use alternative sampler')
    parser.add_argument('--rescale-embed', action='store_true',
                        help='rescale-embedding-grad')
    return parser

def _add_eval_args(parser):
    parser.add_argument('--eval-every', type=int, default=1,
                        help='evalutaion every x epochs')
    parser.add_argument('--eval_size', type=int, default=32,
                        help='batch size')
    return parser

def get_parser(is_train=True):
    parser = argparse.ArgumentParser(description='Language Model on GBW')
    parser.add_argument('--data', type=str, default='./data/ptb.train.txt',
                        help='location of the data corpus')
    parser.add_argument('--vocab', type=str, default='./data/ptb_vocab.txt',
                        help='location of the corpus vocab')
    parser.add_argument('--emsize', type=int, default=1500,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1500,
                        help='number of hidden units per layer')
    parser.add_argument('--num_proj', type=int, default=0,
                        help='number of projection units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--epochs', type=int, default=5,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size per gpu')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--eps', type=float, default=1e-16,
                        help='eps for adagrad')
    parser.add_argument('--bptt', type=int, default=20,
                        help='sequence length')
    parser.add_argument('--k', type=int, default=8192,
                        help='number of noise samples to estimate')
    parser.add_argument('--gpus', type=str,
                        help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
    parser.add_argument('--dense', action='store_true',
                        help='use dense embedding instead of sparse embedding')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--profile', action='store_true',
                        help='whether to use profiler')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='type of kv-store to use')
    parser.add_argument('--bench', action='store_true',
                        help='whether to use tiny data')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoint/',
                        help='dir for checkpoint')
    parser = _add_train_args(parser) if is_train else _add_eval_args(parser)
    return parser
