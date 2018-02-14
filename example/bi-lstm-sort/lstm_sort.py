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

# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import os
import sys
import numpy as np
import mxnet as mx
import random
import argparse

from lstm import bi_lstm_unroll
from sort_io import BucketSentenceIter, default_build_vocab

import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


TRAIN_FILE = "sort.train.txt"
TEST_FILE = "sort.test.txt"
VALID_FILE = "sort.valid.txt"
DATA_DIR = os.path.join(os.getcwd(), "data")
SEQ_LEN = 5

def gen_data(seq_len, start_range, end_range):
    if not os.path.exists(DATA_DIR):
        try:
            logging.info('create directory %s', DATA_DIR)
            os.makedirs(DATA_DIR)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise OSError('failed to create ' + DATA_DIR)
    vocab = [str(x) for x in range(start_range, end_range)]
    sw_train = open(os.path.join(DATA_DIR, TRAIN_FILE), "w")
    sw_test = open(os.path.join(DATA_DIR, TEST_FILE), "w")
    sw_valid = open(os.path.join(DATA_DIR, VALID_FILE), "w")

    for i in range(1000000):
        seq = " ".join([vocab[random.randint(0, len(vocab) - 1)] for j in range(seq_len)])
        k = i % 50
        if k == 0:
            sw_test.write(seq + "\n")
        elif k == 1:
            sw_valid.write(seq + "\n")
        else:
            sw_train.write(seq + "\n")

    sw_train.close()
    sw_test.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Parse args for lstm_sort example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start-range', type=int, default=100,
                        help='starting number of the range')
    parser.add_argument('--end-range', type=int, default=1000,
                        help='Ending number of the range')
    parser.add_argument('--cpu', action='store_true',
                        help='To use CPU for training')
    return parser.parse_args()


def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

def main():
    args = parse_args()
    gen_data(SEQ_LEN, args.start_range, args.end_range)
    batch_size = 100
    buckets = []
    num_hidden = 300
    num_embed = 512
    num_lstm_layer = 2

    num_epoch = 1
    learning_rate = 0.1
    momentum = 0.9

    if args.cpu:
        contexts = [mx.context.cpu(i) for i in range(1)]
    else:
        contexts = [mx.context.gpu(i) for i in range(1)]

    vocab = default_build_vocab(os.path.join(DATA_DIR, TRAIN_FILE))

    def sym_gen(seq_len):
        return bi_lstm_unroll(seq_len, len(vocab),
                              num_hidden=num_hidden, num_embed=num_embed,
                              num_label=len(vocab))

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = BucketSentenceIter(os.path.join(DATA_DIR, TRAIN_FILE), vocab,
                                    buckets, batch_size, init_states)
    data_val = BucketSentenceIter(os.path.join(DATA_DIR, VALID_FILE), vocab,
                                  buckets, batch_size, init_states)

    if len(buckets) == 1:
        symbol = sym_gen(buckets[0])
    else:
        symbol = sym_gen

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)

    model.save("sort")

if __name__ == '__main__':
    sys.exit(main())
