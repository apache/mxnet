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

# pylint: disable=missing-docstring, deprecated-module
from __future__ import print_function

import logging
from optparse import OptionParser

import mxnet as mx
from nce import NceAuc
from text8_data import DataIterSubWords
from wordvec_net import get_subword_net


head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=head)

EMBEDDING_SIZE = 100
BATCH_SIZE = 256
NUM_LABEL = 5
NUM_EPOCH = 20
MIN_COUNT = 5  # only works when doing nagative sampling, keep it same as nce-loss
GRAMS = 3      # here we use triple-letter representation
MAX_SUBWORDS = 10
PADDING_CHAR = '</s>'


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    parser = OptionParser()
    parser.add_option("-g", "--gpu", action="store_true", dest="gpu", default=False,
                      help="use gpu")
    options, args = parser.parse_args()

    batch_size = BATCH_SIZE
    num_label = NUM_LABEL
    embedding_size = EMBEDDING_SIZE

    data_train = DataIterSubWords(
        "./data/text8",
        batch_size=batch_size,
        num_label=num_label,
        min_count=MIN_COUNT,
        gram=GRAMS,
        max_subwords=MAX_SUBWORDS,
        padding_char=PADDING_CHAR)

    network = get_subword_net(data_train.vocab_size, num_label - 1, embedding_size)

    devs = mx.cpu()
    if options.gpu:
        devs = mx.gpu()

    model = mx.mod.Module(
        symbol=network,
        data_names=[x[0] for x in data_train.provide_data],
        label_names=[y[0] for y in data_train.provide_label],
        context=[devs]
    )

    print("Training on {}".format("GPU" if options.gpu else "CPU"))
    metric = NceAuc()
    model.fit(
        train_data=data_train,
        num_epoch=NUM_EPOCH,
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.3, 'momentum': 0.9, 'wd': 0.0000},
        initializer=mx.init.Xavier(factor_type='in', magnitude=2.34),
        eval_metric=metric,
        batch_end_callback=mx.callback.Speedometer(batch_size, 50)
    )
