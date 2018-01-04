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
from text8_data import DataIterWords
from wordvec_net import get_word_net


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    parser = OptionParser()
    parser.add_option("-g", "--gpu", action="store_true", dest="gpu", default=False,
                      help="use gpu")
    options, args = parser.parse_args()

    batch_size = 256
    num_label = 5

    data_train = DataIterWords("./data/text8", batch_size, num_label)

    network = get_word_net(data_train.vocab_size, num_label - 1)

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
        num_epoch=20,
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.3, 'momentum': 0.9, 'wd': 0.0000},
        initializer=mx.init.Xavier(factor_type='in', magnitude=2.34),
        eval_metric=metric,
        batch_end_callback=mx.callback.Speedometer(batch_size, 50)
    )
