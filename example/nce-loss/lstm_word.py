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
from nce import NceLSTMAuc
from text8_data import DataIterLstm
from lstm_net import get_lstm_net


if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    parser = OptionParser()
    parser.add_option("-g", "--gpu", action="store_true", dest="gpu", default=False,
                      help="use gpu")
    options, args = parser.parse_args()

    batch_size = 1024
    seq_len = 5
    num_label = 6
    num_lstm_layer = 2
    num_hidden = 100

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = DataIterLstm("./data/text8", batch_size, seq_len, num_label, init_states)

    network = get_lstm_net(data_train.vocab_size, seq_len, num_lstm_layer, num_hidden)

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
    metric = NceLSTMAuc()
    model.fit(
        train_data=data_train,
        num_epoch=20,
        optimizer='sgd',
        optimizer_params={'learning_rate': 0.3, 'momentum': 0.9, 'wd': 0.0000},
        initializer=mx.init.Xavier(factor_type='in', magnitude=2.34),
        eval_metric=metric,
        batch_end_callback=mx.callback.Speedometer(batch_size, 50)
    )
