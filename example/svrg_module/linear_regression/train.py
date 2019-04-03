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
import mxnet as mx
from common import create_lin_reg_network, create_logger
from data_reader import get_year_prediction_data

parser = argparse.ArgumentParser()
parser.add_argument('-e', dest='epochs', help='number of epochs for training phase', type=int, default=100)
parser.add_argument('-f', dest="updateFreq", help="update frequency for SVRGModule", type=int, default=2)
parser.add_argument('-b', dest="batch_size", help="define the batch size for training", type=int,
                    default=100, required=False)
parser.add_argument('-m', dest='metrics', help="create eval metric", type=str, default='mse')
parser.add_argument('--gpus', type=str, help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
parser.add_argument('--kv-store', type=str, default='local', help='key-value store type')

args = parser.parse_args()
# devices for training
ctx = mx.cpu() if args.gpus is None or args.gpus == "" else [mx.gpu(int(i)) for i in args.gpus.split(',')]

logger = create_logger()
kv = mx.kvstore.create(args.kv_store)

feature_dim, train_features, train_labels = get_year_prediction_data()
train_iter, mod = create_lin_reg_network(train_features, train_labels, feature_dim, args.batch_size, args.updateFreq,
                                         ctx, logger)

mod.fit(train_iter, eval_metric='mse', optimizer='sgd',
        optimizer_params=(('learning_rate', 0.025), ), num_epoch=args.epochs, kvstore=kv)
