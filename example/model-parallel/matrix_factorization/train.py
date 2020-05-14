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
import logging
import time
import mxnet as mx
import numpy as np
from get_data import get_movielens_iter, get_movielens_data
from model import matrix_fact_model_parallel_net


logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run model parallel version of matrix factorization",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epoch', type=int, default=3,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256,
                    help='number of examples per batch')
parser.add_argument('--print-every', type=int, default=100,
                    help='logging interval')
parser.add_argument('--factor-size', type=int, default=128,
                    help="the factor size of the embedding operation")
parser.add_argument('--num-gpus', type=int, default=2,
                    help="number of gpus to use")

MOVIELENS = {
    'dataset': 'ml-10m',
    'train': './ml-10M100K/r1.train',
    'val': './ml-10M100K/r1.test',
    'max_user': 71569,
    'max_movie': 65135,
}

if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    optimizer = 'sgd'
    factor_size = args.factor_size
    print_every = args.print_every
    num_gpus = args.num_gpus    
 
    momentum = 0.9
    learning_rate = 0.1

    # prepare dataset and iterators
    max_user = MOVIELENS['max_user']
    max_movies = MOVIELENS['max_movie']
    get_movielens_data(MOVIELENS['dataset'])
    train_iter = get_movielens_iter(MOVIELENS['train'], batch_size)
    val_iter = get_movielens_iter(MOVIELENS['val'], batch_size)

    # construct the model
    net = matrix_fact_model_parallel_net(factor_size, factor_size, max_user, max_movies)

    # construct the module
    # map the ctx_group attribute to the context assignment
    group2ctxs={'dev1':[mx.cpu()]*num_gpus, 'dev2':[mx.gpu(i) for i in range(num_gpus)]}

    # Creating a module by passing group2ctxs attribute which maps
    # the ctx_group attribute to the context assignment
    mod = mx.module.Module(symbol=net, context=[mx.cpu()]*num_gpus, data_names=['user', 'item'],
        label_names=['score'], group2ctxs=group2ctxs)
    
    # the initializer used to initialize the parameters
    initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    
    # the parameters for the optimizer constructor
    optimizer_params = {
        'learning_rate': learning_rate,
        'wd': 1e-4,
        'momentum': momentum,
        'rescale_grad': 1.0/batch_size}

    # use MSE as the metric
    metric = mx.metric.create(['MSE'])
    
    speedometer = mx.callback.Speedometer(batch_size, print_every)
    
    # start training
    mod.fit(train_iter,
            val_iter,
            eval_metric        = metric,
            num_epoch          = num_epoch,
            optimizer          = optimizer,
            optimizer_params   = optimizer_params,
            initializer        = initializer,
            batch_end_callback = speedometer) 
