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
import mxnet as mx
import numpy as np
from data import get_movielens_iter, get_movielens_data
from model import matrix_fact_net
import os

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run matrix factorization with sparse embedding",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epoch', type=int, default=3,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128,
                    help='number of examples per batch')
parser.add_argument('--print-every', type=int, default=100,
                    help='logging frequency')
parser.add_argument('--factor-size', type=int, default=128,
                    help="the factor size of the embedding operation")
parser.add_argument('--use-dense', action='store_true',
                    help="use the dense embedding operator")
parser.add_argument('--use-gpu', action='store_true',
                    help="use gpu")
parser.add_argument('--dummy-iter', action='store_true',
                    help="use the dummy data iterator for speed test")

MOVIELENS = {
    'dataset': 'ml-10m',
    'train': './data/ml-10M100K/r1.train',
    'val': './data/ml-10M100K/r1.test',
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
    use_sparse = not args.use_dense
    factor_size = args.factor_size
    dummy_iter = args.dummy_iter
    print_every = args.print_every

    momentum = 0.9
    ctx = mx.gpu(0) if args.use_gpu else mx.cpu(0)
    learning_rate = 0.1

    # prepare dataset and iterators
    max_user = MOVIELENS['max_user']
    max_movies = MOVIELENS['max_movie']
    data_dir = os.path.join(os.getcwd(), 'data')
    get_movielens_data(data_dir, MOVIELENS['dataset'])
    train_iter = get_movielens_iter(MOVIELENS['train'], batch_size, dummy_iter)
    val_iter = get_movielens_iter(MOVIELENS['val'], batch_size, dummy_iter)

    # construct the model
    net = matrix_fact_net(factor_size, factor_size, max_user, max_movies, sparse_embed=use_sparse)

    # initialize the module
    mod = mx.module.Module(symbol=net, context=ctx, data_names=['user', 'item'],
                           label_names=['score'])
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    optim = mx.optimizer.create(optimizer, learning_rate=learning_rate, momentum=momentum,
                                wd=1e-4, rescale_grad=1.0/batch_size)
    mod.init_optimizer(optimizer=optim)
    # use MSE as the metric
    metric = mx.metric.create(['MSE'])
    speedometer = mx.callback.Speedometer(batch_size, print_every)
    logging.info('Training started ...')
    for epoch in range(num_epoch):
        nbatch = 0
        metric.reset()
        for batch in train_iter:
            nbatch += 1
            mod.forward_backward(batch)
            # update all parameters
            mod.update()
            # update training metric
            mod.update_metric(metric, batch.label)
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=metric, locals=locals())
            speedometer(speedometer_param)
        # evaluate metric on validation dataset
        score = mod.score(val_iter, ['MSE'])
        logging.info('epoch %d, eval MSE = %s ' % (epoch, score[0][1]))
        # reset the iterator for next pass of data
        train_iter.reset()
        val_iter.reset()
    logging.info('Training completed.')
