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
# 
import os
import time
import argparse
import logging
import math
import random
import numpy as np
import mxnet as mx
from mxnet import gluon
from core.model import get_model
from core.dataset import NCFTrainData, NCFTestData
from core.evaluate import *


logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run matrix factorization with embedding",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', nargs='?', default='./data/',
                    help='Input data path.')
parser.add_argument('--dataset', nargs='?', default='ml-20m',
                    help='The dataset name.')
parser.add_argument('--batch-size', type=int, default=2048,
                    help='number of training examples per batch')
parser.add_argument('--eval-batch-size', type=int, default=1000,
                    help='number of evaluate examples per batch')                  
parser.add_argument('--model-type', type=str, default='neumf', choices=['neumf', 'gmf', 'mlp'],
                    help="mdoel type")
parser.add_argument('--num-negative', type=int, default=4,
                    help="number of negative samples per positive sample while training.")
parser.add_argument('--layers', default='[256, 256, 128, 64]',
                    help="list of number hiddens of fc layers in mlp model.")
parser.add_argument('--factor-size-gmf', type=int, default=64,
                    help="outdim of gmf embedding layers.")
parser.add_argument('--num-hidden', type=int, default=1,
                    help="num-hidden of neumf fc layer")
parser.add_argument('--log-interval', type=int, default=100,
                    help='logging interval')
parser.add_argument('--learning-rate', type=float, default=0.0005,
                    help='learning rate for optimizer')
parser.add_argument('--beta1', '-b1', type=float, default=0.9,
                    help='beta1 for Adam')
parser.add_argument('--beta2', '-b2', type=float, default=0.999,
                    help='beta1 for Adam')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='eps for Adam')
parser.add_argument('--topk', type=int, default=10,
                    help="topk for accuracy evaluation.")
parser.add_argument('--gpu', type=int, default=None,
                    help="list of gpus to run, e.g. 0 or 0,2. empty means using cpu().")
parser.add_argument('--workers', type=int, default=8, help='thread number for dataloader.')
parser.add_argument('--epoch', type=int, default=14, help='training epoch')
parser.add_argument('--seed', type=int, default=3, help='random seed to use. Default=3.')
parser.add_argument('--deploy', action='store_true', help="whether to load static graph for deployment")


def cross_entropy(label, pred, eps=1e-12):
    ce = 0
    for l, p in zip(label, pred):
        ce += -( l*np.log(p+eps) + (1-l)*np.log(1-p+eps))
    return ce

if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    model_type = args.model_type
    model_layers = eval(args.layers)
    factor_size_gmf = args.factor_size_gmf
    factor_size_mlp = int(model_layers[0]/2)
    num_hidden = args.num_hidden
    learning_rate=args.learning_rate
    beta1=args.beta1
    beta2=args.beta2
    eps=args.eps
    ctx = mx.cpu() if args.gpu is None else mx.gpu(args.gpu)
    topK = args.topk
    num_negatives = args.num_negative
    num_workers = args.workers
    epoch = args.epoch
    log_interval = args.log_interval

    # prepare dataset
    logging.info('Prepare Dataset')
    train_dataset = NCFTrainData((args.path + args.dataset + '/train-ratings.csv'), num_negatives)
    test_data = NCFTestData(args.path + args.dataset)
    train_dataloader = mx.gluon.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='rollover')
    logging.info('Prepare Dataset completed')
    # construct the model
    net = get_model(model_type, factor_size_mlp, factor_size_gmf, 
                    model_layers, num_hidden, train_dataset.nb_users, train_dataset.nb_items)

    # initialize the module
    mod = mx.module.Module(net, context=ctx, data_names=['user', 'item'], label_names=['softmax_label'])
    provide_data = [mx.io.DataDesc(name='item', shape=((batch_size,))),
                    mx.io.DataDesc(name='user', shape=((batch_size,)))]
    provide_label = [mx.io.DataDesc(name='softmax_label', shape=((batch_size,)))]
    mod.bind(for_training=True, data_shapes=provide_data, label_shapes=provide_label)
    mod.init_params()
    mod.init_optimizer(optimizer='adam', optimizer_params=[('learning_rate', learning_rate), ('beta1',beta1), ('beta2',beta2), ('epsilon',eps)])
    
    metric = mx.metric.create(cross_entropy)
    speedometer = mx.callback.Speedometer(batch_size, log_interval)
    best_hr, best_ndcg, best_iter = -1, -1, -1 
    logging.info('Training started ...')
    for epoch in range(epoch):
        metric.reset()
        for nbatch, seqs in enumerate(train_dataloader):
            user_id, item_id, labels = seqs
            batch = mx.io.DataBatch(data = [item_id.astype('int32').as_in_context(ctx),
                                            user_id.astype('int32').as_in_context(ctx)],
                                    label = [labels.as_in_context(ctx)])
            mod.forward(batch)
            mod.backward()
            mod.update()
            predicts=mod.get_outputs()[0]
            metric.update(labels = labels, preds = predicts)
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=metric, locals=locals())
            speedometer(speedometer_param)
        
        # save model
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, 'model', args.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        mod.save_checkpoint(os.path.join(model_path, model_type), epoch)
        # compute hit ratio
        (hits, ndcgs) = evaluate_model(mod, test_data.testRatings, test_data.testNegatives, topK, eval_batch_size, ctx, logging)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        logging.info('Iteration %d: HR = %.4f, NDCG = %.4f'  % (epoch, hr, ndcg))
        # best hit ratio
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch

    logging.info("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    logging.info('Training completed.')

