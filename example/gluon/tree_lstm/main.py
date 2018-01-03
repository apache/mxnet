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

# This example is inspired by https://github.com/dasguptar/treelstm.pytorch
import argparse, math, os, random
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag

from tree_lstm import SimilarityTreeLSTM
from dataset import Vocab, SICKDataIter

parser = argparse.ArgumentParser(description='TreeLSTM for Sentence Similarity on Dependency Trees')
parser.add_argument('--data', default='data/sick/',
                    help='path to raw dataset. required when preprocessed dataset is not available.')
parser.add_argument('--word_embed', default='data/glove/glove.840B.300d.txt',
                    help='directory with word embeddings. required when preprocessed dataset is not available.')
parser.add_argument('--batch_size', type=int, default=25,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr', default=0.02, type=float,
                    help='initial learning rate')
parser.add_argument('--wd', default=0.0001, type=float,
                    help='weight decay factor')
parser.add_argument('--optimizer', default='adagrad',
                    help='optimizer (default: adagrad)')
parser.add_argument('--seed', default=123, type=int,
                    help='random seed (default: 123)')
parser.add_argument('--use-gpu', action='store_true',
                    help='whether to use GPU.')

opt = parser.parse_args()

logging.info(opt)

context = [mx.gpu(0) if opt.use_gpu else mx.cpu()]

rnn_hidden_size, sim_hidden_size, num_classes = 150, 50, 5
optimizer = opt.optimizer.lower()

mx.random.seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

batch_size = opt.batch_size

# read dataset
if os.path.exists('dataset.pickle'):
    with open('dataset.pickle', 'rb') as f:
        train_iter, dev_iter, test_iter, vocab = pickle.load(f)
else:
    root_dir = opt.data
    segments = ['train', 'dev', 'test']
    token_files = [os.path.join(root_dir, seg, '%s.toks'%tok)
                   for tok in ['a', 'b']
                   for seg in segments]

    vocab = Vocab(filepaths=token_files, embedpath=opt.word_embed)

    train_iter, dev_iter, test_iter = [SICKDataIter(os.path.join(root_dir, segment), vocab, num_classes)
                                       for segment in segments]
    with open('dataset.pickle', 'wb') as f:
        pickle.dump([train_iter, dev_iter, test_iter, vocab], f)

logging.info('==> SICK vocabulary size : %d ' % vocab.size)
logging.info('==> Size of train data   : %d ' % len(train_iter))
logging.info('==> Size of dev data     : %d ' % len(dev_iter))
logging.info('==> Size of test data    : %d ' % len(test_iter))

# get network
net = SimilarityTreeLSTM(sim_hidden_size, rnn_hidden_size, vocab.size, vocab.embed.shape[1], num_classes)

# use pearson correlation and mean-square error for evaluation
metric = mx.metric.create(['pearsonr', 'mse'])

def to_target(x):
    target = np.zeros((1, num_classes))
    ceil = int(math.ceil(x))
    floor = int(math.floor(x))
    if ceil==floor:
        target[0][floor-1] = 1
    else:
        target[0][floor-1] = ceil - x
        target[0][ceil-1] = x - floor
    return mx.nd.array(target)

def to_score(x):
    levels = mx.nd.arange(1, 6, ctx=x.context)
    return [mx.nd.sum(levels*mx.nd.exp(x), axis=1).reshape((-1,1))]

# when evaluating in validation mode, check and see if pearson-r is improved
# if so, checkpoint and run evaluation on test dataset
def test(ctx, data_iter, best, mode='validation', num_iter=-1):
    data_iter.reset()
    batches = len(data_iter)
    data_iter.set_context(ctx[0])
    preds = []
    labels = [mx.nd.array(data_iter.labels, ctx=ctx[0]).reshape((-1,1))]
    for _ in tqdm(range(batches), desc='Testing in {} mode'.format(mode)):
        l_tree, l_sent, r_tree, r_sent, label = data_iter.next()
        z = net(mx.nd, l_sent, r_sent, l_tree, r_tree)
        preds.append(z)

    preds = to_score(mx.nd.concat(*preds, dim=0))
    metric.update(preds, labels)
    names, values = metric.get()
    metric.reset()
    for name, acc in zip(names, values):
        logging.info(mode+' acc: %s=%f'%(name, acc))
        if name == 'pearsonr':
            test_r = acc
    if mode == 'validation' and num_iter >= 0:
        if test_r >= best:
            best = test_r
            logging.info('New optimum found: {}. Checkpointing.'.format(best))
            net.collect_params().save('childsum_tree_lstm_{}.params'.format(num_iter))
            test(ctx, test_iter, -1, 'test')
        return best


def train(epoch, ctx, train_data, dev_data):

    # initialization with context
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx[0])
    net.embed.weight.set_data(vocab.embed.as_in_context(ctx[0]))
    train_data.set_context(ctx[0])
    dev_data.set_context(ctx[0])

    # set up trainer for optimizing the network.
    trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': opt.lr, 'wd': opt.wd})

    best_r = -1
    Loss = gluon.loss.KLDivLoss()
    for i in range(epoch):
        train_data.reset()
        num_batches = len(train_data)
        # collect predictions and labels for evaluation metrics
        preds = []
        labels = [mx.nd.array(train_data.labels, ctx=ctx[0]).reshape((-1,1))]
        for j in tqdm(range(num_batches), desc='Training epoch {}'.format(i)):
            # get next batch
            l_tree, l_sent, r_tree, r_sent, label = train_data.next()
            # use autograd to record the forward calculation
            with ag.record():
                # forward calculation. the output is log probability
                z = net(mx.nd, l_sent, r_sent, l_tree, r_tree)
                # calculate loss
                loss = Loss(z, to_target(label).as_in_context(ctx[0]))
                # backward calculation for gradients.
                loss.backward()
                preds.append(z)
            # update weight after every batch_size samples
            if (j+1) % batch_size == 0:
                trainer.step(batch_size)

        # translate log-probability to scores, and evaluate
        preds = to_score(mx.nd.concat(*preds, dim=0))
        metric.update(preds, labels)
        names, values = metric.get()
        metric.reset()
        for name, acc in zip(names, values):
            logging.info('training acc at epoch %d: %s=%f'%(i, name, acc))
        best_r = test(ctx, dev_data, best_r, num_iter=i)

train(opt.epochs, context, train_iter, dev_iter)
