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

from __future__ import division

import argparse
import logging
import time

import numpy as np
from bottleneck import argpartition

import mxnet as mx
from data import cub200_iterator
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet import autograd as ag, nd
from model import MarginNet, MarginLoss

logging.basicConfig(level=logging.INFO)

# CLI
parser = argparse.ArgumentParser(description='train a model for image classification.')
parser.add_argument('--data-path', type=str, default='data/CUB_200_2011',
                    help='path of data.')
parser.add_argument('--embed-dim', type=int, default=128,
                    help='dimensionality of image embedding. default is 128.')
parser.add_argument('--batch-size', type=int, default=70,
                    help='training batch size per device (CPU/GPU). default is 70.')
parser.add_argument('--batch-k', type=int, default=5,
                    help='number of images per class in a batch. default is 5.')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to use, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs. default is 20.')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer. default is adam.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate. default is 0.0001.')
parser.add_argument('--lr-beta', type=float, default=0.1,
                    help='learning rate for the beta in margin based loss. default is 0.1.')
parser.add_argument('--margin', type=float, default=0.2,
                    help='margin for the margin based loss. default is 0.2.')
parser.add_argument('--beta', type=float, default=1.2,
                    help='initial value for beta. default is 1.2.')
parser.add_argument('--nu', type=float, default=0.0,
                    help='regularization parameter for beta. default is 0.0.')
parser.add_argument('--factor', type=float, default=0.5,
                    help='learning rate schedule factor. default is 0.5.')
parser.add_argument('--steps', type=str, default='12,14,16,18',
                    help='epochs to update learning rate. default is 12,14,16,18.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. default=123.')
parser.add_argument('--model', type=str, default='resnet50_v2',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--save-model-prefix', type=str, default='margin_loss_model',
                    help='prefix of models to be saved.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer.')
parser.add_argument('--log-interval', type=int, default=20,
                    help='number of batches to wait before logging.')
opt = parser.parse_args()

logging.info(opt)

# Settings.
mx.random.seed(opt.seed)
np.random.seed(opt.seed)

batch_size = opt.batch_size

gpus = [] if opt.gpus is None or opt.gpus is '' else [
    int(gpu) for gpu in opt.gpus.split(',')]
num_gpus = len(gpus)

batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in gpus] if num_gpus > 0 else [mx.cpu()]
steps = [int(step) for step in opt.steps.split(',')]

# Construct model.
kwargs = {'ctx': context, 'pretrained': opt.use_pretrained}
net = models.get_model(opt.model, **kwargs)

if opt.use_pretrained:
    # Use a smaller learning rate for pre-trained convolutional layers.
    for v in net.collect_params().values():
        if 'conv' in v.name:
            setattr(v, 'lr_mult', 0.01)

net.hybridize()
net = MarginNet(net.features, opt.embed_dim, opt.batch_k)
beta = mx.gluon.Parameter('beta', shape=(100,))

# Get iterators.
train_data, val_data = cub200_iterator(opt.data_path, opt.batch_k, batch_size, (3, 224, 224))


def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = nd.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * nd.dot(x, x.transpose()))
    return nd.sqrt(distance_square)


def evaluate_emb(emb, labels):
    """Evaluate embeddings based on Recall@k."""
    d_mat = get_distance_matrix(emb)
    d_mat = d_mat.asnumpy()
    labels = labels.asnumpy()

    names = []
    accs = []
    for k in [1, 2, 4, 8, 16]:
        names.append('Recall@%d' % k)
        correct, cnt = 0.0, 0.0
        for i in range(emb.shape[0]):
            d_mat[i, i] = 1e10
            nns = argpartition(d_mat[i], k)[:k]
            if any(labels[i] == labels[nn] for nn in nns):
                correct += 1
            cnt += 1
        accs.append(correct/cnt)
    return names, accs


def test(ctx):
    """Test a model."""
    val_data.reset()
    outputs = []
    labels = []
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        for x in data:
            outputs.append(net(x)[-1])
        labels += label

    outputs = nd.concatenate(outputs, axis=0)[:val_data.n_test]
    labels = nd.concatenate(labels, axis=0)[:val_data.n_test]
    return evaluate_emb(outputs, labels)


def get_lr(lr, epoch, steps, factor):
    """Get learning rate based on schedule."""
    for s in steps:
        if epoch >= s:
            lr *= factor
    return lr


def train(epochs, ctx):
    """Training function."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)

    opt_options = {'learning_rate': opt.lr, 'wd': opt.wd}
    if opt.optimizer == 'sgd':
        opt_options['momentum'] = 0.9
    if opt.optimizer == 'adam':
        opt_options['epsilon'] = 1e-7
    trainer = gluon.Trainer(net.collect_params(), opt.optimizer,
                            opt_options,
                            kvstore=opt.kvstore)
    if opt.lr_beta > 0.0:
        # Jointly train class-specific beta.
        # See "sampling matters in deep embedding learning" paper for details.
        beta.initialize(mx.init.Constant(opt.beta), ctx=ctx)
        trainer_beta = gluon.Trainer([beta], 'sgd',
                                     {'learning_rate': opt.lr_beta, 'momentum': 0.9},
                                     kvstore=opt.kvstore)

    loss = MarginLoss(margin=opt.margin, nu=opt.nu)

    best_val = 0.0
    for epoch in range(epochs):
        tic = time.time()
        prev_loss, cumulative_loss = 0.0, 0.0

        # Learning rate schedule.
        trainer.set_learning_rate(get_lr(opt.lr, epoch, steps, opt.factor))
        logging.info('Epoch %d learning rate=%f', epoch, trainer.learning_rate)
        if opt.lr_beta > 0.0:
            trainer_beta.set_learning_rate(get_lr(opt.lr_beta, epoch, steps, opt.factor))
            logging.info('Epoch %d beta learning rate=%f', epoch, trainer_beta.learning_rate)

        # Inner training loop.
        for i in range(200):
            batch = train_data.next()
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)

            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    a_indices, anchors, positives, negatives, _ = net(x)

                    if opt.lr_beta > 0.0:
                        L = loss(anchors, positives, negatives, beta, y[a_indices])
                    else:
                        L = loss(anchors, positives, negatives, opt.beta, None)

                    # Store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    cumulative_loss += nd.mean(L).asscalar()

                for L in Ls:
                    L.backward()

            # Update.
            trainer.step(batch.data[0].shape[0])
            if opt.lr_beta > 0.0:
                trainer_beta.step(batch.data[0].shape[0])

            if (i+1) % opt.log_interval == 0:
                logging.info('[Epoch %d, Iter %d] training loss=%f' % (
                    epoch, i+1, cumulative_loss - prev_loss))
                prev_loss = cumulative_loss

        logging.info('[Epoch %d] training loss=%f'%(epoch, cumulative_loss))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))

        names, val_accs = test(ctx)
        for name, val_acc in zip(names, val_accs):
            logging.info('[Epoch %d] validation: %s=%f'%(epoch, name, val_acc))

        if val_accs[0] > best_val:
            best_val = val_accs[0]
            logging.info('Saving %s.' % opt.save_model_prefix)
            net.save_parameters('%s.params' % opt.save_model_prefix)
    return best_val


if __name__ == '__main__':
    best_val_recall = train(opt.epochs, context)
    print('Best validation Recall@1: %.2f.' % best_val_recall)
