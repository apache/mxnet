
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Function used for auxiliary training
# author:kenjewu

import numpy as np
from time import time

import mxnet as mx
from mxnet import autograd, gluon, nd
from sklearn.metrics import accuracy_score, f1_score


def train(
        data_iter_train, data_iter_valid, model, loss, trainer, CTX, num_epochs, penal_coeff=0.0, clip=None,
        class_weight=None, loss_name='wsce'):
    '''
    Function used in training
    Args:
        data_iter_train: the iter of training data
        data_iter_valid: the iter of validation data
        model: model to train
        loss: loss function
        trainer: the way of train
        CTX: context
        num_epochs: number of total epochs
        penal_coeff: Penalty factor, default is 0.0
        clip: gradient clipping threshold, default is None
        class_weight: the weight of every class, default is None
        loss_name: the name of loss function, default is 'wsce'
    '''
    print('Train on ', CTX)

    for epoch in range(1, num_epochs + 1):
        start = time()
        train_loss = 0.
        total_pred = []
        total_true = []
        n_batch = 0

        for batch_x, batch_y in data_iter_train:
            with autograd.record():
                batch_pred, att_output = model(batch_x)
                if loss_name == 'sce':
                    l = loss(batch_pred, batch_y)
                elif loss_name == 'wsce':
                    l = loss(batch_pred, batch_y, class_weight, class_weight.shape[0])

                # 惩罚项
                temp = nd.batch_dot(att_output, nd.transpose(att_output, axes=(0, 2, 1))
                                    ) - nd.eye(att_output.shape[1], ctx=CTX)
                l = l + penal_coeff * temp.norm(axis=(1, 2))
            l.backward()

            # 梯度裁剪
            clip_params = [p.data() for p in model.collect_params().values()]
            if clip is not None:
                norm = nd.array([0.0], CTX)
                for param in clip_params:
                    norm += (param.grad ** 2).sum()
                norm = norm.sqrt().asscalar()
                if norm > clip:
                    for param in clip_params:
                        param.grad[:] *= clip / norm

            # 更新参数
            trainer.step(batch_x.shape[0])

            batch_pred = np.argmax(nd.softmax(batch_pred, axis=1).asnumpy(), axis=1)
            batch_true = np.reshape(batch_y.asnumpy(), (-1, ))
            total_pred.extend(batch_pred.tolist())
            total_true.extend(batch_true.tolist())
            batch_train_loss = l.mean().asscalar()

            n_batch += 1
            train_loss += batch_train_loss

            if n_batch % 400 == 0:
                print('epoch %d, batch %d, bach_train_loss %.4f, batch_train_acc %.3f' %
                      (epoch, n_batch, batch_train_loss, accuracy_score(batch_true, batch_pred)))

        F1_train = f1_score(np.array(total_true), np.array(total_pred), average='weighted')
        acc_train = accuracy_score(np.array(total_true), np.array(total_pred))
        train_loss /= n_batch

        F1_valid, acc_valid, valid_loss = evaluate(data_iter_valid, model, loss, penal_coeff, class_weight, loss_name)

        print('epoch %d, learning_rate %.5f \n\t train_loss %.4f, acc_train %.3f, F1_train %.3f, ' %
              (epoch, trainer.learning_rate, train_loss, acc_train, F1_train))
        print('\t valid_loss %.4f, acc_valid %.3f, F1_valid %.3f, '
              '\ntime %.1f sec' % (valid_loss, acc_valid, F1_valid, time() - start))
        print('='*50)

        # 学习率衰减
        if epoch % 2 == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.9)


def evaluate(data_iter_valid, model, loss, penal_coeff=0.0, class_weight=None, loss_name='wsce'):
    '''
    the evaluation function
    Args:
        data_iter_valid: the iter of validation data
        model: model to train
        loss: loss function
        penal_coeff: Penalty factor, default is 0.0
        class_weight: the weight of every class, default is None
        loss_name: the name of loss function, default is 'wsce'
    Returns:
        F1_valid: the f1 score
        acc_valid: the accuracy score
        valid_loss: the value of loss
    '''
    valid_loss = 0.
    total_pred = []
    total_true = []
    n_batch = 1
    for batch_x, batch_y in data_iter_valid:
        batch_pred, att_output = model(batch_x)
        if loss_name == 'sce':
            l = loss(batch_pred, batch_y)
        elif loss_name == 'wsce':
            l = loss(batch_pred, batch_y, class_weight, class_weight.shape[0])
        # 惩罚项
        temp = nd.batch_dot(att_output, nd.transpose(att_output, axes=(0, 2, 1))
                            ) - nd.eye(att_output.shape[1], ctx=att_output.context)
        l = l + penal_coeff * temp.norm(axis=(1, 2))
        total_pred.extend(np.argmax(nd.softmax(batch_pred, axis=1).asnumpy(), axis=1).tolist())
        total_true.extend(np.reshape(batch_y.asnumpy(), (-1,)).tolist())
        n_batch += 1
        valid_loss += l.mean().asscalar()

    F1_valid = f1_score(np.array(total_true), np.array(total_pred), average='weighted')
    acc_valid = accuracy_score(np.array(total_true), np.array(total_pred))
    valid_loss /= n_batch

    return F1_valid, acc_valid, valid_loss
