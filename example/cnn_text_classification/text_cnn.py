#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import mxnet as mx
import numpy as np
import time
import math
import data_helpers
from collections import namedtuple

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # get a logger to accuracies are printed

CNNModel = namedtuple("CNNModel", ['cnn_exec', 'symbol', 'data', 'label', 'param_blocks'])

def make_text_cnn(sentence_size, num_embed, batch_size, num_label=2, filter_list=[3, 4, 5], num_filter=100, dropout=0.):
    input_x = mx.sym.Variable('data') # placeholder for input
    input_y = mx.sym.Variable('softmax_label') # placeholder for output

    # embedding layer
    # embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
    # embed_layer = mx.sym.Reshape(data=embed_layer, target_shape=(1, 1, sentence_size, num_embed))

    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=input_x, kernel=(filter_size, num_embed), num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1,1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.Concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))

    # dropout layer
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

    # fully connected
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    # softmax output
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    return sm


def setup_cnn_model(ctx, batch_size, sentence_size, num_embed, dropout=0.5, initializer=mx.initializer.Uniform(0.1)):
    cnn = make_text_cnn(sentence_size, num_embed, batch_size=batch_size, dropout=dropout)
    arg_names = cnn.list_arguments()

    input_shapes = {}
    input_shapes['data'] = (batch_size, 1, sentence_size, num_embed)

    arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
    arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        if name in ['softmax_label', 'data']: # input, output
            continue
        args_grad[name] = mx.nd.zeros(shape, ctx)

    cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

    param_blocks = []
    arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
    for i, name in enumerate(arg_names):
        if name in ['softmax_label', 'data']: # input, output
            continue
        initializer(name, arg_dict[name])

        param_blocks.append( (i, arg_dict[name], args_grad[name], name) )

    out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))

    data = cnn_exec.arg_dict['data']
    label = cnn_exec.arg_dict['softmax_label']

    return CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)


def train_cnn(model, X_train_batch, y_train_batch, X_dev_batch, y_dev_batch, batch_size, optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.001, epoch=200):
    m = model
    # create optimizer
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate

    updater = mx.optimizer.get_updater(opt)

    for iteration in range(epoch):
        tic = time.time()
        num_correct = 0
        num_total = 0
        for begin in range(0, X_train_batch.shape[0], batch_size):
            batchX = X_train_batch[begin:begin+batch_size]
            batchY = y_train_batch[begin:begin+batch_size]
            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX
            m.label[:] = batchY

            # forward
            m.cnn_exec.forward(is_train=True)

            # backward
            m.cnn_exec.backward()

            # eval on training data
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

            # update weights
            norm = 0
            for idx, weight, grad, name in m.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in m.param_blocks:
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)

                updater(idx, grad, weight)

                # reset gradient to zero
                grad[:] = 0.0

        # decay learning rate
        if iteration % 50 == 0 and iteration > 0:
            opt.lr *= 0.5
            print >> sys.stderr, 'reset learning rate to %g' % opt.lr

        # end of training loop
        toc = time.time()
        print >> sys.stderr, 'Iter [%d] Train: Time: %.3f, Training Accuracy: %.3f' % (iteration, toc - tic, num_correct * 100 / float(num_total))

        # eval on dev set
        num_correct = 0
        num_total = 0
        for begin in range(0, X_dev_batch.shape[0], batch_size):
            batchX = X_dev_batch[begin:begin+batch_size]
            batchY = y_dev_batch[begin:begin+batch_size]

            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX
            m.cnn_exec.forward(is_train=False)

            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        print >> sys.stderr, 'Dev Accuracy thus far: %.3f' % ( num_correct * 100 / float(num_total) )


def main():
    print 'Loading data...'
    # word2vec = data_helpers.load_google_word2vec('data/GoogleNews-vectors-negative300.bin')
    word2vec = data_helpers.load_pretrained_word2vec('data/rt.vec')
    x, y = data_helpers.load_data_with_word2vec(word2vec)


    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # split train/dev set
    x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
    print 'Train/Dev split: %d/%d' % (len(y_train), len(y_dev))
    print 'train shape:', x_train.shape
    print 'dev shape:', x_dev.shape

    # reshpae for convolution input
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    x_dev = np.reshape(x_dev, (x_dev.shape[0], 1, x_dev.shape[1], x_dev.shape[2]))

    num_embed = x_train.shape[-1]
    sentence_size = x_train.shape[2]
    print 'sentence max words', sentence_size
    print 'embedding size', num_embed
    batch_size = 50

    cnn_model = setup_cnn_model(mx.gpu(0), batch_size, sentence_size, num_embed, dropout=0.5)
    train_cnn(cnn_model, x_train, y_train, x_dev, y_dev, batch_size)


if __name__ == '__main__':
    main()
