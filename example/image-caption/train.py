# -*- coding:utf-8 -*-
# @author: Yuanqin Lu

from net import network_unroll
from bucket_io import BucketImageSentenceIter, build_vocab, DummyIter
import mxnet as mx
import numpy as np
import json

def Perplexity(label, pred):
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

def train(args):
    batch_size = args.batch_size
    num_lstm_layer = args.num_lstm_layer
    num_hidden = args.num_hidden
    num_seq = args.num_seq
    num_embed = args.num_embed
    num_epoch = args.num_epoch
    dummy_data = args.dummy_data
    device = mx.gpu(args.gpuid) if args.gpuid >= 0 else mx.cpu()


    init_c = [('l%d_init_c' %l, (batch_size, num_hidden))
              for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' %l, (batch_size, num_hidden))
              for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    data_train = BucketImageSentenceIter(input_hdf5=args.train_hdf5,
                                         batch_size=batch_size,
                                         init_states=init_states,
                                         seq_per_img=num_seq)
    info = json.load(open(args.train_json))
    vocab_size = len(info['ix_to_word']) + 1

    net = network_unroll(num_lstm_layer=num_lstm_layer,
                         seq_len=data_train.seq_length,
                         vocab_size=vocab_size,
                         num_hidden=num_hidden,
                         num_embed=num_embed,
                         num_seq=num_seq)
    net.infer_shape()

    optimizer = mx.optimizer.create(args.optimizer)



    if dummy_data:
        print "Using dummy data for speed test"
        data_train = DummyIter(data_train)


    model = mx.model.FeedForward(ctx=device,
                                 symbol=net,
                                 num_epoch=num_epoch,
                                 optimizer=optimizer,
                                 #arg_params=vgg_params,
                                 allow_extra_params=True,
                                 )
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50))


if __name__ == '__main__':
    import argparse
    import os


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_hidden', type=int, default=512, help='size of hidden layer of LSTM')
    parser.add_argument('--num_embed', type=int, default=512, help='embedding size')
    parser.add_argument('--num_epoch', type=int, default=50, help='max epoch')
    parser.add_argument('--num_lstm_layer',type=int, default=1, help='number of LSTM layer')
    parser.add_argument('--num_seq', type=int, default=5, help='')
    parser.add_argument('--gpuid', type=int, default=-1, help='index of gpu, -1 for cpu')
    parser.add_argument('--vgg_params', default='vgg16.params', help='path for VGG16 params')
    parser.add_argument('--dummy_data', type=bool, default=False, help='whether to use dummy data')
    parser.add_argument('--train_hdf5', help='path for train json')
    parser.add_argument('--train_json', help='path for train hdf5')
    parser.add_argument('--num_eval', type=int, default=3200, help='number of validation samples to use')
    parser.add_argument('--save_name', type=str, default='nic', help='name to save checkpoint')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam | rmsprop')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--count_threshold', type=int, default=5, help='appear number is less than threshold will not'
                                                                       'be counted in vocab')

    args = parser.parse_args()
    train(args)

