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
    buckets = [10, 15, 20]
    num_hidden = args.num_hidden
    num_embed = args.num_embed
    num_epoch = args.num_epoch
    num_lstm_layer = args.num_lstm_layer
    optimizer = args.optimizer
    device = mx.gpu(args.gpuid) if args.gpuid >= 0 else mx.cpu()
    #device = [mx.gpu(0), mx.gpu(1)]

    vgg_params = mx.nd.load(args.vgg_params)

    dummy_data = args.dummy_data

    train_json = args.train_json
    train_hdf5 = args.train_hdf5
    eval_json = args.val_json
    eval_hdf5 = args.val_hdf5
    num_eval = args.num_eval

    optim_state = {
        'learning_rate': args.learning_rate
    }

    if args.vocab_json and os.path.exists(args.vocab_json):
        vocab = json.load(open(args.vocab))
    else:
        vocab = build_vocab(train_json, args.count_threshold)
        json.dump(vocab, open('vocab.json', 'w'))

    def sym_gen(seq_len):
        return network_unroll(num_lstm_layer=num_lstm_layer,
                              num_embed=num_embed,
                              num_hidden=num_hidden,
                              vocab_size=len(vocab),
                              seq_len=seq_len)

    init_c = [('l%d_init_c' %l, (batch_size, num_hidden))
              for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h' %l, (batch_size, num_hidden))
              for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = BucketImageSentenceIter(train_json, train_hdf5, vocab,
                                         buckets, batch_size, init_states)
    data_val   = BucketImageSentenceIter(eval_json[:num_eval], eval_hdf5, vocab,
                                         buckets, batch_size, init_states,
                                         is_train=False)

    if dummy_data:
        print "Using dummy data for speed test"
        data_train = DummyIter(data_train)
        data_val = DummyIter(data_val)

    if len(buckets) == 1:
        symbol = sym_gen(buckets[0])
    else:
        symbol = sym_gen

    model = mx.model.FeedForward(ctx=device,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 optimizer=optimizer,
                                 arg_params=vgg_params,
                                 allow_extra_params=True,
                                 **optim_state
                                 )
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),
              epoch_end_callback=mx.callback.do_checkpoint(args.save_name))


if __name__ == '__main__':
    import argparse
    import os

    os.environ["MXNET_EXEC_INPLACE_GRAD_SUM_CAP"]= '100'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_hidden', type=int, default=512, help='size of hidden layer of LSTM')
    parser.add_argument('--num_embed', type=int, default=512, help='embedding size')
    parser.add_argument('--num_epoch', type=int, default=50, help='max epoch')
    parser.add_argument('--num_lstm_layer',type=int, default=1, help='number of LSTM layer')
    parser.add_argument('--gpuid', type=int, default=-1, help='index of gpu, -1 for cpu')
    parser.add_argument('--vgg_params', default='vgg16.params', help='path for VGG16 params')
    parser.add_argument('--dummy_data', type=bool, default=False, help='whether to use dummy data')
    parser.add_argument('--train_json', help='path for train json')
    parser.add_argument('--train_hdf5', help='path for train hdf5')
    parser.add_argument('--val_json', help='path for val json')
    parser.add_argument('--val_hdf5', help='path for val hdf5')
    parser.add_argument('--num_eval', type=int, default=3200, help='number of validation samples to use')
    parser.add_argument('--save_name', type=str, default='nic', help='name to save checkpoint')
    parser.add_argument('--vocab_json', type=str, help='json file of vocab')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam | rmsprop')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--count_threshold', type=int, default=5, help='appear number is less than threshold will not'
                                                                       'be counted in vocab')

    args = parser.parse_args()
    train(args)

