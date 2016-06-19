#!/usr/bin/env python
# distributed lenet
import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../../example/image-classification"))
sys.path.append(os.path.join(curr_path, "../../python"))
import mxnet as mx
import argparse
import train_mnist
import logging

if __name__ == '__main__':
    args = train_mnist.parse_args()
    args.batch_size = 100
    data_shape = (1, 28, 28)
    loader = train_mnist.get_iterator(data_shape)
    kv = mx.kvstore.create(args.kv_store)
    (train, val) = loader(args, kv)
    net = train_mnist.get_lenet()

    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    mx.model.FeedForward.create(
        ctx           = mx.gpu(kv.rank),
        kvstore       = kv,
        symbol        = net,
        X             = train,
        eval_data     = val,
        num_epoch     = args.num_epochs,
        learning_rate = args.lr,
        momentum      = 0.9,
        wd            = 0.00001)
