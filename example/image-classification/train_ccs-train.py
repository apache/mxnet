import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

def download_cifar10():
    fnames = range(2)
    fnames[1] = "/home/yuanshuai/data/code/kaggle-dog-breed-identification/pretrained-based-on-standford-dog-dataset/standford-512-qua-99-ratio-0.95_val.rec"
    fnames[0] = "/home/yuanshuai/data/code/kaggle-dog-breed-identification/pretrained-based-on-standford-dog-dataset/standford-512-qua-99-ratio-0.95_train.rec"
    return fnames

if __name__ == '__main__':
    # download data
    (train_fname, val_fname) = download_cifar10()

    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'inception-resnet-v2',#'resnet',
        num_layers     = 50,#50,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 120,
        # train_add
        num_examples   = 50000,
        #num_examples   = 1406,#train:1406.95 all:1481
        image_shape    = '3,512,512',
        pad_size       = 4,
        # train
        batch_size     = 30,
        num_epochs     = 300,
        batch_end_call_back = 800,
        #default lr:0.05
        lr             = .01,
        lr_step_epochs = '200,250',
        gpus           = '1,2',
        top_k           = 5,
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)
