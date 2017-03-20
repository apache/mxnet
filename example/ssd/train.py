import argparse
import tools.find_mxnet
import mxnet as mx
import os
import sys
from train.train_net import train_net

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
    parser.add_argument('--dataset', dest='dataset', help='which dataset to use',
                        default='pascal', type=str)
    parser.add_argument('--image-set', dest='image_set', help='train set, can be trainval or train',
                        default='trainval', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2012',
                        default='2007,2012', type=str)
    parser.add_argument('--val-image-set', dest='val_image_set', help='validation set, can be val or test',
                        default='test', type=str)
    parser.add_argument('--val-year', dest='val_year', help='can be 2007, 2010, 2012',
                        default='2007', type=str)
    parser.add_argument('--devkit-path', dest='devkit_path', help='VOCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        choices=['vgg16_reduced'], help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--finetune', dest='finetune', type=int, default=-1,
                        help='finetune from epoch n, rename the model before doing this')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16_reduced'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=100, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--lr-epoch', dest='lr_refactor_epoch', type=int, default=50,
                        help='refactor learning rate every N epoch')
    parser.add_argument('--lr-ratio', dest='lr_refactor_ratio', type=float, default=0.9,
                        help='ratio to refactor learning rate')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                        help='save training log to file')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    ctx = mx.cpu() if not ctx else ctx
    train_net(args.network, args.dataset, args.image_set, args.year,
              args.devkit_path, args.batch_size,
              args.data_shape, [args.mean_r, args.mean_g, args.mean_b],
              args.resume, args.finetune, args.pretrained,
              args.epoch, args.prefix, ctx, args.begin_epoch, args.end_epoch,
              args.frequent, args.learning_rate, args.momentum, args.weight_decay,
              args.val_image_set, args.val_year, args.lr_refactor_epoch,
              args.lr_refactor_ratio, args.monitor, args.log_file)
