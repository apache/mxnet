import argparse
import mxnet as mx
import os
from tools.train_net import train_net


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--image_set', dest='image_set', help='can be trainval or train',
                        default='train', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2010, 2012',
                        default='2007', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='VOCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'frcnn'), type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=8, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    train_net(args.image_set, args.year, args.root_path, args.devkit_path, args.pretrained, args.epoch,
              args.prefix, ctx, args.begin_epoch, args.end_epoch, args.frequent)
