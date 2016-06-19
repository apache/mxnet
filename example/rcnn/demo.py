import argparse
import mxnet as mx
import os
from tools.load_model import load_param
from rcnn.symbol import get_symbol_vgg_test
from rcnn.detector import Detector
from tools.demo_net import demo_net


def get_net(prefix, epoch, ctx):
    args, auxs = load_param(prefix, epoch, convert=True, ctx=ctx)
    sym = get_symbol_vgg_test()
    detector = Detector(sym, ctx, args, auxs)
    return detector


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Fast R-CNN network')
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'frcnn'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=9, type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to test with',
                        default=0, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    detector = get_net(args.prefix, args.epoch, ctx)
    demo_net(detector, os.path.join(os.getcwd(), 'data', 'demo', '000004'))
    demo_net(detector, os.path.join(os.getcwd(), 'data', 'demo', '001551'))
