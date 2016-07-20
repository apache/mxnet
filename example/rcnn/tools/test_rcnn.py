import argparse
import logging
import os

import mxnet as mx

from rcnn.loader import ROIIter
from rcnn.detector import Detector
from rcnn.symbol import get_vgg_rcnn_test
from rcnn.tester import pred_eval
from utils.load_data import load_test_rpn_roidb
from utils.load_model import load_param


def test_net(imageset, year, root_path, devkit_path, prefix, epoch, ctx, vis):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # load testing data
    voc, roidb = load_test_rpn_roidb(imageset, year, root_path, devkit_path)
    test_data = ROIIter(roidb, batch_size=1, shuffle=False, mode='test')

    # load model
    args, auxs = load_param(prefix, epoch, convert=True, ctx=ctx)

    # load symbol
    sym = get_vgg_rcnn_test()

    # detect
    detector = Detector(sym, ctx, args, auxs)
    pred_eval(detector, test_data, voc, vis=vis)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--image_set', dest='image_set', help='can be test',
                        default='test', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2010, 2012',
                        default='2007', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='VOCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--prefix', dest='prefix', help='model to test with', type=str)
    parser.add_argument('--epoch', dest='epoch', help='model to test with',
                        default=8, type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to test with',
                        default=0, type=int)
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    test_net(args.image_set, args.year, args.root_path, args.devkit_path, args.prefix, args.epoch, ctx, args.vis)
