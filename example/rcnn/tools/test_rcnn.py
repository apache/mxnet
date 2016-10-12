import argparse
import os

import mxnet as mx

from rcnn.config import config
from rcnn.loader import ROIIter
from rcnn.detector import Detector
from rcnn.symbol import get_vgg_test, get_vgg_rcnn_test
from rcnn.tester import pred_eval
from utils.load_data import load_gt_roidb, load_test_ss_roidb, load_test_rpn_roidb
from utils.load_model import load_param


def test_rcnn(imageset, year, root_path, devkit_path, prefix, epoch, ctx, vis=False, has_rpn=True, proposal='rpn',
              end2end=False):
    # load symbol and testing data
    if has_rpn:
        sym = get_vgg_test()
        config.TEST.HAS_RPN = True
        config.TEST.RPN_PRE_NMS_TOP_N = 6000
        config.TEST.RPN_POST_NMS_TOP_N = 300
        voc, roidb = load_gt_roidb(imageset, year, root_path, devkit_path)
    else:
        sym = get_vgg_rcnn_test()
        voc, roidb = eval('load_test_' + proposal + '_roidb')(imageset, year, root_path, devkit_path)

    # get test data iter
    test_data = ROIIter(roidb, batch_size=1, shuffle=False, mode='test')

    # load model
    args, auxs, _ = load_param(prefix, epoch, convert=True, ctx=ctx)

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
    parser.add_argument('--has_rpn', dest='has_rpn', help='generate proposals on the fly',
                        action='store_true')
    parser.add_argument('--proposal', dest='proposal', help='can be ss for selective search or rpn',
                        default='rpn', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    if args.end2end:
        args.has_rpn = True
    test_rcnn(args.image_set, args.year, args.root_path, args.devkit_path, args.prefix, args.epoch, ctx, args.vis,
              args.has_rpn, args.proposal)
