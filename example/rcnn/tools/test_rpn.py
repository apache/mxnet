import argparse
import os

import mxnet as mx

from rcnn.config import config
from rcnn.loader import ROIIter
from rcnn.rpn.generate import Detector, generate_detections
from rcnn.symbol import get_vgg_rpn_test
from utils.load_data import load_gt_roidb
from utils.load_model import load_param

# rpn generate proposal config
config.TEST.HAS_RPN = True
config.TEST.RPN_PRE_NMS_TOP_N = -1
config.TEST.RPN_POST_NMS_TOP_N = 2000


def test_rpn(image_set, year, root_path, devkit_path, prefix, epoch, ctx, vis=False):
    # load symbol
    sym = get_vgg_rpn_test()

    # load testing data
    voc, roidb = load_gt_roidb(image_set, year, root_path, devkit_path)
    test_data = ROIIter(roidb, batch_size=1, shuffle=False, mode='test')

    # load model
    args, auxs = load_param(prefix, epoch, convert=True, ctx=ctx)

    # start testing
    detector = Detector(sym, ctx, args, auxs)
    imdb_boxes = generate_detections(detector, test_data, voc, vis=vis)
    voc.evaluate_recall(roidb, candidate_boxes=imdb_boxes)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Region Proposal Network')
    parser.add_argument('--image_set', dest='image_set', help='can be trainval or train',
                        default='trainval', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2010, 2012',
                        default='2007', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='VOCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--prefix', dest='prefix', help='model to test with', type=str)
    parser.add_argument('--epoch', dest='epoch', help='model to test with',
                        default=8, type=int)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    test_rpn(args.image_set, args.year, args.root_path, args.devkit_path, args.prefix, args.epoch, ctx, args.vis)
