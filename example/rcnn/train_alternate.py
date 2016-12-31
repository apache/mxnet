import argparse
import logging
import os

import mxnet as mx

from rcnn.config import config
from rcnn.loader import AnchorLoader, ROIIter
from tools.train_rpn import train_rpn
from tools.train_rcnn import train_rcnn
from tools.test_rpn import test_rpn
from utils.combine_model import combine_model


def alternate_train(image_set, test_image_set, year, root_path, devkit_path, pretrained, epoch,
                    ctx, begin_epoch, rpn_epoch, rcnn_epoch, frequent, kv_store, work_load_list=None):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    config.TRAIN.BG_THRESH_LO = 0.0

    logging.info('########## TRAIN RPN WITH IMAGENET INIT')
    config.TRAIN.HAS_RPN = True
    config.TRAIN.BATCH_SIZE = 1
    train_rpn(image_set, year, root_path, devkit_path, pretrained, epoch,
              'model/rpn1', ctx, begin_epoch, rpn_epoch, frequent, kv_store, work_load_list)

    logging.info('########## GENERATE RPN DETECTION')
    config.TEST.HAS_RPN = True
    config.TEST.RPN_PRE_NMS_TOP_N = -1
    config.TEST.RPN_POST_NMS_TOP_N = 2000
    test_rpn(image_set, year, root_path, devkit_path, 'model/rpn1', rpn_epoch, ctx[0])

    logging.info('########## TRAIN RCNN WITH IMAGENET INIT AND RPN DETECTION')
    config.TRAIN.HAS_RPN = False
    config.TRAIN.BATCH_SIZE = 128
    train_rcnn(image_set, year, root_path, devkit_path, pretrained, epoch,
               'model/rcnn1', ctx, begin_epoch, rcnn_epoch, frequent, kv_store, work_load_list)

    logging.info('########## TRAIN RPN WITH RCNN INIT')
    config.TRAIN.HAS_RPN = True
    config.TRAIN.BATCH_SIZE = 1
    config.TRAIN.FINETUNE = True
    train_rpn(image_set, year, root_path, devkit_path, 'model/rcnn1', rcnn_epoch,
              'model/rpn2', ctx, begin_epoch, rpn_epoch, frequent, kv_store, work_load_list)

    logging.info('########## GENERATE RPN DETECTION')
    config.TEST.HAS_RPN = True
    config.TEST.RPN_PRE_NMS_TOP_N = -1
    config.TEST.RPN_POST_NMS_TOP_N = 2000
    test_rpn(image_set, year, root_path, devkit_path, 'model/rpn2', rpn_epoch, ctx[0])

    logger.info('########## COMBINE RPN2 WITH RCNN1')
    combine_model('model/rpn2', rpn_epoch, 'model/rcnn1', rcnn_epoch, 'model/rcnn2', 0)

    logger.info('########## TRAIN RCNN WITH RPN INIT AND DETECTION')
    config.TRAIN.HAS_RPN = False
    config.TRAIN.BATCH_SIZE = 128
    train_rcnn(image_set, year, root_path, devkit_path, 'model/rcnn2', 0,
               'model/rcnn2', ctx, begin_epoch, rcnn_epoch, frequent, kv_store, work_load_list)

    logger.info('########## COMBINE RPN2 WITH RCNN2')
    combine_model('model/rpn2', rpn_epoch, 'model/rcnn2', rcnn_epoch, 'model/final', 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN Network')
    parser.add_argument('--image_set', dest='image_set', help='can be trainval or train',
                        default='trainval', type=str)
    parser.add_argument('--test_image_set', dest='test_image_set', help='can be test or val',
                        default='test', type=str)
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
    parser.add_argument('--gpus', dest='gpu_ids', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--rpn_epoch', dest='rpn_epoch', help='end epoch of rpn training',
                        default=8, type=int)
    parser.add_argument('--rcnn_epoch', dest='rcnn_epoch', help='end epoch of rcnn training',
                        default=8, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--kv_store', dest='kv_store', help='the kv-store type',
                        default='device', type=str)
    parser.add_argument('--work_load_list', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    alternate_train(args.image_set, args.test_image_set, args.year, args.root_path, args.devkit_path,
                    args.pretrained, args.epoch, ctx, args.begin_epoch, args.rpn_epoch, args.rcnn_epoch,
                    args.frequent, args.kv_store, args.work_load_list)
