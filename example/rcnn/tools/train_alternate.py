import argparse
import logging
import os

import mxnet as mx

from rcnn.config import config
from rcnn.loader import AnchorLoader, ROIIter
from rcnn.solver import Solver
from rcnn.symbol import get_vgg_rpn, get_vgg_rpn_test, get_vgg_rcnn
from utils.load_data import load_gt_roidb, load_rpn_roidb
from utils.load_model import load_checkpoint, load_param
from utils.save_model import save_checkpoint
from utils.combine_model import combine_model


def train_rpn(image_set, year, root_path, devkit_path, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent, kv_store, work_load_list=None):
    # load symbol
    sym = get_vgg_rpn()
    feat_sym = get_vgg_rpn().get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    config.TRAIN.BATCH_IMAGES *= len(ctx)
    config.TRAIN.BATCH_SIZE *= len(ctx)

    # load training data
    voc, roidb = load_gt_roidb(image_set, year, root_path, devkit_path, flip=True)
    train_data = AnchorLoader(feat_sym, roidb, batch_size=config.TRAIN.BATCH_SIZE, shuffle=True, mode='train',
                              ctx=ctx, work_load_list=work_load_list)

    # infer max shape
    max_data_shape = [('data', (1, 3, 1000, 1000))]
    max_data_shape_dict = {k: v for k, v in max_data_shape}
    _, feat_shape, _ = feat_sym.infer_shape(**max_data_shape_dict)
    from rcnn.minibatch import assign_anchor
    import numpy as np
    label = assign_anchor(feat_shape[0], np.zeros((0, 5)), [[1000, 1000, 1.0]])
    max_label_shape = [('label', label['label'].shape),
                       ('bbox_target', label['bbox_target'].shape),
                       ('bbox_inside_weight', label['bbox_inside_weight'].shape),
                       ('bbox_outside_weight', label['bbox_outside_weight'].shape)]
    print 'providing maximum shape', max_data_shape, max_label_shape

    # load pretrained
    args, auxs = load_param(pretrained, epoch, convert=True)

    # initialize params
    arg_shape, _, _ = sym.infer_shape(data=(1, 3, 224, 224))
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    args['rpn_conv_3x3_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
    args['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
    args['rpn_cls_score_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
    args['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
    args['rpn_bbox_pred_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
    args['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])

    # train
    solver = Solver(prefix, sym, ctx, begin_epoch, end_epoch, kv_store, args, auxs, momentum=0.9, wd=0.0005,
                    learning_rate=1e-3, lr_scheduler=mx.lr_scheduler.FactorScheduler(60000, 0.1),
                    mutable_data_shape=True, max_data_shape=max_data_shape, max_label_shape=max_label_shape)
    solver.fit(train_data, frequent=frequent)


def test_rpn(image_set, year, root_path, devkit_path, trained, epoch, ctx):
    from rcnn.rpn.generate import Detector, generate_detections

    # load symbol
    sym = get_vgg_rpn_test()

    # load testing data
    voc, roidb = load_gt_roidb(image_set, year, root_path, devkit_path)
    test_data = ROIIter(roidb, batch_size=1, shuffle=False, mode='test')

    # load trained
    args, auxs = load_param(trained, epoch, convert=True, ctx=ctx[0])

    # start testing
    detector = Detector(sym, ctx[0], args, auxs)
    imdb_boxes = generate_detections(detector, test_data, voc, vis=False)
    voc.evaluate_recall(roidb, candidate_boxes=imdb_boxes)


def train_rcnn(image_set, year, root_path, devkit_path, pretrained, epoch,
               prefix, ctx, begin_epoch, end_epoch, frequent, kv_store, work_load_list=None):
    # load symbol
    sym = get_vgg_rcnn()

    # setup multi-gpu
    config.TRAIN.BATCH_IMAGES *= len(ctx)
    config.TRAIN.BATCH_SIZE *= len(ctx)

    # load training data
    voc, roidb, means, stds = load_rpn_roidb(image_set, year, root_path, devkit_path, flip=True)
    train_data = ROIIter(roidb, batch_size=config.TRAIN.BATCH_IMAGES, shuffle=True, mode='train',
                         ctx=ctx, work_load_list=work_load_list)

    # infer max shape
    max_data_shape = [('data', (1, 3, 1000, 1000))]

    # load pretrained
    args, auxs = load_param(pretrained, epoch, convert=True)

    # initialize params
    arg_shape, _, _ = sym.infer_shape(data=(1, 3, 224, 224), rois=(1, 5))
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    args['cls_score_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['cls_score_weight'])
    args['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
    args['bbox_pred_weight'] = mx.random.normal(mean=0, stdvar=0.001, shape=arg_shape_dict['bbox_pred_weight'])
    args['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

    # train
    solver = Solver(prefix, sym, ctx, begin_epoch, end_epoch, kv_store, args, auxs, momentum=0.9, wd=0.0005,
                    learning_rate=1e-3, lr_scheduler=mx.lr_scheduler.FactorScheduler(30000, 0.1),
                    mutable_data_shape=True, max_data_shape=max_data_shape)
    solver.fit(train_data, frequent=frequent)

    # edit params and save
    for epoch in range(begin_epoch + 1, end_epoch + 1):
        arg_params, aux_params = load_checkpoint(prefix, epoch)
        arg_params['bbox_pred_weight'] = (arg_params['bbox_pred_weight'].T * mx.nd.array(stds)).T
        arg_params['bbox_pred_bias'] = arg_params['bbox_pred_bias'] * mx.nd.array(stds) + \
                                       mx.nd.array(means)
        save_checkpoint(prefix, epoch, arg_params, aux_params)


def alternate_train(image_set, year, root_path, devkit_path, pretrained, epoch,
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
    test_rpn(image_set, year, root_path, devkit_path, 'model/rpn1', rpn_epoch, ctx)

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
    test_rpn(image_set, year, root_path, devkit_path, 'model/rpn2', rpn_epoch, ctx)

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
                        default=os.path.join(os.getcwd(), 'model', 'rcnn'), type=str)
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
    alternate_train(args.image_set, args.year, args.root_path, args.devkit_path, args.pretrained, args.epoch,
                    ctx, args.begin_epoch, args.rpn_epoch, args.rcnn_epoch, args.frequent,
                    args.kv_store, args.work_load_list)
