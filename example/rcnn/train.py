# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import ast
import pprint

import mxnet as mx
from mxnet.module import Module

from symdata.loader import AnchorGenerator, AnchorSampler, AnchorLoader
from symnet.logger import logger
from symnet.model import load_param, infer_data_shape, check_shape, initialize_frcnn, get_fixed_params
from symnet.metric import RPNAccMetric, RPNLogLossMetric, RPNL1LossMetric, RCNNAccMetric, RCNNLogLossMetric, RCNNL1LossMetric


def train_net(sym, roidb, args):
    # print config
    logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))

    # setup multi-gpu
    ctx = [mx.cpu()] if not args.gpus else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = args.rcnn_batch_size * len(ctx)

    # load training data
    feat_sym = sym.get_internals()['rpn_cls_score_output']
    ag = AnchorGenerator(feat_stride=args.rpn_feat_stride,
                         anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios)
    asp = AnchorSampler(allowed_border=args.rpn_allowed_border, batch_rois=args.rpn_batch_rois,
                        fg_fraction=args.rpn_fg_fraction, fg_overlap=args.rpn_fg_overlap,
                        bg_overlap=args.rpn_bg_overlap)
    train_data = AnchorLoader(roidb, batch_size, args.img_short_side, args.img_long_side,
                              args.img_pixel_means, args.img_pixel_stds, feat_sym, ag, asp, shuffle=True)

    # produce shape max possible
    _, out_shape, _ = feat_sym.infer_shape(data=(1, 3, args.img_long_side, args.img_long_side))
    feat_height, feat_width = out_shape[0][-2:]
    rpn_num_anchors = len(args.rpn_anchor_scales) * len(args.rpn_anchor_ratios)
    data_names = ['data', 'im_info', 'gt_boxes']
    label_names = ['label', 'bbox_target', 'bbox_weight']
    data_shapes = [('data', (batch_size, 3, args.img_long_side, args.img_long_side)),
                   ('im_info', (batch_size, 3)),
                   ('gt_boxes', (batch_size, 100, 5))]
    label_shapes = [('label', (batch_size, 1, rpn_num_anchors * feat_height, feat_width)),
                    ('bbox_target', (batch_size, 4 * rpn_num_anchors, feat_height, feat_width)),
                    ('bbox_weight', (batch_size, 4 * rpn_num_anchors, feat_height, feat_width))]

    # print shapes
    data_shape_dict, out_shape_dict = infer_data_shape(sym, data_shapes + label_shapes)
    logger.info('max input shape\n%s' % pprint.pformat(data_shape_dict))
    logger.info('max output shape\n%s' % pprint.pformat(out_shape_dict))

    # load and initialize params
    if args.resume:
        arg_params, aux_params = load_param(args.resume)
    else:
        arg_params, aux_params = load_param(args.pretrained)
        arg_params, aux_params = initialize_frcnn(sym, data_shapes, arg_params, aux_params)

    # check parameter shapes
    check_shape(sym, data_shapes + label_shapes, arg_params, aux_params)

    # check fixed params
    fixed_param_names = get_fixed_params(sym, args.net_fixed_params)
    logger.info('locking params\n%s' % pprint.pformat(fixed_param_names))

    # metric
    rpn_eval_metric = RPNAccMetric()
    rpn_cls_metric = RPNLogLossMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    eval_metric = RCNNAccMetric()
    cls_metric = RCNNLogLossMetric()
    bbox_metric = RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = mx.callback.Speedometer(batch_size, frequent=args.log_interval, auto_reset=False)
    epoch_end_callback = mx.callback.do_checkpoint(args.save_prefix)

    # learning schedule
    base_lr = args.lr
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in args.lr_decay_epoch.split(',')]
    lr_epoch_diff = [epoch - args.start_epoch for epoch in lr_epoch if epoch > args.start_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod = Module(sym, data_names=data_names, label_names=label_names,
                 logger=logger, context=ctx, work_load_list=None,
                 fixed_param_names=fixed_param_names)
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore='device',
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=args.start_epoch, num_epoch=args.epochs)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='vgg16', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpus', type=str, help='GPU devices, eg: "0,1,2,3" , not set to use CPU')
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
    parser.add_argument('--lr-decay-epoch', type=str, default='7', help='epoch to decay lr')
    parser.add_argument('--resume', type=str, default='', help='path to last saved model')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch for resuming')
    parser.add_argument('--log-interval', type=int, default=100, help='logging mini batch interval')
    parser.add_argument('--save-prefix', type=str, default='', help='saving params prefix')
    # faster rcnn params
    parser.add_argument('--img-short-side', type=int, default=600)
    parser.add_argument('--img-long-side', type=int, default=1000)
    parser.add_argument('--img-pixel-means', type=str, default='(0.0, 0.0, 0.0)')
    parser.add_argument('--img-pixel-stds', type=str, default='(1.0, 1.0, 1.0)')
    parser.add_argument('--net-fixed-params', type=str, default='["conv0", "stage1", "gamma", "beta"]')
    parser.add_argument('--rpn-feat-stride', type=int, default=16)
    parser.add_argument('--rpn-anchor-scales', type=str, default='(8, 16, 32)')
    parser.add_argument('--rpn-anchor-ratios', type=str, default='(0.5, 1, 2)')
    parser.add_argument('--rpn-pre-nms-topk', type=int, default=12000)
    parser.add_argument('--rpn-post-nms-topk', type=int, default=2000)
    parser.add_argument('--rpn-nms-thresh', type=float, default=0.7)
    parser.add_argument('--rpn-min-size', type=int, default=16)
    parser.add_argument('--rpn-batch-rois', type=int, default=256)
    parser.add_argument('--rpn-allowed-border', type=int, default=0)
    parser.add_argument('--rpn-fg-fraction', type=float, default=0.5)
    parser.add_argument('--rpn-fg-overlap', type=float, default=0.7)
    parser.add_argument('--rpn-bg-overlap', type=float, default=0.3)
    parser.add_argument('--rcnn-num-classes', type=int, default=21)
    parser.add_argument('--rcnn-feat-stride', type=int, default=16)
    parser.add_argument('--rcnn-pooled-size', type=str, default='(14, 14)')
    parser.add_argument('--rcnn-batch-size', type=int, default=1)
    parser.add_argument('--rcnn-batch-rois', type=int, default=128)
    parser.add_argument('--rcnn-fg-fraction', type=float, default=0.25)
    parser.add_argument('--rcnn-fg-overlap', type=float, default=0.5)
    parser.add_argument('--rcnn-bbox-stds', type=str, default='(0.1, 0.1, 0.2, 0.2)')
    args = parser.parse_args()
    args.img_pixel_means = ast.literal_eval(args.img_pixel_means)
    args.img_pixel_stds = ast.literal_eval(args.img_pixel_stds)
    args.net_fixed_params = ast.literal_eval(args.net_fixed_params)
    args.rpn_anchor_scales = ast.literal_eval(args.rpn_anchor_scales)
    args.rpn_anchor_ratios = ast.literal_eval(args.rpn_anchor_ratios)
    args.rcnn_pooled_size = ast.literal_eval(args.rcnn_pooled_size)
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    return args


def get_voc(args):
    from symimdb.pascal_voc import PascalVOC
    if not args.imageset:
        args.imageset = '2007_trainval'
    args.rcnn_num_classes = len(PascalVOC.classes)

    isets = args.imageset.split('+')
    roidb = []
    for iset in isets:
        imdb = PascalVOC(iset, 'data', 'data/VOCdevkit')
        imdb.append_flipped_images()
        roidb.extend(imdb.roidb)
    return roidb


def get_coco(args):
    from symimdb.coco import coco
    if not args.imageset:
        args.imageset = 'train2017'
    args.rcnn_num_classes = len(coco.classes)

    isets = args.imageset.split('+')
    roidb = []
    for iset in isets:
        imdb = coco(iset, 'data', 'data/coco')
        imdb.filter_roidb()
        imdb.append_flipped_images()
        roidb.extend(imdb.roidb)
    return roidb


def get_vgg16_train(args):
    from symnet.symbol_vgg import get_vgg_train
    if not args.pretrained:
        args.pretrained = 'model/vgg16-0000.params'
    if not args.save_prefix:
        args.save_prefix = 'model/vgg16'
    args.img_pixel_means = (123.68, 116.779, 103.939)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.net_fixed_params = ['conv1', 'conv2']
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (7, 7)
    return get_vgg_train(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                         rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                         rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                         rpn_min_size=args.rpn_min_size, rpn_batch_rois=args.rpn_batch_rois,
                         num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                         rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                         rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                         rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds)


def get_resnet50_train(args):
    from symnet.symbol_resnet import get_resnet_train
    if not args.pretrained:
        args.pretrained = 'model/resnet-50-0000.params'
    if not args.save_prefix:
        args.save_prefix = 'model/resnet50'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.net_fixed_params = ['conv0', 'stage1', 'gamma', 'beta']
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_train(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                            rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                            rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                            rpn_min_size=args.rpn_min_size, rpn_batch_rois=args.rpn_batch_rois,
                            num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                            rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                            rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                            rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds,
                            units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))


def get_resnet101_train(args):
    from symnet.symbol_resnet import get_resnet_train
    if not args.pretrained:
        args.pretrained = 'model/resnet-101-0000.params'
    if not args.save_prefix:
        args.save_prefix = 'model/resnet101'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.net_fixed_params = ['conv0', 'stage1', 'gamma', 'beta']
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_train(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                            rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                            rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                            rpn_min_size=args.rpn_min_size, rpn_batch_rois=args.rpn_batch_rois,
                            num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                            rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                            rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                            rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds,
                            units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))


def get_dataset(dataset, args):
    datasets = {
        'voc': get_voc,
        'coco': get_coco
    }
    if dataset not in datasets:
        raise ValueError("dataset {} not supported".format(dataset))
    return datasets[dataset](args)


def get_network(network, args):
    networks = {
        'vgg16': get_vgg16_train,
        'resnet50': get_resnet50_train,
        'resnet101': get_resnet101_train
    }
    if network not in networks:
        raise ValueError("network {} not supported".format(network))
    return networks[network](args)


def main():
    args = parse_args()
    roidb = get_dataset(args.dataset, args)
    sym = get_network(args.network, args)
    train_net(sym, roidb, args)


if __name__ == '__main__':
    main()
