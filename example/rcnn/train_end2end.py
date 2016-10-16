import argparse
import logging
import os
import mxnet as mx
from rcnn.callback import Speedometer
from rcnn.config import config
from rcnn.loader import AnchorLoader
from rcnn.metric import AccuracyMetric, LogLossMetric, SmoothL1LossMetric
from rcnn.module import MutableModule
from rcnn.symbol import get_faster_rcnn
from utils.load_data import load_gt_roidb
from utils.load_model import do_checkpoint, load_param

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def end2end_train(image_set, test_image_set, year, root_path, devkit_path, pretrained, epoch, prefix,
                  ctx, begin_epoch, num_epoch, frequent, kv_store, mom, wd, lr, num_classes, monitor,
                  work_load_list=None, resume=False, use_flip=True, factor_step=50000):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    mon = None
    config.TRAIN.BG_THRESH_HI = 0.5  # TODO(verify)
    config.TRAIN.BG_THRESH_LO = 0.0  # TODO(verify)
    config.TRAIN.RPN_MIN_SIZE = 16

    logging.info('########## TRAIN FASTER-RCNN WITH APPROXIMATE JOINT END2END #############')
    config.TRAIN.HAS_RPN = True
    config.END2END = 1
    config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
    sym = get_faster_rcnn(num_classes=num_classes)
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    config.TRAIN.IMS_PER_BATCH *= len(ctx)
    config.TRAIN.BATCH_SIZE *= len(ctx)  # no used here

    # infer max shape
    max_data_shape = [('data', (config.TRAIN.IMS_PER_BATCH, 3, 1000, 1000))]
    max_data_shape_dict = {k: v for k, v in max_data_shape}
    _, feat_shape, _ = feat_sym.infer_shape(**max_data_shape_dict)
    from rcnn.minibatch import assign_anchor
    import numpy as np
    label = assign_anchor(feat_shape[0], np.zeros((0, 5)), [[1000, 1000, 1.0]])
    max_label_shape = [('label', label['label'].shape),
                       ('bbox_target', label['bbox_target'].shape),
                       ('bbox_inside_weight', label['bbox_inside_weight'].shape),
                       ('bbox_outside_weight', label['bbox_outside_weight'].shape),
                       ('gt_boxes', (config.TRAIN.IMS_PER_BATCH, 5*100))]  # assume at most 100 object in image
    print 'providing maximum shape', max_data_shape, max_label_shape

    # load training data
    voc, roidb = load_gt_roidb(image_set, year, root_path, devkit_path, flip=use_flip)
    train_data = AnchorLoader(feat_sym, roidb, batch_size=config.TRAIN.IMS_PER_BATCH, shuffle=True, mode='train',
                              ctx=ctx, work_load_list=work_load_list)
    # load pretrained
    args, auxs, _ = load_param(pretrained, epoch, convert=True)

    # initialize params
    if not resume:
        del args['fc8_weight']
        del args['fc8_bias']
        input_shapes = {k: (1,)+ v[1::] for k, v in train_data.provide_data + train_data.provide_label}
        arg_shape, _, _ = sym.infer_shape(**input_shapes)
        arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))

        args['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
        args['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
        args['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
        args['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
        args['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['rpn_bbox_pred_weight'])  # guarantee not likely explode with bbox_delta
        args['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
        args['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
        args['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
        args['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['bbox_pred_weight'])
        args['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

    # prepare training
    if config.TRAIN.FINETUNE:
        fixed_param_prefix = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    else:
        fixed_param_prefix = ['conv1', 'conv2']
    data_names = [k[0] for k in train_data.provide_data]
    label_names = [k[0] for k in train_data.provide_label]
    batch_end_callback = Speedometer(train_data.batch_size, frequent=frequent)
    epoch_end_callback = do_checkpoint(prefix)
    rpn_eval_metric = AccuracyMetric(use_ignore=True, ignore=-1, ex_rpn=True)
    rpn_cls_metric = LogLossMetric(use_ignore=True, ignore=-1, ex_rpn=True)
    rpn_bbox_metric = SmoothL1LossMetric(ex_rpn=True)
    eval_metric = AccuracyMetric()
    cls_metric = LogLossMetric()
    bbox_metric = SmoothL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    optimizer_params = {'momentum': mom,
                        'wd': wd,
                        'learning_rate': lr,
                        'lr_scheduler': mx.lr_scheduler.FactorScheduler(factor_step, 0.1),
                        'clip_gradient': 1.0,
                        'rescale_grad': 1.0 }
                        # 'rescale_grad': (1.0 / config.TRAIN.RPN_BATCH_SIZE)}
    # train
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=work_load_list,
                        max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                        fixed_param_prefix=fixed_param_prefix)
    if monitor:
        def norm_stat(d):
            return mx.nd.norm(d)/np.sqrt(d.size)
        mon = mx.mon.Monitor(100, norm_stat)

    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=kv_store,
            optimizer='sgd', optimizer_params=optimizer_params, monitor=mon,
            arg_params=args, aux_params=auxs, begin_epoch=begin_epoch, num_epoch=num_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN Network')
    parser.add_argument('--image_set', dest='image_set', help='can be trainval or train',
                        default='trainval', type=str)
    parser.add_argument('--num-classes', dest='num_classes', help='the class number of dataset',
                        default=21, type=int)
    parser.add_argument('--test_image_set', dest='test_image_set', help='can be test or val',
                        default='test', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2010, 2012',
                        default='2007', type=str)
    parser.add_argument('--no-flip', action='store_true', default=False,
                        help='if true, then will flip the dataset')
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='VOCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16'), type=str)
    parser.add_argument('--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'faster-rcnn'), type=str)
    parser.add_argument('--gpus', dest='gpu_ids', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--num_epoch', dest='num_epoch', help='end epoch of faster rcnn end2end training',
                        default=7, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--kv_store', dest='kv_store', help='the kv-store type',
                        default='device', type=str)
    parser.add_argument('--work_load_list', dest='work_load_list', help='work load for different devices',
                        default=None, type=list)
    parser.add_argument('--lr', type=float, default=0.001, help='initialization learning reate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay for sgd')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='if true, then will retrain the model from rcnn')
    parser.add_argument('--factor-step',type=int, default=50000, help='the step used for lr factor')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='if true, then will use monitor debug')
    args = parser.parse_args()
    logging.info(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    end2end_train(args.image_set, args.test_image_set, args.year, args.root_path, args.devkit_path,
                  args.pretrained, args.load_epoch, args.prefix, ctx, args.load_epoch, args.num_epoch,
                  args.frequent, args.kv_store, args.mom, args.wd, args.lr, args.num_classes, args.monitor,
                  args.work_load_list, args.resume, not args.no_flip, args.factor_step)
