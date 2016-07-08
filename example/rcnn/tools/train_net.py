import mxnet as mx
import logging
from rcnn.config import config
from load_data import load_train_roidb
from rcnn.data_iter import ROIIter
from rcnn.symbol import get_symbol_vgg
from load_model import load_checkpoint, load_param
from rcnn.solver import Solver
from save_model import save_checkpoint


def train_net(image_set, year, root_path, devkit_path, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent):
    """
    wrapper for solver
    :param image_set: image set to train on
    :param year: year of image set
    :param root_path: 'data' folder
    :param devkit_path: 'VOCdevkit' folder
    :param pretrained: prefix of pretrained model
    :param epoch: epoch of pretrained model
    :param prefix: prefix of new model
    :param ctx: context to train in
    :param begin_epoch: begin epoch number
    :param end_epoch: end epoch number
    :param frequent: frequency to print
    :return: None
    """
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # load training data
    voc, roidb, means, stds = load_train_roidb(image_set, year, root_path, devkit_path, flip=True)
    train_data = ROIIter(roidb, batch_size=config.TRAIN.BATCH_IMAGES, shuffle=True, mode='train')

    # load pretrained
    args, auxs = load_param(pretrained, epoch, convert=True, ctx=ctx)
    del args['fc8_bias']
    del args['fc8_weight']

    # load symbol
    sym = get_symbol_vgg()

    # initialize params
    arg_shape, _, _ = sym.infer_shape(data=(1, 3, 224, 224), rois=(1, 5))
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    args['cls_score_weight'] = mx.random.normal(loc=0, scale=0.01, shape=arg_shape_dict['cls_score_weight'], ctx=ctx)
    args['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'], ctx=ctx)
    args['bbox_pred_weight'] = mx.random.normal(loc=0, scale=0.001, shape=arg_shape_dict['bbox_pred_weight'], ctx=ctx)
    args['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'], ctx=ctx)

    # train
    solver = Solver(prefix, sym, ctx, begin_epoch, end_epoch, args, auxs, momentum=0.9, wd=0.0005,
                    learning_rate=0.001, lr_scheduler=mx.lr_scheduler.FactorScheduler(30000, 0.1))
    solver.fit(train_data, frequent=frequent)

    # edit params and save
    for epoch in range(begin_epoch + 1, end_epoch + 1):
        arg_params, aux_params = load_checkpoint(prefix, epoch)
        arg_params['bbox_pred_weight'] = (arg_params['bbox_pred_weight'].T * mx.nd.array(stds, ctx=ctx)).T
        arg_params['bbox_pred_bias'] = arg_params['bbox_pred_bias'] * mx.nd.array(stds, ctx=ctx) + \
                                       mx.nd.array(means, ctx=ctx)
        save_checkpoint(prefix, epoch, arg_params, aux_params)
