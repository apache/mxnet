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
              prefix, ctx, begin_epoch, end_epoch, frequent, kv_store, work_load_list=None):
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
    train_data = ROIIter(roidb, ctx=ctx,  batch_size=config.TRAIN.BATCH_IMAGES, shuffle=True, mode='train', work_load_list=work_load_list)

    # load pretrained
    args, auxs = load_param(pretrained, epoch, convert=True, ctx=ctx[0])
    del args['fc8_bias']
    del args['fc8_weight']

    # load symbol
    sym = get_symbol_vgg()

    # initialize params
    arg_shape, _, _ = sym.infer_shape(data=(1, 3, 224, 224), rois=(1, 5))
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    args['cls_score_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['cls_score_weight'], ctx=ctx[0])
    args['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'], ctx=ctx[0])
    args['bbox_pred_weight'] = mx.random.normal(mean=0, stdvar=0.001, shape=arg_shape_dict['bbox_pred_weight'], ctx=ctx[0])
    args['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'], ctx=ctx[0])

    # train
    solver = Solver(prefix, sym, ctx, begin_epoch, end_epoch, kv_store, args, auxs, momentum=0.9, wd=0.0005,
                    learning_rate=0.001, lr_scheduler=mx.lr_scheduler.FactorScheduler(30000, 0.1), max_data_shape=[3, 1000, 1000])
    solver.fit(train_data, frequent=frequent)

    # edit params and save
    for epoch in range(begin_epoch + 1, end_epoch + 1):
        arg_params, aux_params = load_checkpoint(prefix, epoch)
        arg_params['bbox_pred_weight'] = (arg_params['bbox_pred_weight'].T * mx.nd.array(stds, ctx=ctx[0])).T
        arg_params['bbox_pred_bias'] = arg_params['bbox_pred_bias'] * mx.nd.array(stds, ctx=ctx[0]) + \
                                       mx.nd.array(means, ctx=ctx[0])
        save_checkpoint(prefix, epoch, arg_params, aux_params)
