import os
import argparse
import mxnet as mx
import numpy as np
import logging
from data_iter import FileIter
from symbols import segnet, segnet_bn
from common import contrib_metrics

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _load_model_for_vgg16(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch_for_vgg16 is None:
        return (None, None, None)
    assert args.model_prefix_for_vgg16 is not None
    model_prefix = args.model_prefix_for_vgg16
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch_for_vgg16)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch_for_vgg16)
    return (sym, arg_params, aux_params)

def _load_vgg_model(args, rank=0):
    sym, arg_params, aux_params = _load_model(args, rank)
    _, arg_params_vgg, aux_params_vgg = _load_model_for_vgg16(args, rank)
    for name, val in arg_params_vgg.items():
        if arg_params.get(name) is not None:
            arg_params[name] = val
    for name, val in aux_params_vgg.items():
        if aux_params.get(name) is not None:
            aux_params[name] = val
    return (sym, arg_params, aux_params)

def train(args):
    """
    train segnet
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if args.log_file:
        file_path = logging.FileHandler(args.log_file)
        logger.addHandler(file_path)

    if args.gpus is None or args.gpus is '':
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    train_dataiter = FileIter(
        batch_size=args.batch_size,
        root_dir=args.data_path,
        flist_name=args.train_file,
        rgb_mean=[float(i) for i in args.rgb_mean.split(',')],
        shuffle=True
        )
    val_dataiter = FileIter(
        batch_size=args.batch_size,
        root_dir=args.data_path,
        flist_name=args.val_file,
        rgb_mean=[float(i) for i in args.rgb_mean.split(',')],
        shuffle=False)
    # load model
    from importlib import import_module
    network = import_module('symbols.'+args.network).get_symbol(args.num_classes)
    # callback fuc
    batch_end_callback = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    epoch_end_callback = mx.callback.do_checkpoint(args.model_prefix, args.save_frequency)
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    # optimizer params
    if 'lr_factor' not in args or args.lr_factor >= 1:
        lr_scheduler = None
    else:
        steps = [int(l) for l in args.lr_steps.split(',')]
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor)
    optimizer_params = {'learning_rate': args.lr,
                        'multi_precision': True,
                        'momentum' : args.mom,
                        'wd' : args.wd,
                        'lr_scheduler': lr_scheduler}
    # evaluation metrices
    eval_metrics = [contrib_metrics.Accuracy(ignore_label=11),
                    contrib_metrics.CrossEntropy()]
    # parameter init fuc
    softmax_weight = mx.nd.array(np.array([0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826,
                                           9.6446, 1.8418, 0.6823, 6.2478, 7.3614]))
    height = train_dataiter.data[0][1].shape[2]
    width = train_dataiter.data[0][1].shape[3]
    softmax_weight = softmax_weight.reshape((1, 11, 1, 1))
    softmax_weight = softmax_weight.broadcast_to((args.batch_size/len(ctx), 11, height, width))
    xavier = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    initializer = mx.init.Mixed(['softmax_weight', '.*'],
                                [mx.init.Load({'softmax_weight':softmax_weight}), xavier])
    # load model
    if 'load_epoch' not in args or args.load_epoch_for_vgg16 is not None:
        sym, arg_params, aux_params = _load_vgg_model(args, kv.rank)
    else:
        sym, arg_params, aux_params = _load_model(args, kv.rank)

    if sym is not None:
        assert sym.tojson() == network.tojson()
    # create model
    model = mx.mod.Module(context=ctx, symbol=network, logger=logger)

    model.fit(
        train_data=train_dataiter,
        begin_epoch=args.load_epoch if args.load_epoch else 0,
        eval_data=val_dataiter,
        num_epoch=args.num_epochs,
        eval_metric=eval_metrics,
        kvstore=kv,
        optimizer=args.optimizer,
        optimizer_params=optimizer_params,
        initializer=initializer,
        arg_params=arg_params,
        aux_params=aux_params,
        batch_end_callback=batch_end_callback,
        epoch_end_callback=epoch_end_callback,
        allow_missing=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vgg16 model to vgg16fc model.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    parser.add_argument('--kv_store', type=str, default='device',
                        help='key-value store type')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--disp_batches', type=int, default=5,
                        help='show progress for every n batches')
    parser.add_argument('--data_path', type=str, default="/data/CamVid",
                        help='data path')
    parser.add_argument('--train_file', type=str, default="train.txt",
                        help='train data list file name in data path')
    parser.add_argument('--val_file', type=str, default="test.txt",
                        help='val data list file name in data path')
    parser.add_argument('--num_classes', type=int, default=11,
                        help='num classes to clissufy')
    parser.add_argument('--model', default='fcnxs',
                        help='The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--model_prefix', type=str, default='models/segnet',
                        help='model prefix')
    parser.add_argument('--load_epoch', type=int,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--model_prefix_for_vgg16', type=str, default='vgg16',
                        help='model prefix')
    parser.add_argument('--load_epoch_for_vgg16', type=int,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--init-type', default="vgg16",
                        help='the init type of fcn-xs model, e.g. vgg16, fcnxs')
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='true means continue training.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='initial learning rate')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='the ratio to reduce lr on each step')
    parser.add_argument('--lr_steps', type=str, default='2000, 3000',
                        help='the epochs to reduce the lr, e.g. 30,60')
    parser.add_argument('--save_frequency', type=int, default=5,
                        help='the frequency to save model')
    parser.add_argument('--mom', type=float, default=0.9,
                        help='momentum for sgd')
    parser.add_argument('--wd', type=float, default=0.001,
                        help='weight decay for sgd')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='the optimizer type')
    parser.add_argument('--log_file', type=str, default="log.txt",
                        help='the name of log file')
    parser.add_argument('--rgb_mean', type=str, default='123.68, 116.779, 103.939',
                        help='rgb mean')
    parser.add_argument('--network', type=str, required=True,
                        help='the neural network to use')
    args = parser.parse_args()
    logging.info(args)
    train(args)