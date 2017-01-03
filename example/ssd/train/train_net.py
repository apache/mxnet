import tools.find_mxnet
import mxnet as mx
import logging
import sys
import os
import importlib
from initializer import ScaleInitializer
from metric import MultiBoxMetric
from dataset.iterator import DetIter
from dataset.pascal_voc import PascalVoc
from dataset.concat_db import ConcatDB
from config.config import cfg

def load_pascal(image_set, year, devkit_path, shuffle=False):
    """
    wrapper function for loading pascal voc dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    year : str
        2007, 2012 or combinations splitted by comma
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"
    year = [y.strip() for y in year.split(',')]
    assert year, "No year specified"

    # make sure (# sets == # years)
    if len(image_set) > 1 and len(year) == 1:
        year = year * len(image_set)
    if len(image_set) == 1 and len(year) > 1:
        image_set = image_set * len(year)
    assert len(image_set) == len(year), "Number of sets and year mismatch"

    imdbs = []
    for s, y in zip(image_set, year):
        imdbs.append(PascalVoc(s, y, devkit_path, shuffle, is_train=True))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]

def convert_pretrained(name, args):
    """
    Special operations need to be made due to name inconsistance, etc

    Parameters:
    ---------
    args : dict
        loaded arguments

    Returns:
    ---------
    processed arguments as dict
    """
    if name == 'vgg16_reduced':
        args['conv6_bias'] = args.pop('fc6_bias')
        args['conv6_weight'] = args.pop('fc6_weight')
        args['conv7_bias'] = args.pop('fc7_bias')
        args['conv7_weight'] = args.pop('fc7_weight')
        del args['fc8_weight']
        del args['fc8_bias']
    return args

def train_net(net, dataset, image_set, year, devkit_path, batch_size,
               data_shape, mean_pixels, resume, finetune, pretrained, epoch, prefix,
               ctx, begin_epoch, end_epoch, frequent, learning_rate,
               momentum, weight_decay, val_set, val_year,
               lr_refactor_epoch, lr_refactor_ratio,
               iter_monitor=0, log_file=None):
    """
    Wrapper for training module

    Parameters:
    ---------
    net : mx.Symbol
        training network
    dataset : str
        pascal, imagenet...
    image_set : str
        train, trainval...
    year : str
        2007, 2012 or combinations splitted by comma
    devkit_path : str
        root directory of dataset
    batch_size : int
        training batch size
    data_shape : int or (int, int)
        resize image size
    mean_pixels : tuple (float, float, float)
        mean pixel values in (R, G, B)
    resume : int
        if > 0, will load trained epoch with name given by prefix
    finetune : int
        if > 0, will load trained epoch with name given by prefix, in this mode
        all convolutional layers except the last(prediction layer) are fixed
    pretrained : str
        prefix of pretrained model name
    epoch : int
        epoch of pretrained model
    prefix : str
        prefix of new model
    ctx : mx.gpu(?) or list of mx.gpu(?)
        training context
    begin_epoch : int
        begin epoch, default should be 0
    end_epoch : int
        when to stop training
    frequent : int
        frequency to log out batch_end_callback
    learning_rate : float
        learning rate, will be divided by batch_size automatically
    momentum : float
        (0, 1), training momentum
    weight_decay : float
        decay weights regardless of gradient
    val_set : str
        similar to image_set, used for validation
    val_year : str
        similar to year, used for validation
    lr_refactor_epoch : int
        number of epoch to change learning rate
    lr_refactor_ratio : float
        new_lr = old_lr * lr_refactor_ratio
    iter_monitor : int
        if larger than 0, will print weights/gradients every iter_monitor iters
    log_file : str
        log to file if not None

    Returns:
    ---------
    None
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # kvstore
    kv = mx.kvstore.create("device")

    # check args
    if isinstance(data_shape, int):
        data_shape = (data_shape, data_shape)
    assert len(data_shape) == 2, "data_shape must be (h, w) tuple or list or int"
    prefix += '_' + str(data_shape[0])

    if isinstance(mean_pixels, (int, float)):
        mean_pixels = [mean_pixels, mean_pixels, mean_pixels]
    assert len(mean_pixels) == 3, "must provide all RGB mean values"

    # load dataset
    if dataset == 'pascal':
        imdb = load_pascal(image_set, year, devkit_path, cfg.TRAIN.INIT_SHUFFLE)
        if val_set and val_year:
            val_imdb = load_pascal(val_set, val_year, devkit_path, False)
        else:
            val_imdb = None
    else:
        raise NotImplementedError("Dataset " + dataset + " not supported")

    # init data iterator
    train_iter = DetIter(imdb, batch_size, data_shape, mean_pixels,
                         cfg.TRAIN.RAND_SAMPLERS, cfg.TRAIN.RAND_MIRROR,
                         cfg.TRAIN.EPOCH_SHUFFLE, cfg.TRAIN.RAND_SEED,
                         is_train=True)
    # save per N epoch, avoid saving too frequently
    resize_epoch = int(cfg.TRAIN.RESIZE_EPOCH)
    if resize_epoch > 1:
        batches_per_epoch = ((imdb.num_images - 1) // batch_size + 1) * resize_epoch
        train_iter = mx.io.ResizeIter(train_iter, batches_per_epoch)
    train_iter = mx.io.PrefetchingIter(train_iter)
    if val_imdb:
        val_iter = DetIter(val_imdb, batch_size, data_shape, mean_pixels,
                           cfg.VALID.RAND_SAMPLERS, cfg.VALID.RAND_MIRROR,
                           cfg.VALID.EPOCH_SHUFFLE, cfg.VALID.RAND_SEED,
                           is_train=True)
        val_iter = mx.io.PrefetchingIter(val_iter)
    else:
        val_iter = None

    # load symbol
    sys.path.append(os.path.join(cfg.ROOT_DIR, 'symbol'))
    net = importlib.import_module("symbol_" + net).get_symbol_train(imdb.num_classes)

    # define layers with fixed weight/bias
    fixed_param_names = [name for name in net.list_arguments() \
        if name.startswith('conv1_') or name.startswith('conv2_')]

    # load pretrained or resume from previous state
    ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
    if resume > 0:
        logger.info("Resume training with {} from epoch {}"
            .format(ctx_str, resume))
        _, args, auxs = mx.model.load_checkpoint(prefix, resume)
        begin_epoch = resume
    elif finetune > 0:
        logger.info("Start finetuning with {} from epoch {}"
            .format(ctx_str, finetune))
        _, args, auxs = mx.model.load_checkpoint(prefix, finetune)
        begin_epoch = finetune
        # the prediction convolution layers name starts with relu, so it's fine
        fixed_param_names = [name for name in net.list_arguments() \
            if name.startswith('conv')]
    elif pretrained:
        logger.info("Start training with {} from pretrained model {}"
            .format(ctx_str, pretrained))
        _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)
        args = convert_pretrained(pretrained, args)
    else:
        logger.info("Experimental: start training from scratch with {}"
            .format(ctx_str))
        args = None
        auxs = None
        fixed_param_names = None

    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    # init training module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
                        fixed_param_names=fixed_param_names)

    # fit
    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    iter_refactor = lr_refactor_epoch * imdb.num_images // train_iter.batch_size
    lr_scheduler = mx.lr_scheduler.FactorScheduler(iter_refactor, lr_refactor_ratio)
    optimizer_params={'learning_rate':learning_rate,
                      'momentum':momentum,
                      'wd':weight_decay,
                      'lr_scheduler':lr_scheduler,
                      'clip_gradient':None,
                      'rescale_grad': 1.0}
    monitor = mx.mon.Monitor(iter_monitor, pattern=".*") if iter_monitor > 0 else None
    initializer = mx.init.Mixed([".*scale", ".*"], \
        [ScaleInitializer(), mx.init.Xavier(magnitude=1)])

    mod.fit(train_iter,
            eval_data=val_iter,
            eval_metric=MultiBoxMetric(),
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            optimizer='sgd',
            optimizer_params=optimizer_params,
            kvstore = kv,
            begin_epoch=begin_epoch,
            num_epoch=end_epoch,
            initializer=initializer,
            arg_params=args,
            aux_params=auxs,
            allow_missing=True,
            monitor=monitor)
