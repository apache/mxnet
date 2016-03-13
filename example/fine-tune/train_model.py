import find_mxnet
import mxnet as mx
import logging
import os

def fit(args, network, data_loader):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)
        logger = logging

    # load model
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}

    if args.finetune_from is not None:
        assert args.load_epoch is None
        finetune_from_prefix, finetune_from_epoch = args.finetune_from.rsplit('-', 1)
        finetune_from_epoch = int(finetune_from_epoch)
        logger.info('finetune from %s at epoch %d', finetune_from_prefix, finetune_from_epoch)
        tmp = mx.model.FeedForward.load(finetune_from_prefix, finetune_from_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params} 

    # save model
    checkpoint = None if model_prefix is None else mx.callback.do_checkpoint(model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    # optimizer
    batch_size = args.batch_size
    # reference: model.FeedForward.fit()
    if kv and kv.type == 'dist_sync':
        batch_size *= kv.num_workers
    optimizer = mx.optimizer.create('sgd',
        rescale_grad=(1.0/batch_size),
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.00001,)

    # lr_scale
    if args.finetune_from is not None:
        # convention: for argument param_name, if args.dataset in param_name, then it is
        # to be fine-tuned
        lr_scale = {}
        net_args = network.list_arguments()
        for i, name in enumerate(net_args):
            if args.dataset in name:
                lr_scale[i] = args.finetune_lr_scale
        optimizer.set_lr_scale(lr_scale)
        logger.info('lr_scale: %s', {net_args[i]: s for i,s in lr_scale.items()})

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        # learning_rate      = args.lr,
        # momentum           = 0.9,
        # wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        optimizer          = optimizer,
        **model_args)

    model.fit(
        X                  = train,
        eval_data          = val,
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 50),
        epoch_end_callback = checkpoint)
