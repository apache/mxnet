import find_mxnet
import mxnet as mx
import logging

def fit(args, network, data_loader):
    # kvstore
    kv = mx.kvstore.create(args.kv_type)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # load model?
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    load_model = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        load_model = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
    # save model?
    checkpoint = None if model_prefix is None else mx.callback.do_checkpoint(model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu()
    if args.gpus is not None:
        devs = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.00001,
        **load_model)
    model.fit(
        X                  = train,
        eval_data          = val,
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 50),
        epoch_end_callback = checkpoint)
