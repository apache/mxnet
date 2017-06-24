from common import modelzoo
import mxnet as mx
import os
import logging
import math
import argparse

def train_imagenet(args):

    # arguments to change
    lr = args.lr
    lr_steps = [int(i) for i in args.lr_steps.split(',')]
    lr_factor = args.lr_factor
    weight_sparsity = [float(i) for i in args.weight_sparsity.split(',')]
    bias_sparsity = [float(i) for i in args.bias_sparsity.split(',')]
    switch_epoch = [int(i) for i in args.switch_epoch.split(',')]
    do_pruning = args.do_pruning
    gpus = [int(i) for i in args.gpus.split(',')]

    # fixed arguments
    num_epoch = 20
    batch_size = 512
    label_name = 'softmax_label'

    # create data iterators
    train = mx.io.ImageRecordIter(
        path_imgrec        = '/home/ubuntu/data/train_480_q90.rec',
        label_width        = 1,
        preprocess_threads = 4,
        batch_size         = batch_size,
        data_shape         = (3,224,224),
        label_name         = label_name,
        rand_crop          = False,
        rand_mirror        = False,
        mean_r             = 0.0,
        mean_g             = 0.0,
        mean_b             = 0.0)
    val = mx.io.ImageRecordIter(
        path_imgrec        = '/home/ubuntu/data/val_480_q90.rec',
        label_width        = 1,
        preprocess_threads = 4,
        batch_size         = batch_size,
        data_shape         = (3,224,224),
        label_name         = label_name,
        rand_crop          = False,
        rand_mirror        = False,
        mean_r             = 0.0,
        mean_g             = 0.0,
        mean_b             = 0.0)

    # download model
    model = 'imagenet1k-resnet-18'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    (prefix, epoch) = modelzoo.download_model(model, os.path.join(dir_path, 'model'))
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # training
    context = [mx.gpu(i) for i in gpus]
    mod = mx.mod.Module(symbol = sym, context = context, label_names = [label_name,])
    begin_epoch = 0
    eval_metric = ['accuracy']
    kv = 'local'
    optimizer = 'sgd'
    batches_per_epoch = math.ceil(1281167.0 / batch_size)
    lr_steps = [batches_per_epoch * i for i in lr_steps]
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step = lr_steps,
        factor = lr_factor)
    start_pruning = True
    optimizer_params = {
            'learning_rate'     : lr,
            'lr_scheduler'      : lr_scheduler,
            'weight_sparsity'   : weight_sparsity,
            'bias_sparsity'     : bias_sparsity,
            'switch_epoch'      : switch_epoch,
            'batches_per_epoch' : batches_per_epoch,
            'do_pruning'        : do_pruning,
            'start_prune'       : start_pruning,
    }
    disp_batches = 50
    batch_end_callbacks = [mx.callback.Speedometer(batch_size, disp_batches)]
    mod.fit(train,
        begin_epoch                 = begin_epoch,
        num_epoch                   = num_epoch,
        eval_data                   = val,
        eval_metric                 = eval_metric,
        kvstore                     = kv,
        optimizer                   = optimizer,
        optimizer_params            = optimizer_params,
        #initializer                 = initializer,
        arg_params                  = arg_params,
        aux_params                  = aux_params,
        batch_end_callback          = batch_end_callbacks,
        #epoch_end_callback          = None,
        allow_missing               = True,
        #monitor                     = None,
        #eval_end_callback           = None,
        #eval_batch_end_callback     = None
    )

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'train Imagenet')
    parser.add_argument('--lr', type = float)
    parser.add_argument('--lr_steps', type = str)
    parser.add_argument('--lr_factor', type = float)
    parser.add_argument('--weight_sparsity', type = str)
    parser.add_argument('--bias_sparsity', type = str)
    parser.add_argument('--switch_epoch', type = str)
    parser.add_argument('--do_pruning', type = bool)
    parser.add_argument('--gpus', type = str)
    args = parser.parse_args()

    print args

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_imagenet(args)
