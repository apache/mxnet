from common import modelzoo
import mxnet as mx
import os
import logging

def train_imagenet():

    batch_size = 128
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
    mod = mx.mod.Module(symbol = sym, context = [mx.gpu(i) for i in range(8)],
        label_names = [label_name,])
    begin_epoch = 0
    num_epoch = 1
    eval_metric = ['accuracy']
    kv = 'local'
    optimizer = 'sgd'
    optimizer_params = {
            'learning_rate'     : 0.1,
            #'lr_scheduler'      : lr_scheduler,
            'weight_sparsity'   : [0],
            'bias_sparsity'     : [0],
            'switch_epoch'      : [0,0],
            'batches_per_epoch' : 100,
            'do_pruning'        : False}
    disp_batches = 20
    batch_end_callbacks = [mx.callback.Speedometer(batch_size, disp_batches)]
    mod.fit(val,
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

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_imagenet()
