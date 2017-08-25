import sys

sys.path.insert(0, "../../python")
import os.path
import mxnet as mx
from config_util import get_checkpoint_path, parse_contexts
from stt_metric import STTMetric
#tensorboard setting
from tensorboard import SummaryWriter
import numpy as np


def get_initializer(args):
    init_type = getattr(mx.initializer, args.config.get('train', 'initializer'))
    init_scale = args.config.getfloat('train', 'init_scale')
    if init_type is mx.initializer.Xavier:
        return mx.initializer.Xavier(magnitude=init_scale, factor_type=args.config.get('train', 'factor_type'))
    return init_type(init_scale)

class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
    """A simple lr schedule that simply return `dynamic_lr`. We will set `dynamic_lr`
    dynamically based on performance on the validation set.
    """
    def __init__(self, learning_rate=0.001):
        super(SimpleLRScheduler, self).__init__()
        self.learning_rate = learning_rate

    def __call__(self, num_update):
        return self.learning_rate

def do_training(args, module, data_train, data_val, begin_epoch=0):
    from distutils.dir_util import mkpath
    from log_util import LogUtil

    log = LogUtil().getlogger()
    mkpath(os.path.dirname(get_checkpoint_path(args)))

    seq_len = args.config.get('arch', 'max_t_count')
    batch_size = args.config.getint('common', 'batch_size')
    save_checkpoint_every_n_epoch = args.config.getint('common', 'save_checkpoint_every_n_epoch')
    save_checkpoint_every_n_batch = args.config.getint('common', 'save_checkpoint_every_n_batch')
    enable_logging_train_metric = args.config.getboolean('train', 'enable_logging_train_metric')
    enable_logging_validation_metric = args.config.getboolean('train', 'enable_logging_validation_metric')

    contexts = parse_contexts(args)
    num_gpu = len(contexts)
    eval_metric = STTMetric(batch_size=batch_size, num_gpu=num_gpu, seq_length=seq_len,is_logging=enable_logging_validation_metric,is_epoch_end=True)
    # tensorboard setting
    loss_metric = STTMetric(batch_size=batch_size, num_gpu=num_gpu, seq_length=seq_len,is_logging=enable_logging_train_metric,is_epoch_end=False)

    optimizer = args.config.get('train', 'optimizer')
    momentum = args.config.getfloat('train', 'momentum')
    learning_rate = args.config.getfloat('train', 'learning_rate')
    learning_rate_annealing = args.config.getfloat('train', 'learning_rate_annealing')

    mode = args.config.get('common', 'mode')
    num_epoch = args.config.getint('train', 'num_epoch')
    clip_gradient = args.config.getfloat('train', 'clip_gradient')
    weight_decay = args.config.getfloat('train', 'weight_decay')
    save_optimizer_states = args.config.getboolean('train', 'save_optimizer_states')
    show_every = args.config.getint('train', 'show_every')
    n_epoch=begin_epoch

    if clip_gradient == 0:
        clip_gradient = None

    module.bind(data_shapes=data_train.provide_data,
                label_shapes=data_train.provide_label,
                for_training=True)

    if begin_epoch == 0 and mode == 'train':
        module.init_params(initializer=get_initializer(args))


    lr_scheduler = SimpleLRScheduler(learning_rate=learning_rate)

    def reset_optimizer(force_init=False):
        if optimizer == "sgd":
            module.init_optimizer(kvstore='device',
                                  optimizer=optimizer,
                                  optimizer_params={'lr_scheduler': lr_scheduler,
                                                    'momentum': momentum,
                                                    'clip_gradient': clip_gradient,
                                                    'wd': weight_decay},
                                  force_init=force_init)
        elif optimizer == "adam":
            module.init_optimizer(kvstore='device',
                                  optimizer=optimizer,
                                  optimizer_params={'lr_scheduler': lr_scheduler,
                                                    #'momentum': momentum,
                                                    'clip_gradient': clip_gradient,
                                                    'wd': weight_decay},
                                  force_init=force_init)
        else:
            raise Exception('Supported optimizers are sgd and adam. If you want to implement others define them in train.py')
    if mode == "train":
        reset_optimizer(force_init=True)
    else:
        reset_optimizer(force_init=False)

    #tensorboard setting
    tblog_dir = args.config.get('common', 'tensorboard_log_dir')
    summary_writer = SummaryWriter(tblog_dir)
    while True:

        if n_epoch >= num_epoch:
            break

        loss_metric.reset()
        log.info('---------train---------')
        for nbatch, data_batch in enumerate(data_train):

            module.forward_backward(data_batch)
            module.update()
            # tensorboard setting
            if (nbatch + 1) % show_every == 0:
                module.update_metric(loss_metric, data_batch.label)
            #summary_writer.add_scalar('loss batch', loss_metric.get_batch_loss(), nbatch)
            if (nbatch+1) % save_checkpoint_every_n_batch == 0:
                log.info('Epoch[%d] Batch[%d] SAVE CHECKPOINT', n_epoch, nbatch)
                module.save_checkpoint(prefix=get_checkpoint_path(args)+"n_epoch"+str(n_epoch)+"n_batch", epoch=(int((nbatch+1)/save_checkpoint_every_n_batch)-1), save_optimizer_states=save_optimizer_states)
        # commented for Libri_sample data set to see only train cer
        log.info('---------validation---------')
        data_val.reset()
        eval_metric.reset()
        for nbatch, data_batch in enumerate(data_val):
            # when is_train = False it leads to high cer when batch_norm
            module.forward(data_batch, is_train=True)
            module.update_metric(eval_metric, data_batch.label)

        # tensorboard setting
        val_cer, val_n_label, val_l_dist, _ = eval_metric.get_name_value()
        log.info("Epoch[%d] val cer=%f (%d / %d)", n_epoch, val_cer, int(val_n_label - val_l_dist), val_n_label)
        curr_acc = val_cer
        summary_writer.add_scalar('CER validation', val_cer, n_epoch)
        assert curr_acc is not None, 'cannot find Acc_exclude_padding in eval metric'

        data_train.reset()

        # tensorboard setting
        train_cer, train_n_label, train_l_dist, train_ctc_loss = loss_metric.get_name_value()
        summary_writer.add_scalar('loss epoch', train_ctc_loss, n_epoch)
        summary_writer.add_scalar('CER train', train_cer, n_epoch)

        # save checkpoints
        if n_epoch % save_checkpoint_every_n_epoch == 0:
            log.info('Epoch[%d] SAVE CHECKPOINT', n_epoch)
            module.save_checkpoint(prefix=get_checkpoint_path(args), epoch=n_epoch, save_optimizer_states=save_optimizer_states)

        n_epoch += 1

        lr_scheduler.learning_rate=learning_rate/learning_rate_annealing

    log.info('FINISH')
