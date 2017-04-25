import sys

sys.path.insert(0, "../../python")
import os.path
import mxnet as mx
from config_util import get_checkpoint_path, parse_contexts
from stt_metric import STTMetric


class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
    """A simple lr schedule that simply return `dynamic_lr`. We will set `dynamic_lr`
    dynamically based on performance on the validation set.
    """

    def __init__(self, dynamic_lr, effective_sample_count=1, momentum=0.9, optimizer="nag"):
        super(SimpleLRScheduler, self).__init__()
        self.dynamic_lr = dynamic_lr
        self.effective_sample_count = effective_sample_count
        self.momentum = momentum
        self.optimizer = optimizer

    def __call__(self, num_update):
        return self.dynamic_lr / self.effective_sample_count


def score_with_state_forwarding(module, eval_data, eval_metric):
    eval_data.reset()
    eval_metric.reset()

    for eval_batch in eval_data:
        module.forward(eval_batch, is_train=False)
        module.update_metric(eval_metric, eval_batch.label)
        # copy over states
        outputs = module.get_outputs()
        # outputs[0] is softmax, 1:end are states
        for i in range(1, len(outputs)):
            outputs[i].copyto(eval_data.init_state_arrays[i - 1])


def get_initializer(args):
    init_type = getattr(mx.initializer, args.config.get('train', 'initializer'))
    init_scale = args.config.getfloat('train', 'init_scale')
    if init_type is mx.initializer.Xavier:
        return mx.initializer.Xavier(magnitude=init_scale, factor_type=args.config.get('train', 'factor_type'))
    return init_type(init_scale)


def do_training(args, module, data_train, data_val, begin_epoch=0):
    from distutils.dir_util import mkpath
    from log_util import LogUtil

    log = LogUtil().getlogger()
    mkpath(os.path.dirname(get_checkpoint_path(args)))

    seq_len = args.config.get('arch', 'max_t_count')
    batch_size = args.config.getint('common', 'batch_size')
    save_checkpoint_every_n_epoch = args.config.getint('common', 'save_checkpoint_every_n_epoch')

    contexts = parse_contexts(args)
    num_gpu = len(contexts)
    eval_metric = STTMetric(batch_size=batch_size, num_gpu=num_gpu, seq_length=seq_len)

    optimizer = args.config.get('train', 'optimizer')
    momentum = args.config.getfloat('train', 'momentum')
    learning_rate = args.config.getfloat('train', 'learning_rate')
    lr_scheduler = SimpleLRScheduler(learning_rate, momentum=momentum, optimizer=optimizer)

    n_epoch = begin_epoch
    num_epoch = args.config.getint('train', 'num_epoch')
    clip_gradient = args.config.getfloat('train', 'clip_gradient')
    weight_decay = args.config.getfloat('train', 'weight_decay')
    save_optimizer_states = args.config.getboolean('train', 'save_optimizer_states')
    show_every = args.config.getint('train', 'show_every')

    if clip_gradient == 0:
        clip_gradient = None

    module.bind(data_shapes=data_train.provide_data,
                label_shapes=data_train.provide_label,
                for_training=True)

    if begin_epoch == 0:
        module.init_params(initializer=get_initializer(args))

    def reset_optimizer():
        module.init_optimizer(kvstore='device',
                              optimizer=args.config.get('train', 'optimizer'),
                              optimizer_params={'clip_gradient': clip_gradient,
                                                'wd': weight_decay},
                              force_init=True)

    reset_optimizer()

    while True:

        if n_epoch >= num_epoch:
            break

        eval_metric.reset()
        log.info('---------train---------')
        for nbatch, data_batch in enumerate(data_train):

            if data_batch.effective_sample_count is not None:
                lr_scheduler.effective_sample_count = data_batch.effective_sample_count

            module.forward_backward(data_batch)
            module.update()
            if (nbatch+1) % show_every == 0:
                module.update_metric(eval_metric, data_batch.label)
        # commented for Libri_sample data set to see only train cer
        log.info('---------validation---------')
        for nbatch, data_batch in enumerate(data_val):
            module.update_metric(eval_metric, data_batch.label)
        #module.score(eval_data=data_val, num_batch=None, eval_metric=eval_metric, reset=True)

        data_train.reset()
        # save checkpoints
        if n_epoch % save_checkpoint_every_n_epoch == 0:
            log.info('Epoch[%d] SAVE CHECKPOINT', n_epoch)
            module.save_checkpoint(prefix=get_checkpoint_path(args), epoch=n_epoch, save_optimizer_states=save_optimizer_states)

        n_epoch += 1

    log.info('FINISH')