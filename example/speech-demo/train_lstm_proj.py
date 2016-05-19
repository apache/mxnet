import re
import sys
sys.path.insert(0, "../../python")
import time
import logging
import os.path

import mxnet as mx
import numpy as np
from speechSGD import speechSGD
from lstm_proj import lstm_unroll
from io_util import BucketSentenceIter, TruncatedSentenceIter, DataReadStream
from config_util import parse_args, get_checkpoint_path, parse_contexts


# some constants
METHOD_BUCKETING = 'bucketing'
METHOD_TBPTT = 'truncated-bptt'

def prepare_data(args):
    batch_size = args.config.getint('train', 'batch_size')
    num_hidden = args.config.getint('arch', 'num_hidden')
    num_hidden_proj = args.config.getint('arch', 'num_hidden_proj')
    num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    if num_hidden_proj > 0:
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden_proj)) for l in range(num_lstm_layer)]
    else:
        init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]

    init_states = init_c + init_h

    file_train = args.config.get('data', 'train')
    file_dev = args.config.get('data', 'dev')
    file_format = args.config.get('data', 'format')
    feat_dim = args.config.getint('data', 'xdim')

    train_data_args = {
            "gpu_chunk": 32768,
            "lst_file": file_train,
            "file_format": file_format,
            "separate_lines": True
            }

    dev_data_args = {
            "gpu_chunk": 32768,
            "lst_file": file_dev,
            "file_format": file_format,
            "separate_lines": True
            }

    train_sets = DataReadStream(train_data_args, feat_dim)
    dev_sets = DataReadStream(dev_data_args, feat_dim)

    return (init_states, train_sets, dev_sets)

def CrossEntropy(labels, preds):
    labels = labels.reshape((-1,))
    preds = preds.reshape((-1, preds.shape[2]))
    loss = 0.
    num_inst = 0
    for i in range(preds.shape[0]):
        label = labels[i]

        if label > 0:
            loss += -np.log(max(1e-10, preds[i][int(label)]))
            num_inst += 1
    return loss , num_inst

def Acc_exclude_padding(labels, preds):
    labels = labels.reshape((-1,))
    preds = preds.reshape((-1, preds.shape[2]))
    sum_metric = 0
    num_inst = 0
    for i in range(preds.shape[0]):
        pred_label = np.argmax(preds[i], axis=0)
        label = labels[i]

        ind = np.nonzero(label.flat)
        pred_label_real = pred_label.flat[ind]
        label_real = label.flat[ind]
        sum_metric += (pred_label_real == label_real).sum()
        num_inst += len(pred_label_real)
    return sum_metric, num_inst

class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
    """A simple lr schedule that simply return `dynamic_lr`. We will set `dynamic_lr`
    dynamically based on performance on the validation set.
    """
    def __init__(self, dynamic_lr, effective_sample_count=1, momentum=0.9, optimizer="sgd"):
        super(SimpleLRScheduler, self).__init__()
        self.dynamic_lr = dynamic_lr
        self.effective_sample_count = effective_sample_count
        self.momentum = momentum
        self.optimizer = optimizer

    def __call__(self, num_update):
        if self.optimizer == "speechSGD":
            return self.dynamic_lr / self.effective_sample_count, self.momentum
        else:
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
            outputs[i].copyto(eval_data.init_state_arrays[i-1])


def get_initializer(args):
    init_type = getattr(mx.initializer, args.config.get('train', 'initializer'))
    init_scale = args.config.getfloat('train', 'init_scale')
    if init_type is mx.initializer.Xavier:
        return mx.initializer.Xavier(magnitude=init_scale)
    return init_type(init_scale)


def do_training(training_method, args, module, data_train, data_val):
    from distutils.dir_util import mkpath
    mkpath(os.path.dirname(get_checkpoint_path(args)))

    batch_size = data_train.batch_size
    batch_end_callbacks = [mx.callback.Speedometer(batch_size, 
                                                   args.config.getint('train', 'show_every'))]
    eval_allow_extra = True if training_method == METHOD_TBPTT else False
    eval_metric = [mx.metric.np(CrossEntropy, allow_extra_outputs=eval_allow_extra),
                   mx.metric.np(Acc_exclude_padding, allow_extra_outputs=eval_allow_extra)]
    eval_metric = mx.metric.create(eval_metric)
    optimizer = args.config.get('train', 'optimizer')
    momentum = args.config.getfloat('train', 'momentum')
    learning_rate = args.config.getfloat('train', 'learning_rate')
    lr_scheduler = SimpleLRScheduler(learning_rate, momentum=momentum, optimizer=optimizer)

    if training_method == METHOD_TBPTT:
        lr_scheduler.seq_len = data_train.truncate_len

    n_epoch = 0
    num_epoch = args.config.getint('train', 'num_epoch')
    learning_rate = args.config.getfloat('train', 'learning_rate')
    decay_factor = args.config.getfloat('train', 'decay_factor')
    decay_bound = args.config.getfloat('train', 'decay_lower_bound')
    clip_gradient = args.config.getfloat('train', 'clip_gradient')
    weight_decay = args.config.getfloat('train', 'weight_decay')
    if clip_gradient == 0:
        clip_gradient = None

    last_acc = -float("Inf")
    last_params = None

    module.bind(data_shapes=data_train.provide_data,
                label_shapes=data_train.provide_label,
                for_training=True)
    module.init_params(initializer=get_initializer(args))

    def reset_optimizer():
        if optimizer == "sgd" or optimizer == "speechSGD":
            module.init_optimizer(kvstore='local',
                              optimizer=args.config.get('train', 'optimizer'),
                              optimizer_params={'lr_scheduler': lr_scheduler,
                                                'momentum': momentum,
                                                'rescale_grad': 1.0,
                                                'clip_gradient': clip_gradient,
                                                'wd': weight_decay},
                              force_init=True)
        else:
            module.init_optimizer(kvstore='local',
                              optimizer=args.config.get('train', 'optimizer'),
                              optimizer_params={'lr_scheduler': lr_scheduler,
                                                'rescale_grad': 1.0,
                                                'clip_gradient': clip_gradient,
                                                'wd': weight_decay},
                              force_init=True)
    reset_optimizer()

    while True:
        tic = time.time()
        eval_metric.reset()

        for nbatch, data_batch in enumerate(data_train):
            if training_method == METHOD_TBPTT:
                lr_scheduler.effective_sample_count = data_train.batch_size * truncate_len
                lr_scheduler.momentum = np.power(np.power(momentum, 1.0/(data_train.batch_size * truncate_len)), data_batch.effective_sample_count)
            else:
                if data_batch.effective_sample_count is not None:
                    lr_scheduler.effective_sample_count = data_batch.effective_sample_count

            module.forward_backward(data_batch)
            module.update()
            module.update_metric(eval_metric, data_batch.label)

            batch_end_params = mx.model.BatchEndParam(epoch=n_epoch, nbatch=nbatch,
                                                      eval_metric=eval_metric,
                                                      locals=None)
            for callback in batch_end_callbacks:
                callback(batch_end_params)

            if training_method == METHOD_TBPTT:
                # copy over states
                outputs = module.get_outputs()
                # outputs[0] is softmax, 1:end are states
                for i in range(1, len(outputs)):
                    outputs[i].copyto(data_train.init_state_arrays[i-1])

        for name, val in eval_metric.get_name_value():
            logging.info('Epoch[%d] Train-%s=%f', n_epoch, name, val)
        toc = time.time()
        logging.info('Epoch[%d] Time cost=%.3f', n_epoch, toc-tic)

        data_train.reset()

        # test on eval data
        score_with_state_forwarding(module, data_val, eval_metric)

        # test whether we should decay learning rate
        curr_acc = None
        for name, val in eval_metric.get_name_value():
            logging.info("Epoch[%d] Dev-%s=%f", n_epoch, name, val)
            if name == 'CrossEntropy':
                curr_acc = val
        assert curr_acc is not None, 'cannot find Acc_exclude_padding in eval metric'

        if n_epoch > 0 and lr_scheduler.dynamic_lr > decay_bound and curr_acc > last_acc:
            logging.info('Epoch[%d] !!! Dev set performance drops, reverting this epoch',
                         n_epoch)
            logging.info('Epoch[%d] !!! LR decay: %g => %g', n_epoch,
                         lr_scheduler.dynamic_lr, lr_scheduler.dynamic_lr / float(decay_factor))

            lr_scheduler.dynamic_lr /= decay_factor
            # we reset the optimizer because the internal states (e.g. momentum)
            # might already be exploded, so we want to start from fresh
            reset_optimizer()
            module.set_params(*last_params)
        else:
            last_params = module.get_params()
            last_acc = curr_acc
            n_epoch += 1

            # save checkpoints
            mx.model.save_checkpoint(get_checkpoint_path(args), n_epoch,
                                     module.symbol, *last_params)

        if n_epoch == num_epoch:
            break

if __name__ == '__main__':
    args = parse_args()
    args.config.write(sys.stdout)

    training_method = args.config.get('train', 'method')
    contexts = parse_contexts(args)

    init_states, train_sets, dev_sets = prepare_data(args)
    state_names = [x[0] for x in init_states]

    batch_size = args.config.getint('train', 'batch_size')
    num_hidden = args.config.getint('arch', 'num_hidden')
    num_hidden_proj = args.config.getint('arch', 'num_hidden_proj')
    num_lstm_layer = args.config.getint('arch', 'num_lstm_layer')
    feat_dim = args.config.getint('data', 'xdim')
    label_dim = args.config.getint('data', 'ydim')

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

    if training_method == METHOD_BUCKETING:
        buckets = args.config.get('train', 'buckets')
        buckets = list(map(int, re.split(r'\W+', buckets)))
        data_train = BucketSentenceIter(train_sets, buckets, batch_size, init_states, feat_dim=feat_dim)
        data_val   = BucketSentenceIter(dev_sets, buckets, batch_size, init_states, feat_dim=feat_dim)

        def sym_gen(seq_len):
            sym = lstm_unroll(num_lstm_layer, seq_len, feat_dim, num_hidden=num_hidden,
                              num_label=label_dim, num_hidden_proj=num_hidden_proj)
            data_names = ['data'] + state_names
            label_names = ['softmax_label']
            return (sym, data_names, label_names)

        module = mx.mod.BucketingModule(sym_gen,
                                        default_bucket_key=data_train.default_bucket_key,
                                        context=contexts)
        do_training(training_method, args, module, data_train, data_val)
    elif training_method == METHOD_TBPTT:
        truncate_len = args.config.getint('train', 'truncate_len')
        data_train = TruncatedSentenceIter(train_sets, batch_size, init_states,
                                           truncate_len=truncate_len, feat_dim=feat_dim)
        data_val = TruncatedSentenceIter(dev_sets, batch_size, init_states,
                                         truncate_len=truncate_len, feat_dim=feat_dim,
                                         do_shuffling=False, pad_zeros=True)
        sym = lstm_unroll(num_lstm_layer, truncate_len, feat_dim, num_hidden=num_hidden,
                          num_label=label_dim, output_states=True, num_hidden_proj=num_hidden_proj)
        data_names = [x[0] for x in data_train.provide_data]
        label_names = [x[0] for x in data_train.provide_label]
        module = mx.mod.Module(sym, context=contexts, data_names=data_names,
                               label_names=label_names)
        do_training(training_method, args, module, data_train, data_val)
    else:
        raise RuntimeError('Unknown training method: %s' % training_method)

    print("="*80)
    print("Finished Training")
    print("="*80)
    args.config.write(sys.stdout)
