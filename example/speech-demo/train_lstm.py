import re
import sys
import time
import logging
import os.path
import argparse
import configparser

import mxnet as mx

from io_util import BucketSentenceIter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", help="config file for training parameters")
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.configfile)
    args.config = cfg
    return args

def prepare_data(args):
    batch_size = int(args.config.get('train', 'batch_size'))
    num_hidden = int(args.config.get('arch', 'num_hidden'))
    num_lstm_layer = int(args.config.get('arch', 'num_lstm_layer'))

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]

    init_states = init_c + init_h

    file_train = args.config.get('data', 'train')
    file_dev = args.config.get('data', 'dev')
    file_format = args.config.get('data', 'format')
    feat_dim = int(args.config.get('data', 'xdim'))

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

def get_checkpoint_path(args):
    prefix = args.config.get('train', 'prefix')
    if os.path.isabs(prefix):
        return prefix
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', prefix))

def Acc_exlude_padding(labels, preds):
    labels = labels.T.reshape((-1,))
    sum_metric = 0
    num_inst = 0
    for i in range(preds.shape[0]):
        pred_label = np.argmax(preds[i], axis=0)
        label = labels[i]

        ind = np.nonzero(label.flat)
        pred_label_real = pred_label.flat[ind]
        #print label, pred_label, ind
        label_real = label.flat[ind]
        sum_metric += (pred_label_real == label_real).sum()
        num_inst += len(pred_label_real)
    return sum_metric, num_inst

class SimpleLRScheduler(mx.lr_scheduler.LRScheduler):
    """A simple lr schedule that simply return `base_lr`. We will set `base_lr`
    dynamically based on performance on the validation set.
    """
    def __init__(self, base_lr):
        self.base_lr = base_lr

    def __call__(self, num_update):
        return self.base_lr

def do_training(args, module, data_train, data_val):
    momentum = float(args.config.get('train', 'momentum', fallback=0.0))
    learning_rate = float(args.config.get('train', 'learning_rate'))
    lr_scheduler = SimpleLRScheduler(learning_rate)
    eval_metric  = mx.metric.np(Acc_exlude_padding)

    batch_size = data_train.batch_size
    batch_end_callbacks = [mx.callback.Speedometer(batch_size, 100)]

    n_epoch = 0
    num_epoch = int(args.config.get('train', 'num_epoch'))
    learning_rate = float(args.config.get('train', 'learning_rate'))
    decay_factor = float(args.config.get('train', 'decay_factor'))
    decay_bound = float(args.config.get('train', 'decay_lower_bound'))

    last_acc = -float("Inf")
    last_params = None

    module.bind(data_shapes=data_train.provide_data,
                label_shapes=data_train.provide_label,
                for_training=True)
    module.init_params(initializer=mx.initializer.Uniform(0.01))
    module.init_optimizer(kvstore='local',
                          optimizer=args.config.get('train', 'optimizer'),
                          optimizer_params={'lr_scheduler': lr_scheduler,
                                            'momentum': momentum})

    while True:
        tic = time.time()
        eval_metric.reset()

        for nbatch, data_batch in enumerate(train_data):
            module.forward_backward(data_batch)
            module.update()
            module.update_metric(eval_metric, data_batch.label)

            batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                             eval_metric=eval_metric,
                                             locals=locals())
            for callback in batch_end_callbacks:
                callback(batch_end_params)

        for name, val in eval_metric.get_name_value():
            self.logger.info('Epoch[%d] Train-%s=%f', n_epoch, name, val)
        toc = time.time()
        logging.info('Epoch[%d] Time cost=%.3f', n_epoch, toc-tic)

        train_data.reset()

        # test on eval data
        module.score(data_val, eval_metric, epoch=n_epoch)

        # test whether we should decay learning rate
        curr_acc = None
        for name, val in eval_metric.get_name_value():
            if name == 'Acc_exclude_padding':
                curr_acc = val
                break
        assert curr_acc is not None, 'cannot find Acc_exclude_padding in eval metric'

        if n_epoch > 0 and lr_scheduler.base_lr > decay_bound and curr_acc < last_acc:
            logging.info('Epoch[%d] !!! Dev set performance drops, reverting this epoch',
                         n_epoch)
            logging.info('Epoch[%d] !!! LR decay: %g => %g',
                         lr_scheduler.base_lr, lr_scheduler.base_lr / float(decay_factor))

            lr_scheduler.base_lr /= decay_factor
            module.set_params(*last_params)
        else:
            last_params = model.get_params
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

    training_method = args.config.get('train', 'method', fallback='bucketing')

    # parse context into Context objects
    contexts = re.split(r'\W+', args.config.get('train', 'context', fallback='gpu0'))
    for i, ctx in enumerate(contexts):
        if ctx[:3] == 'gpu':
            contexts[i] = mx.context.gpu(int(ctx[3:]))
        else:
            contexts[i] = mx.context.cpu(int(ctx[3:]))

    init_states, train_sets, dev_sets = prepare_data(args)
    state_names = [x[0] for x in init_states]

    batch_size = int(args.config.get('train', 'batch_size'))
    num_hidden = int(args.config.get('arch', 'num_hidden'))
    num_lstm_layer = int(args.config.get('arch', 'num_lstm_layer'))
    feat_dim = int(args.config.get('data', 'xdim'))
    label_dim = int(args.config.get('data', 'ydim'))

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

    if training_method == 'bucketing':
        buckets = args.config.get('train', 'buckets')
        buckets = list(map(int, re.split(r'\W+', buckets)))
        data_train = BucketSentenceIter(train_sets, buckets, batch_size, init_states, feat_dim=feat_dim)
        data_val   = BucketSentenceIter(dev_sets, buckets, batch_size, init_states, feat_dim=feat_dim)

        def sym_gen(seq_len):
            sym = lstm_unroll(num_lstm_layer, seq_len, feat_dim, num_hidden=num_hidden,
                              num_label=label_dim)
            data_names = ['data'] + state_names
            label_names = ['softmax_label']
            return (sym, data_names, label_names)

        module = mx.mod.BucketingModule(sym_gen,
                                        default_bucket_key=data_train.default_bucket_key,
                                        context=contexts)
        do_training(args, module, data_train, data_val)
    else:
        raise RuntimeError('Unknown training method: %s' % training_method)

