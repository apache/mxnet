# coding: utf-8
"""Callback functions that can be used to track various status during epoch."""
from __future__ import absolute_import

import logging
import math
import time
import warnings
import numpy as np

from . import metric

from .model import save_checkpoint

def module_checkpoint(mod, prefix, period=1, save_optimizer_states=False):
    """Callback to checkpoint Module to prefix every epoch.

    Parameters
    ----------
    mod : subclass of BaseModule
        The module to checkpoint.
    prefix : str
        The file prefix for this checkpoint.
    period : int
        How many epochs to wait before checkpointing. Defaults to 1.
    save_optimizer_states : bool
        Indicates whether or not to save optimizer states for continued training.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """
    period = int(max(1, period))
    # pylint: disable=unused-argument
    def _callback(iter_no, sym=None, arg=None, aux=None):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            mod.save_checkpoint(prefix, iter_no + 1, save_optimizer_states)
    return _callback


def do_checkpoint(prefix, period=1):
    """A callback that saves a model checkpoint every few epochs.
    Each checkpoint is made up of a couple of binary files: a model description file and a
    parameters (weights and biases) file. The model description file is named
    `prefix`--symbol.json and the parameters file is named `prefix`-`epoch_number`.params

    Parameters
    ----------
    prefix : str
        Prefix for the checkpoint filenames.
    period : int, optional
        Interval (number of epochs) between checkpoints. Default `period` is 1.

    Returns
    -------
    callback : function
        A callback function that can be passed as `epoch_end_callback` to fit.

    Example
    -------
    >>> module.fit(iterator, num_epoch=n_epoch,
    ... epoch_end_callback  = mx.callback.do_checkpoint("mymodel", 1))
    Start training with [cpu(0)]
    Epoch[0] Resetting Data Iterator
    Epoch[0] Time cost=0.100
    Saved checkpoint to "mymodel-0001.params"
    Epoch[1] Resetting Data Iterator
    Epoch[1] Time cost=0.060
    Saved checkpoint to "mymodel-0002.params"
    """
    period = int(max(1, period))
    def _callback(iter_no, sym, arg, aux):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
    return _callback


def log_train_metric(period, auto_reset=False):
    """Callback to log the training evaluation result every period.

    Parameters
    ----------
    period : int
        The number of batch to log the training evaluation metric.
    auto_reset : bool
        Reset the metric after each log.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_epoch_callback to fit.
    """
    def _callback(param):
        """The checkpoint function."""
        if param.nbatch % period == 0 and param.eval_metric is not None:
            name_value = param.eval_metric.get_name_value()
            for name, value in name_value:
                logging.info('Iter[%d] Batch[%d] Train-%s=%f',
                             param.epoch, param.nbatch, name, value)
            if auto_reset:
                param.eval_metric.reset()
    return _callback


class Speedometer(object):
    """Logs training speed and evaluation metrics periodically.

    Parameters
    ----------
    batch_size: int
        Batch size of data.
    frequent: int
        Specifies how frequently training speed and evaluation metrics
        must be logged. Default behavior is to log once every 50 batches.
    auto_reset : bool
        Reset the evaluation metrics after each log.

    Example
    -------
    >>> # Print training speed and evaluation metrics every ten batches. Batch size is one.
    >>> module.fit(iterator, num_epoch=n_epoch,
    ... batch_end_callback=mx.callback.Speedometer(1, 10))
    Epoch[0] Batch [10] Speed: 1910.41 samples/sec  Train-accuracy=0.200000
    Epoch[0] Batch [20] Speed: 1764.83 samples/sec  Train-accuracy=0.400000
    Epoch[0] Batch [30] Speed: 1740.59 samples/sec  Train-accuracy=0.500000
    """
    def __init__(self, batch_size, frequent=50, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                    msg = 'Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec'
                    msg += '\t%s=%f'*len(name_value)
                    logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


class EarlyStopping(object):
    """Apply the early stopping strategy to avoid over-fitting. During the training process,
    it uses the metrics of performance on validation if the eval data is available, otherwise,
    the metrics for training data will be used.

    Parameters
    ----------
    min_delta: int
        Defaults to 0. Specifies the minimum delta to be accepted between two epochs,
        any improvement that is small than this number will be considered as non-improvement.
        If 0, any improvement will count.
    patience : int
        Defaults to 1. Specifies the number of epochs to be tolerated without any improvement.
        Once this patience was exceeded, the training process will be stopped.
    verbose: int
        Defaults to 0. Higher verbose will give more information
    mode: str
        Defaults to 'auto'. Specifies if the performance is an accuray (higher values are good) or
        a loss (lower values are good). 'auto' mode automatically detects built in metrics.
        Other possible options are:
        'min', 'max'
    save_model : bool
        Defaults to False. Specifies whether to save the model when early stopping happens.
    model_name: str
        Defaults to 'dummy_model'. Specifies the model name to be saved, if not given, the name
        'dummy_model' will be used.

       Example
       -------
       >>> # Stop the training when there is no improvements in 5 rounds. Using
       >>> module.fit(iterator, num_epoch=n_epoch, eval_metric='acc',
       ... epoch_end_callback=EarlyStopping(eval_metric='acc', patience=5, verbose=1))
        INFO:root:Epoch[21] Train-accuracy=0.640086
        INFO:root:Epoch[21] Time cost=0.531
        INFO:root:Epoch[22] Train-accuracy=0.640086
        INFO:root:Epoch[22] Time cost=0.533
        INFO:root:Epoch[23] Train-accuracy=0.640086
        INFO:root:Epoch[23] Time cost=0.531
        INFO:root:Epoch[24] Train-accuracy=0.640086
        INFO:root:Epoch[24] Time cost=0.535
        INFO:root:Epoch[25] Train-accuracy=0.640086
        INFO:root:Epoch[25] Time cost=0.534
        Epoch 00025: early stopping
        INFO:root:Saved checkpoint to "dummy_model-0025.params"
    """

    def __init__(self, min_delta=0, patience=1, verbose=0, mode='auto', save_model=False,
                 model_name='dummy_model'):
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.save_model = save_model
        self.model_name = model_name

        self.wait = 0
        self.stopped_epoch = 0
        self.continue_training = True

        self.eval_metric = None
        self.best = None
        self.metric_op = None

    def _set_eval(self, eval_metric):
        """Initialise the evaluation metrics and the best performance"""
        if isinstance(eval_metric, metric.EvalMetric):
            eval_metric = eval_metric.name
        self.eval_metric = eval_metric

        if self.mode == 'min':
            self.metric_op = np.less
        elif self.mode == 'max':
            self.metric_op = np.greater
        else:
            if ('acc' in self.eval_metric or
                    'f1' in self.eval_metric or
                    'top_k_accuracy' in self.eval_metric):
                self.metric_op = np.greater
            else:
                self.metric_op = np.less

        if self.metric_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        self.best = np.Inf if self.metric_op == np.less else -np.Inf

    def __call__(self, epoch, symbol, arg_params, aux_params, eval_metric,
                 epoch_train_eval_metrics):
        if self.eval_metric is None:
            self._set_eval(eval_metric)
        if epoch_train_eval_metrics is None:
            warnings.warn('Early stopping requires metric available!', RuntimeWarning)
        current = epoch_train_eval_metrics
        if self.metric_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.continue_training = False
                if self.stopped_epoch > 0 and self.verbose > 0:
                    print('Epoch %05d: early stopping' % self.stopped_epoch)
                if self.save_model:
                    save_checkpoint(self.model_name, self.stopped_epoch, symbol, arg_params,
                                    aux_params)
            self.wait += 1
        return self.continue_training


class ProgressBar(object):
    """Displays a progress bar, indicating the percentage of batches processed within each epoch.

    Parameters
    ----------
    total: int
        total number of batches per epoch
    length: int
        number of chars to define maximum length of progress bar

    Examples
    --------
    >>> progress_bar = mx.callback.ProgressBar(total=2)
    >>> mod.fit(data, num_epoch=5, batch_end_callback=progress_bar)
    [========--------] 50.0%
    [================] 100.0%
    """
    def __init__(self, total, length=80):
        self.bar_len = length
        self.total = total

    def __call__(self, param):
        """Callback to Show progress bar."""
        count = param.nbatch
        filled_len = int(round(self.bar_len * count / float(self.total)))
        percents = math.ceil(100.0 * count / float(self.total))
        prog_bar = '=' * filled_len + '-' * (self.bar_len - filled_len)
        logging.info('[%s] %s%s\r', prog_bar, percents, '%')


class LogValidationMetricsCallback(object):
    """Just logs the eval metrics at the end of an epoch."""

    def __call__(self, param):
        if not param.eval_metric:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            logging.info('Epoch[%d] Validation-%s=%f', param.epoch, name, value)
