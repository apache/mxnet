# coding: utf-8
"""Callback functions that can be used to track various status during epoch."""
from __future__ import absolute_import

import logging
import math
import sys
import time
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
                    for name, value in name_value:
                        logging.info('Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-%s=%f',
                                     param.epoch, count, speed, name, value)
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


class ProgressBar(object):
    """Show a progress bar.

    Parameters
    ----------
    total: int
        total batch size
    length: int
        length or progress bar
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
        sys.stdout.write('[%s] %s%s\r' % (prog_bar, percents, '%'))


class LogValidationMetricsCallback(object):
    """Just logs the eval metrics at the end of an epoch."""

    def __call__(self, param):
        if not param.eval_metric:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            logging.info('Epoch[%d] Validation-%s=%f', param.epoch, name, value)
