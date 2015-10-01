# coding: utf-8
"""Callback functions that can be used to track various status during iteration."""
from __future__ import absolute_import

import sys
import math
import logging
import time
from .model import save_checkpoint

def do_checkpoint(prefix):
    """Callback to checkpoint the model to prefix every iteration.

    Parameters
    ----------
    prefix : str
        The file prefix to checkpoint to

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """
    def _callback(iter_no, sym, arg, aux):
        """The checkpoint function."""
        save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
    return _callback


def log_train_metric(period):
    """Callback to log the training evaluation result every period.

    Parameters
    ----------
    period : int
        The number of batch to log the training evaluation metric.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_epoch_callback to fit.
    """
    def _callback(param):
        """The checkpoint function."""
        if param.nbatch % period == 0:
            name, value = param.eval_metric.get()
            logging.info('Iter[%d] Batch[%d] Train-%s=%f',
                         param.iteration, param.nbatch, name, value)
    return _callback


class Speedometer(object):
    """Calculate training speed in frequent

    Parameters
    ----------
    batch_size: int
        batch_size of data
    frequent: int
        calcutaion frequent
    """
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                             param.iteration, count, speed)
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
