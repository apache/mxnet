# coding: utf-8
"""Callback functions that can be used to track various status during epoch."""
from __future__ import absolute_import

import sys
import math
import logging
import time
from .model import save_checkpoint

def do_checkpoint(prefix):
    """Callback to checkpoint the model to prefix every epoch.

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


def log_train_metric(period, auto_reset=False):
    """Callback to log the training evaluation result every period.

    Parameters
    ----------
    period : int
        The number of batch to log the training evaluation metric.
    auto_reset : bool
        Reset the metric after each log

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_epoch_callback to fit.
    """
    def _callback(param):
        """The checkpoint function."""
        if param.nbatch % period == 0:
            if not isinstance(param.eval_metric, list):
                name, value = param.eval_metric.get()
                logging.info('Iter[%d] Batch[%d] Train-%s=%f',
                             param.epoch, param.nbatch, name, value)
            else:
                logging_string = 'Iter[%d] Batch[%d] ' % (param.epoch, param.nbatch)
                for i in range(len(param.eval_metric)):
                    name, value = param.eval_metric[i].get()
                    logging_string += 'Train-%s=%f ' % (name, value)
                logging.info(logging_string)
            if auto_reset:
                if not isinstance(param.eval_metric, list):
                    param.eval_metric.reset()
                else:
                    for i in range(len(param.eval_metric)):
                        param.eval_metric[i].reset()
    return _callback


class Speedometer(object):
    """Calculate training speed in frequent

    Parameters
    ----------
    batch_size: int
        batch_size of data
    frequent: int
        calculation frequent
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
                if param.eval_metric is not None:
                    if not isinstance(param.eval_metric, list):
                        name, value = param.eval_metric.get()
                        logging.info("Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-%s=%f",
                                     param.epoch, count, speed, name, value)
                    else:
                        logging_string = 'Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec\t' \
                                         % (param.epoch, count, speed)
                        for i in range(len(param.eval_metric)):
                            name, value = param.eval_metric[i].get()
                            logging_string += 'Train-%s=%f ' % (name, value)
                        logging.info(logging_string)
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
