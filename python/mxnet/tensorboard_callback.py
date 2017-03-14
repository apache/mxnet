# coding: utf-8
"""TensorBoard Callback functions that can be used to log various status during epoch."""
from __future__ import absolute_import

import logging
import time


class Speedometer(object):
    """Calculate, log training speed and evaluation metrics periodically in TensorBoard.
    This callback works almost same as `callback.Speedometer`, but support TensorBoard logging,
    you can choose either. For more usage, please refer https://github.com/dmlc/tensorboard

    Parameters
    ----------
    batch_size: int
        batch_size of data
    logging_dir: str
        TensorBoard logging file directory.
    frequent: int
        How many batches between calculations.
        Defaults to calculating & logging every 50 batches.

    Examples
    --------
    >>> logging_dir = 'log/'
    >>> batch_end_callbacks = [mx.tensorboard_callback.Speedometer(batch_size, logging_dir, freq)]
    """
    def __init__(self, batch_size, logging_dir, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        try:
            from tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

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
                    param.eval_metric.reset()
                    for name, value in name_value:
                        logging.info('Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-%s=%f',
                                     param.epoch, count, speed, name, value)
                        self.summary_writer.add_scalar('Training-Speed', speed)
                        self.summary_writer.add_scalar('Training-%s' % name, value)
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                    self.summary_writer.add_scalar('Training-Speed', speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()

