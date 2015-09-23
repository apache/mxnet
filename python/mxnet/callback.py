# pylint: disable=logging-not-lazy, blacklisted-name, invalid-name
"""model helper for knowing training status"""
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
    def _callback(iter_no, s, arg, aux):
        """The checkpoint function."""
        save_checkpoint(prefix, iter_no + 1, s, arg, aux)
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

    def __call__(self, count):
        """
        Show speed

        Parameters
        ----------
        count: int
            current batch count
        """

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                logging.info("Batch [%d]\tSpeed: %.2f samples/sec" % (count, speed))
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()

class ProgressBar(object):
    """Show a progress bar

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

    def __call__(self, count):
        """
        Update progress bar

        Parameters
        ----------
        count: int
            current batch count
        """

        filled_len = int(round(self.bar_len * count / float(self.total)))
        percents = math.ceil(100.0 * count / float(self.total))
        bar = '=' * filled_len + '-' * (self.bar_len - filled_len)
        sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))


