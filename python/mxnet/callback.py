# coding: utf-8
"""Callback functions that can be used to track various status during epoch."""
from __future__ import absolute_import

import logging
import math
import sys
import time
from collections import defaultdict
from twisted.internet.protocol import DatagramProtocol
from twisted.internet import reactor
from CommandParsing import *
import socket
import Queue
from .model import save_checkpoint

def do_checkpoint(prefix, period=1):
    """Callback to checkpoint the model to prefix every epoch.

    Parameters
    ----------
    prefix : str
        The file prefix to checkpoint to
    period : int
    	How many epochs to wait before checkpointing. Default is 1.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
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
        Reset the metric after each log

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
    """Calculate and log training speed periodically.

    Parameters
    ----------
    batch_size: int
        batch_size of data
    frequent: int
        How many batches between calculations.
        Defaults to calculating & logging every 50 batches.
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
                    name_value = param.eval_metric.get_name_value()
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

class ConnectedSpeedometer(DatagramProtocol):
    #myRole: scheduler aggregate and log, workers push only. Servers ignore
    def __init__(self, batch_size,frequent=50,myRole, numWorker, (schedIP,schedPort)):
        self.Role = myRole
        self.MaxBatch = 0
        self.History = 2
        self.MaxEpoch = 0
        self.Records = defaultdict(defaultdict(list))
        self.NumWorker = numWorker
        assert isinstance(self.Role,int) and self.Role >= 0 and self.Role <= 2;
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.SendTo = (schedIP,schedPort)

        
    def datagramReceived(self, data, (host,port)):
        if self.Role <= 1:
            return
        #is the format of %d:epoch,%d:batch,%d:speed
        strcmd = str(data)
        segments = strcmd.split('|')
        head = segments[0]
        epoch = int(head[0])
        batch = int(head[1])
        kvs = segments[1]
        keyPair = (epoch,batch)
        for kv in kvs.split(','):
            kvSeg = kv.split('=')
            key = kvSeg[0]
            val = float(kvSeg[1])
            self.Records[keyPair][key].append(val)
        self.Records[keyPair]["Counter"].append(0)

        if len(self.Records[keyPair]["Counter"]) == self.NumWorker:
            #I can produce a broadcast.
            #foreach key, produce an average
            output = 'Epoch=%d;Batch=%d;'
            for key in self.Records[keyPair]:
                output = output + (';%s=%f' % (key, sum(self.Records[keyPair][key])/float(len(self.Records[keyPair][key]))))
            logging.info(output)
        #else do nothing.
        #purge older
        if len(self.Records) > self.History:
            keys = self.Records.keys()
            keys.sort()
            for key in keys[-1 * self.History:]:
                del self.Records[key]
        #done!
        
    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                str = '%d,%d|speed=%.2f'  % (param.epoch, count, speed)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    param.eval_metric.reset()
                    for name, value in name_value:
                        #logging.info('Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tT\rain-%s=%f',param.epoch, count, speed, name, value)
                        str = str + (',%s=%f'  % (name, value))
                self.transport.write(str,self.SendTo)
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
    """Just logs the eval metrics at the end of an epoch.
    """

    def __call__(self, param):
        if not param.eval_metric:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            logging.info('Epoch[%d] Validation-%s=%f', param.epoch, name, value)
