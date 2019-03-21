# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import
"""Gluon EventHandlers for Estimators"""

__all__ = ['EventHandler', 'LoggingHandler']
import logging
import os
import time
import warnings

import numpy as np


class EventHandler(object):
    """Basic for event handlers

        :py:class:`EventHandler` can perform user defined functions at
        different stages of training: train begin, epoch begin, batch begin,
        batch end, epoch end, train end.

        Parameters
        ----------
        estimator : Estimator
            The :py:class:`Estimator` to get training statistics
        """
    def __init__(self):
        self._history = None
        self._net = None

    @property
    def train_history(self):
        return self._history

    @train_history.setter
    def train_history(self, history):
        self._history = history

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, net):
        self._net = net

    def train_begin(self):
        pass

    def train_end(self):
        pass

    def batch_begin(self):
        pass

    def batch_end(self):
        pass

    def epoch_begin(self):
        pass

    def epoch_end(self):
        pass


class LoggingHandler(EventHandler):
    """Basic Logging Handler that applies to every Gluon estimator by default.

    :py:class:`LoggingHandler` logs hyper-parameters, training statistics,
    and other useful information during training

    Parameters
    ----------
    estimator : Estimator
        The :py:class:`Estimator` to get training statistics
    file_name : str
        file name to save the logs
    file_location: str
        file location to save the logs
    """

    def __init__(self, file_name=None, file_location=None, ):
        super(LoggingHandler, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)
        # save logger to file only if file name or location is specified
        if file_name or file_location:
            file_name = file_name or 'estimator_log'
            file_location = file_location or './'
            file_handler = logging.FileHandler(os.path.join(file_location, file_name))
            self.logger.addHandler(file_handler)

    def train_begin(self):
        self.train_start = time.time()
        self.logger.info("Training begin: using optimizer %s "
                         "with current learning rate %.4f "
                         % (self.train_history.optimizer.__class__.__name__,
                            self.train_history.learning_rate))
        self.logger.info("Train for %d epochs." % self.train_history.max_epoch)

    def train_end(self):
        train_time = time.time() - self.train_start
        epoch = self.train_history.epoch
        train_stats = self.train_history.get_train_stats(epoch=epoch)
        val_stats = self.train_history.get_train_stats(epoch=epoch)
        msg = 'Train finished using total %ds at epoch %d' % (train_time, epoch)
        for key in train_stats:
            msg += 'train %s : %.4f ' % (key, train_stats[key])
        for key in val_stats:
            msg += 'val %s : %.4f ' % (key, val_stats[key])
        self.logger.info(msg)

    def batch_begin(self):
        self.batch_start = time.time()

    def batch_end(self):
        batch_time = time.time() - self.batch_start
        epoch = self.train_history.epoch
        batch = self.train_history.batch_idx
        msg = '[Epoch %d] [Batch %d] ' % (epoch, batch)
        if self.train_history.samples:
            msg += '[Samples %s] ' % (self.train_history.samples)
        msg += 'time/batch: %.3fs ' % batch_time
        batch_stats = self.train_history.get_batch_stats()
        for key in batch_stats:
            msg += key + ': ' + '%.4f ' % batch_stats[key]
        self.logger.info(msg)

    def epoch_begin(self):
        self.epoch_start = time.time()

    def epoch_end(self):
        epoch_time = time.time() - self.epoch_start
        epoch = self.train_history.epoch
        msg = '\n[Epoch %d] finished in %.3fs: ' % (epoch, epoch_time)
        train_stats = self.train_history.get_train_stats(epoch=epoch)
        val_stats = self.train_history.get_train_stats(epoch=epoch)
        for key in train_stats:
            msg += 'train %s : %.4f ' % (key, train_stats[key])
        for key in val_stats:
            msg += 'val %s : %.4f ' % (key, val_stats[key])
        self.logger.info(msg)


class CheckpointHandler(EventHandler):
    """Save the model after every epoch.

    :py:class:`CheckpointHandler` save the network parameters every epoch

    Parameters
    ----------
    estimator : Estimator
        The :py:class:`Estimator` to get training statistics
    filepath : str
        file name to save the parameters, it can contain directories,
        for example: ./saved_model/resnet.params
    monitor: str
        the metrics to monitor
    verbose: int, default 0
        verbosity mode
    save_best_only: bool
        if True, only save the parameters if monitored value improved
    mode: str, default 'auto'
        one of {auto, min, max}, if `save_best_only=True`, the comparison to make
        and determine if the monitored value has improved
    period: int, default 1
        intervals between saving the network
    """

    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 mode='auto',
                 period=1):
        super(CheckpointHandler, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.logger = logging.getLogger(__name__)

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            # use greater for accuracy and less otherwise
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def epoch_end(self, ):
        epoch = self.train_history.epoch
        # add extension for weights
        if '.params' not in self.filepath:
            self.filepath += '.params'
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                # check if monitor exists in train_stats
                if self.monitor not in self.train_history.get_train_stats().keys():
                    warnings.warn(RuntimeWarning('Unable to find %s in training statistics, make sure '
                                                 'you are passing one of the metric names as monitor', self.monitor))
                    self.net.save_parameters(self.filepath)
                else:
                    current = self.train_history.get_train_stats(self.monitor)
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            self.logger.info('\n[Epoch %d] %s improved from %0.5f to %0.5f,'
                                             ' saving model to %s',
                                             epoch, self.monitor, self.best, current, self.filepath)
                        self.best = current
                        self.net.save_parameters(self.filepath)
                    else:
                        if self.verbose > 0:
                            self.logger.info('\n[Epoch %d] %s did not improve from %0.5f, skipping save model',
                                             epoch, self.monitor, self.best)
            else:
                if self.verbose > 0:
                    logging.info('\nEpoch %d: saving model to %s', epoch, self.filepath)
                self.net.save_parameters(self.filepath)


class EarlyStoppingHandler(EventHandler):
    """Early stop training if monitored value is not improving

    Parameters
    ----------
    estimator : Estimator
        The :py:class:`Estimator` to get training statistics
    monitor: str
        the metrics to monitor
    min_delta: float, default 0
        minimal change in monitored value to be considered as an improvement
    patience: int, default 0
        number of epochs to wait for improvement before terminate training
    mode: str, default 'auto'
        one of {auto, min, max}, the comparison to make
        and determine if the monitored value has improved
    baseline: float
        baseline value to compare the monitored value with
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 mode='auto',
                 baseline=None):
        super(EarlyStoppingHandler, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.logger = logging.getLogger(__name__)

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(RuntimeWarning('EarlyStopping mode %s is unknown, '
                                         'fallback to auto mode.', mode))
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def epoch_end(self):
        epoch = self.train_history.epoch
        if self.monitor not in self.train_history.get_train_stats().keys():
            warnings.warn(RuntimeWarning('Unable to find %s in training statistics, make sure '
                                         'you are passing one of the metric names as monitor' % self.monitor))
        else:
            current = self.train_history.get_train_stats(self.monitor)
            if current is None:
                return

            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.train_history.stop_training(True)

    def train_end(self):
        if self.stopped_epoch > 0:
            self.logger.info('Epoch %d: early stopping due to %s not improving', self.stopped_epoch, self.monitor)
