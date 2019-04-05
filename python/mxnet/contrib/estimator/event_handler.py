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
        self._estimator = None

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        self._estimator = estimator

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
    verbose: int, default LOG_VERBOSITY_PER_EPOCH
        Limit the granularity of metrics displayed during training process
        verbose=LOG_VERBOSITY_PER_EPOCH: display metrics every epoch
        verbose=LOG_VERBOSITY_PER_BATCH: display metrics every batch
    """

    LOG_VERBOSITY_PER_EPOCH = 1
    LOG_VERBOSITY_PER_BATCH = 2

    def __init__(self, file_name=None, file_location=None, verbose=LOG_VERBOSITY_PER_EPOCH):
        super(LoggingHandler, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)
        if verbose not in [self.LOG_VERBOSITY_PER_EPOCH, self.LOG_VERBOSITY_PER_BATCH]:
            raise ValueError("verbose level must be either LOG_VERBOSITY_PER_EPOCH or "
                             "LOG_VERBOSITY_PER_BATCH, received %s. "
                             "E.g: LoggingHandler(verbose=LoggingHandler.LOG_VERBOSITY_PER_EPOCH)"
                             % verbose)
        self.verbose = verbose
        # save logger to file only if file name or location is specified
        if file_name or file_location:
            file_name = file_name or 'estimator_log'
            file_location = file_location or './'
            file_handler = logging.FileHandler(os.path.join(file_location, file_name))
            self.logger.addHandler(file_handler)

    def train_begin(self):
        self.train_start = time.time()
        self.logger.info("Training begin: using optimizer %s "
                         "with current learning rate %.4f ",
                         self.estimator.trainer.optimizer.__class__.__name__,
                         self.estimator.trainer.learning_rate)
        self.logger.info("Train for %d epochs.", self.estimator.max_epoch)

    def train_end(self):
        train_time = time.time() - self.train_start
        epoch = self.estimator.current_epoch
        msg = 'Train finished using total %ds at epoch %d. ' % (train_time, epoch)
        # log every result in train stats including train/validation loss & metrics
        for key in self.estimator.train_stats:
            msg += '%s : %.4f ' % (key, self.estimator.train_stats[key])
        self.logger.info(msg)

    def batch_begin(self):
        if self.verbose == self.LOG_VERBOSITY_PER_BATCH:
            self.batch_start = time.time()

    def batch_end(self):
        if self.verbose == self.LOG_VERBOSITY_PER_BATCH:
            batch_time = time.time() - self.batch_start
            epoch = self.estimator.current_epoch
            batch = self.estimator.batch_idx
            msg = '[Epoch %d] [Batch %d] ' % (epoch, batch)
            if self.estimator.processed_samples:
                msg += '[Samples %s] ' % (self.estimator.processed_samples)
            msg += 'time/batch: %.3fs ' % batch_time
            for key in self.estimator.train_stats:
                # only log current training loss & metric after each batch
                if key.startswith('train_'):
                    msg += key + ': ' + '%.4f ' % self.estimator.train_stats[key]
            self.logger.info(msg)

    def epoch_begin(self):
        if self.verbose >= self.LOG_VERBOSITY_PER_EPOCH:
            self.epoch_start = time.time()

    def epoch_end(self):
        if self.verbose >= self.LOG_VERBOSITY_PER_EPOCH:
            epoch_time = time.time() - self.epoch_start
            epoch = self.estimator.current_epoch
            msg = '\n[Epoch %d] finished in %.3fs: ' % (epoch, epoch_time)
            # log every result in train stats including train/validation loss & metrics
            for key in self.estimator.train_stats:
                msg += '%s : %.4f ' % (key, self.estimator.train_stats[key])
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
                 monitor='val_accuracy',
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
        epoch = self.estimator.current_epoch
        # add extension for weights
        if '.params' not in self.filepath:
            self.filepath += '.params'
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                # check if monitor exists in train stats
                if self.monitor not in self.estimator.train_stats:
                    warnings.warn(RuntimeWarning('Unable to find %s in training statistics, make sure the monitor value'
                                                 'starts with `train_ `or `val_` and contains loss/metric name, ',
                                                 'for example val_accuracy', self.monitor))
                    self.estimator.net.save_parameters(self.filepath)
                else:
                    current = self.estimator.train_stats[self.monitor]
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            self.logger.info('\n[Epoch %d] %s improved from %0.5f to %0.5f,'
                                             ' saving model to %s',
                                             epoch, self.monitor, self.best, current, self.filepath)
                        self.best = current
                        self.estimator.net.save_parameters(self.filepath)
                    else:
                        if self.verbose > 0:
                            self.logger.info('\n[Epoch %d] %s did not improve from %0.5f, skipping save model',
                                             epoch, self.monitor, self.best)
            else:
                if self.verbose > 0:
                    logging.info('\nEpoch %d: saving model to %s', epoch, self.filepath)
                self.estimator.net.save_parameters(self.filepath)


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
                 monitor='val_accuracy',
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
        epoch = self.estimator.current_epoch
        if self.monitor not in self.estimator.train_stats:
            warnings.warn(RuntimeWarning('Unable to find %s in training statistics, make sure the monitor value'
                                         'starts with `train_ `or `val_` and contains loss/metric name, ',
                                         'for example val_accuracy', self.monitor))
        else:
            current = self.estimator.train_stats[self.monitor]
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.estimator.stop_training = True

    def train_end(self):
        if self.stopped_epoch > 0:
            self.logger.info('Epoch %d: early stopping due to %s not improving', self.stopped_epoch, self.monitor)
