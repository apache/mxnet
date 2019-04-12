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

import logging
import os
import time
import warnings

import numpy as np


class TrainBegin(object):
    def train_begin(self, *args, **kwargs):
        pass

class TrainEnd(object):
    def train_end(self, *args, **kwargs):
        pass

class EpochBegin(object):
    def epoch_begin(self, *args, **kwargs):
        pass

class EpochEnd(object):
    def epoch_end(self, *args, **kwargs):
        return False

class BatchBegin(object):
    def batch_begin(self, *args, **kwargs):
        pass

class BatchEnd(object):
    def batch_end(self, *args, **kwargs):
        return False


class MetricHandler(EpochBegin, BatchEnd):
    def __init__(self, train_loss, train_metrics):
        self.train_loss = train_loss
        self.train_metrics = train_metrics
        # order to be called among all callbacks
        # metrics need to be calculated before other callbacks can access them
        self.rank = 1

    def epoch_begin(self, *args, **kwargs):
        for metric in self.train_loss + self.train_metrics:
            metric.reset()

    def batch_end(self, *args, **kwargs):
        pred = kwargs['pred']
        label = kwargs['label']
        loss = kwargs['loss']
        for metric in self.train_metrics:
            metric.update(label, pred)
        for metric in self.train_loss:
            metric.update(0, loss)

class ValidationHandler(BatchEnd, EpochEnd):
    def __init__(self,
                 val_data,
                 eval_fn,
                 val_loss,
                 val_metrics=None,
                 epoch_period=1,
                 batch_period=None):
        self.val_data = val_data
        self.eval_fn = eval_fn
        self.epoch_period = epoch_period
        self.batch_period = batch_period
        self.val_loss = val_loss
        self.val_metrics = val_metrics
        self.num_batches = 0
        self.num_epochs = 0
        # order to be called among all callbacks
        # validation metrics need to be calculated before other callbacks can access them
        self.rank = 1

    def batch_end(self, *args, **kwargs):
        if self.batch_period and self.num_batches % self.batch_period == 0:
            self.eval_fn(val_data=self.val_data,
                         val_loss= self.val_loss,
                         val_metrics=self.val_metrics)
        self.num_batches += 1

    def epoch_end(self, *args, **kwargs):
        if self.num_epochs % self.epoch_period == 0:
            self.eval_fn(val_data=self.val_data,
                         val_loss= self.val_loss,
                         val_metrics=self.val_metrics)

        self.num_epochs += 1


class LoggingHandler(TrainBegin, TrainEnd, EpochBegin, EpochEnd, BatchBegin, BatchEnd):
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

    def __init__(self, file_name=None,
                 file_location=None,
                 verbose=LOG_VERBOSITY_PER_EPOCH,
                 train_metrics=None,
                 val_metrics=None):
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
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.batch_index = 0
        self.current_epoch = 0
        self.processed_samples = 0


    def train_begin(self, *args, **kwargs):
        self.train_start = time.time()
        if 'trainer' in kwargs:
            optimizer = kwargs['trainer'].optimizer.__class__.__name__
            lr = kwargs['trainer'].learning_rate
            self.logger.info("Training begin: using optimizer %s "
                             "with current learning rate %.4f ",
                             optimizer, lr)
        if 'epochs' in kwargs:
            self.logger.info("Train for %d epochs.", kwargs['epochs'])

    def train_end(self, *args, **kwargs):
        train_time = time.time() - self.train_start
        msg = 'Train finished using total %ds with %d epochs.' % (train_time, self.current_epoch)
        # log every result in train stats including train/validation loss & metrics
        for metric in self.train_metrics + self.val_metrics:
            name, value = metric.get()
            msg += '%s : %.4f ' % (name, value)
        self.logger.info(msg)

    def batch_begin(self, *args, **kwargs):
        if self.verbose == self.LOG_VERBOSITY_PER_BATCH:
            self.batch_start = time.time()

    def batch_end(self, *args, **kwargs):
        if self.verbose == self.LOG_VERBOSITY_PER_BATCH:
            batch_time = time.time() - self.batch_start
            msg = '[Epoch %d] [Batch %d] ' % (self.current_epoch, self.batch_index)
            self.processed_samples += kwargs['batch_size']
            msg += '[Samples %s] ' % (self.processed_samples)
            msg += 'time/batch: %.3fs ' % batch_time
            for metric in self.train_metrics:
                # only log current training loss & metric after each batch
                name, value = metric.get()
                msg += '%s : %.4f ' % (name, value)
            self.logger.info(msg)
            self.batch_index += 1

    def epoch_begin(self, *args, **kwargs):
        if self.verbose >= self.LOG_VERBOSITY_PER_EPOCH:
            self.epoch_start = time.time()

    def epoch_end(self, *args, **kwargs):
        if self.verbose >= self.LOG_VERBOSITY_PER_EPOCH:
            epoch_time = time.time() - self.epoch_start
            msg = '\n[Epoch %d] finished in %.3fs: ' % (self.current_epoch, epoch_time)
            for monitor in self.train_metrics + self.val_metrics:
                name, value = monitor.get()
                msg += '%s : %.4f ' % (name, value)
            self.logger.info(msg)
            self.current_epoch += 1


class CheckpointHandler(object):
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


class EarlyStoppingHandler(object):
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
