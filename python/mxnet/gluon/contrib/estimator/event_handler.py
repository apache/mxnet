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
# pylint: disable=wildcard-import, unused-argument
"""Gluon EventHandlers for Estimators"""

import logging
import os
import time
import warnings

import numpy as np

from ....metric import EvalMetric, Loss


class TrainBegin(object):
    def train_begin(self, estimator, *args, **kwargs):
        pass


class TrainEnd(object):
    def train_end(self, estimator, *args, **kwargs):
        pass


class EpochBegin(object):
    def epoch_begin(self, estimator, *args, **kwargs):
        pass


class EpochEnd(object):
    def epoch_end(self, estimator, *args, **kwargs):
        return False


class BatchBegin(object):
    def batch_begin(self, estimator, *args, **kwargs):
        pass


class BatchEnd(object):
    def batch_end(self, estimator, *args, **kwargs):
        return False


class MetricHandler(EpochBegin, BatchEnd):
    """Metric Handler that update metric values at batch end

    :py:class:`MetricHandler` takes model predictions and true labels
    and update the metrics, it also update metric wrapper for loss with loss values
    Validation loss and metrics will be handled by :py:class:`ValidationHandler`

    Parameters
    ----------
    train_metrics : List of EvalMetrics
        training metrics to be updated at batch end
    """

    def __init__(self, train_metrics):
        self.train_metrics = train_metrics or []
        # order to be called among all callbacks
        # metrics need to be calculated before other callbacks can access them
        self.priority = -np.Inf

    def epoch_begin(self, estimator, *args, **kwargs):
        for metric in self.train_metrics:
            metric.reset()

    def batch_end(self, estimator, *args, **kwargs):
        pred = kwargs['pred']
        label = kwargs['label']
        loss = kwargs['loss']
        for metric in self.train_metrics:
            if isinstance(metric, Loss):
                # metric wrapper for loss values
                metric.update(0, loss)
            else:
                metric.update(label, pred)


class ValidationHandler(BatchEnd, EpochEnd):
    """"Validation Handler that evaluate model on validation dataset

    :py:class:`ValidationHandler` takes validation dataset, an evaluation function,
    metrics to be evaluated, and how often to run the validation. You can provide custom
    evaluation function or use the one provided my :py:class:`Estimator`

    Parameters
    ----------
    val_data : DataLoader
        validation data set to run evaluation
    eval_fn : function
        a function defines how to run evaluation and
        calculate loss and metrics
    val_metrics : List of EvalMetrics
        validation metrics to be updated
    epoch_period : int, default 1
        how often to run validation at epoch end, by default
        validate every epoch
    batch_period : int, default None
        how often to run validation at batch end, by default
        does not validate at batch end
    """

    def __init__(self,
                 val_data,
                 eval_fn,
                 val_metrics=None,
                 epoch_period=1,
                 batch_period=None):
        self.val_data = val_data
        self.eval_fn = eval_fn
        self.epoch_period = epoch_period
        self.batch_period = batch_period
        self.val_metrics = val_metrics
        self.num_batches = 0
        self.num_epochs = 0
        # order to be called among all callbacks
        # validation metrics need to be calculated before other callbacks can access them
        self.priority = -np.Inf

    def batch_end(self, estimator, *args, **kwargs):
        if self.batch_period and self.num_batches % self.batch_period == 0:
            self.eval_fn(val_data=self.val_data,
                         val_metrics=self.val_metrics)
        self.num_batches += 1

    def epoch_end(self, estimator, *args, **kwargs):
        if self.num_epochs % self.epoch_period == 0:
            self.eval_fn(val_data=self.val_data,
                         val_metrics=self.val_metrics)

        self.num_epochs += 1


class LoggingHandler(TrainBegin, TrainEnd, EpochBegin, EpochEnd, BatchBegin, BatchEnd):
    """Basic Logging Handler that applies to every Gluon estimator by default.

    :py:class:`LoggingHandler` logs hyper-parameters, training statistics,
    and other useful information during training

    Parameters
    ----------
    file_name : str
        file name to save the logs
    file_location : str
        file location to save the logs
    verbose : int, default LOG_VERBOSITY_PER_EPOCH
        Limit the granularity of metrics displayed during training process
        verbose=LOG_VERBOSITY_PER_EPOCH: display metrics every epoch
        verbose=LOG_VERBOSITY_PER_BATCH: display metrics every batch
    train_metrics : list of EvalMetrics
        training metrics to be logged, logged at batch end, epoch end, train end
    val_metrics : list of EvalMetrics
        validation metrics to be logged, logged at epoch end, train end
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
        self.train_metrics = train_metrics or []
        self.val_metrics = val_metrics or []
        self.batch_index = 0
        self.current_epoch = 0
        self.processed_samples = 0
        # logging handler need to be called at last to make sure all states are updated
        # it will also shut down logging at train end
        self.priority = np.Inf

    def train_begin(self, estimator, *args, **kwargs):
        self.train_start = time.time()
        trainer = estimator.trainer
        optimizer = trainer.optimizer.__class__.__name__
        lr = trainer.learning_rate
        self.logger.info("Training begin: using optimizer %s "
                         "with current learning rate %.4f ",
                         optimizer, lr)
        self.logger.info("Train for %d epochs.", estimator.max_epochs)

    def train_end(self, estimator, *args, **kwargs):
        train_time = time.time() - self.train_start
        msg = 'Train finished using total %ds with %d epochs.' % (train_time, self.current_epoch)
        # log every result in train stats including train/validation loss & metrics
        for metric in self.train_metrics + self.val_metrics:
            name, value = metric.get()
            msg += '%s : %.4f ' % (name, value)
        self.logger.info(msg)
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        logging.shutdown()

    def batch_begin(self, estimator, *args, **kwargs):
        if self.verbose == self.LOG_VERBOSITY_PER_BATCH:
            self.batch_start = time.time()

    def batch_end(self, estimator, *args, **kwargs):
        if self.verbose == self.LOG_VERBOSITY_PER_BATCH:
            batch_time = time.time() - self.batch_start
            msg = '[Epoch %d] [Batch %d] ' % (self.current_epoch, self.batch_index)
            self.processed_samples += kwargs['batch'][0].shape[0]
            msg += '[Samples %s] ' % (self.processed_samples)
            msg += 'time/batch: %.3fs ' % batch_time
            for metric in self.train_metrics:
                # only log current training loss & metric after each batch
                name, value = metric.get()
                msg += '%s : %.4f ' % (name, value)
            self.logger.info(msg)
            self.batch_index += 1

    def epoch_begin(self, estimator, *args, **kwargs):
        if self.verbose >= self.LOG_VERBOSITY_PER_EPOCH:
            self.epoch_start = time.time()

    def epoch_end(self, estimator, *args, **kwargs):
        if self.verbose >= self.LOG_VERBOSITY_PER_EPOCH:
            epoch_time = time.time() - self.epoch_start
            msg = '\n[Epoch %d] finished in %.3fs: ' % (self.current_epoch, epoch_time)
            for monitor in self.train_metrics + self.val_metrics:
                name, value = monitor.get()
                msg += '%s : %.4f ' % (name, value)
            self.logger.info(msg)
            self.current_epoch += 1
            self.batch_index = 0


class CheckpointHandler(BatchEnd, EpochEnd):
    """Save the model after every epoch.

    :py:class:`CheckpointHandler` save the network parameters every epoch

    Parameters
    ----------
    filepath : str
        file name to save the parameters, it can contain directories,
        for example: ./saved_model/resnet.params
    monitor: EvalMetric
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
                 monitor=None,
                 verbose=0,
                 save_best_only=False,
                 mode='auto',
                 epoch_period=1,
                 batch_period=None):
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        if self.save_best_only and not isinstance(self.monitor, EvalMetric):
            raise ValueError("To save best model only, please provide one of the metric objects as monitor, "
                             "You can create these objects using estimator.prepare_loss_and_metric()")
        self.epoch_period = epoch_period
        self.batch_period = batch_period
        self.num_batches = 0
        self.num_epochs = 0
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
            if 'acc' in self.monitor.get()[0].lower():
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def batch_end(self, estimator, *args, **kwargs):
        self._save_checkpoint(estimator.net, "Batch", self.num_batches)
        self.num_batches += 1

    def epoch_end(self, estimator, *args, **kwargs):
        self._save_checkpoint(estimator.net, "Epoch", self.num_epochs)
        self.num_epochs += 1

    def _save_checkpoint(self, net, period_name, period_value):
        # add extension for weights
        if '.params' not in self.filepath:
            self.filepath += '.params'
        if self.num_epochs % self.epoch_period == 0:
            if self.save_best_only:
                monitor_name, monitor_value = self.monitor.get()
                # check if monitor exists in train stats
                if np.isnan(monitor_value):
                    warnings.warn(RuntimeWarning('%s is not updated, make sure you pass one of the metric objects'
                                                 'as monitor, you can use estimator.prepare_loss_and_metrics to'
                                                 'create all metric objects', monitor_name))
                    net.save_parameters(self.filepath)
                else:
                    if self.monitor_op(monitor_value, self.best):
                        if self.verbose > 0:
                            self.logger.info('\n[%s %d] %s improved from %0.5f to %0.5f,'
                                             ' saving model to %s',
                                             period_name, period_value, monitor_name,
                                             self.best, monitor_value, self.filepath)
                        self.best = monitor_value
                        net.save_parameters(self.filepath)
                    else:
                        if self.verbose > 0:
                            self.logger.info('\n[%s %d] %s did not improve from %0.5f, skipping save model',
                                             period_name, period_value, monitor_name, self.best)
            else:
                if self.verbose > 0:
                    logging.info('\n%s %d: saving model to %s', period_name, period_value, self.filepath)
                net.save_parameters(self.filepath)


class EarlyStoppingHandler(TrainBegin, EpochEnd, TrainEnd):
    """Early stop training if monitored value is not improving

    Parameters
    ----------
    estimator : Estimator
        The :py:class:`Estimator` to get training statistics
    monitor: EvalMetric
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
                 monitor,
                 min_delta=0,
                 patience=0,
                 mode='auto',
                 baseline=None):
        super(EarlyStoppingHandler, self).__init__()

        if not isinstance(monitor, EvalMetric):
            raise ValueError("Please provide one of the metric objects as monitor, "
                             "You can create these objects using estimator.prepare_loss_and_metric()")
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.num_epochs = 0
        self.stop_training = False
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
            if 'acc' in self.monitor.get()[0].lower():
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def train_begin(self, estimator, *args, **kwargs):
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def epoch_end(self, estimator, *args, **kwargs):
        monitor_name, monitor_value = self.monitor.get()
        if np.isnan(monitor_value):
            warnings.warn(RuntimeWarning('%s is not updated, make sure you pass one of the metric objects'
                                         'as monitor, you can use estimator.prepare_loss_and_metrics to'
                                         'create all metric objects', monitor_name))
        else:
            if self.monitor_op(monitor_value - self.min_delta, self.best):
                self.best = monitor_value
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = self.num_epochs
                    self.stop_training = True
        return self.stop_training

    def train_end(self, estimator, *args, **kwargs):
        if self.stopped_epoch > 0:
            self.logger.info('Epoch %d: early stopping due to %s not improving',
                             self.stopped_epoch, self.monitor.get()[0])
