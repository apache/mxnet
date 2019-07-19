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


class StoppingHandler(TrainBegin, BatchEnd, EpochEnd):
    """Stop conditions to stop training
    Stop training if maximum number of batches or epochs
    reached.

    Parameters
    ----------
    max_epoch : int, default None
        Number of maximum epochs to train.
    max_batch : int, default None
        Number of maximum batches to train.

    """

    def __init__(self, max_epoch=None, max_batch=None):
        self.max_epoch = max_epoch
        self.max_batch = max_batch
        self.current_batch = 0
        self.current_epoch = 0
        self.stop_training = False

    def train_begin(self, estimator, *args, **kwargs):
        self.max_epoch = estimator.max_epoch
        self.max_batch = estimator.max_batch
        self.current_batch = 0
        self.current_epoch = 0

    def batch_end(self, estimator, *args, **kwargs):
        self.current_batch += 1
        if self.current_batch == self.max_batch:
            self.stop_training = True
        return self.stop_training

    def epoch_end(self, estimator, *args, **kwargs):
        self.current_epoch += 1
        if self.current_epoch == self.max_epoch:
            self.stop_training = True
        return self.stop_training


class MetricHandler(EpochBegin, BatchEnd):
    """Metric Handler that update metric values at batch end

    :py:class:`MetricHandler` takes model predictions and true labels
    and update the metrics, it also update metric wrapper for loss with loss values.
    Validation loss and metrics will be handled by :py:class:`ValidationHandler`

    Parameters
    ----------
    train_metrics : List of EvalMetrics
        Training metrics to be updated at batch end.
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


class ValidationHandler(TrainBegin, BatchEnd, EpochEnd):
    """"Validation Handler that evaluate model on validation dataset

    :py:class:`ValidationHandler` takes validation dataset, an evaluation function,
    metrics to be evaluated, and how often to run the validation. You can provide custom
    evaluation function or use the one provided my :py:class:`Estimator`

    Parameters
    ----------
    val_data : DataLoader
        Validation data set to run evaluation.
    eval_fn : function
        A function defines how to run evaluation and
        calculate loss and metrics.
    val_metrics : List of EvalMetrics
        Validation metrics to be updated.
    epoch_period : int, default 1
        How often to run validation at epoch end, by default
        :py:class:`ValidationHandler` validate every epoch.
    batch_period : int, default None
        How often to run validation at batch end, by default
        :py:class:`ValidationHandler` does not validate at batch end.
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
        self.current_batch = 0
        self.current_epoch = 0
        # order to be called among all callbacks
        # validation metrics need to be calculated before other callbacks can access them
        self.priority = -np.Inf
        self.logger = logging.getLogger(__name__)

    def train_begin(self, estimator, *args, **kwargs):
        # reset epoch and batch counter
        self.current_batch = 0
        self.current_epoch = 0

    def batch_end(self, estimator, *args, **kwargs):
        self.current_batch += 1
        if self.batch_period and self.current_batch % self.batch_period == 0:
            self.eval_fn(val_data=self.val_data,
                         val_metrics=self.val_metrics)
            msg = '[Epoch %d] ValidationHandler: %d batches reached, ' \
                  % (self.current_epoch, self.current_batch)
            for monitor in self.val_metrics:
                name, value = monitor.get()
                msg += '%s: %.4f, ' % (name, value)
            self.logger.info(msg.rstrip(','))

    def epoch_end(self, estimator, *args, **kwargs):
        self.current_epoch += 1
        if self.epoch_period and self.current_epoch % self.epoch_period == 0:
            self.eval_fn(val_data=self.val_data,
                         val_metrics=self.val_metrics)


class LoggingHandler(TrainBegin, TrainEnd, EpochBegin, EpochEnd, BatchBegin, BatchEnd):
    """Basic Logging Handler that applies to every Gluon estimator by default.

    :py:class:`LoggingHandler` logs hyper-parameters, training statistics,
    and other useful information during training

    Parameters
    ----------
    file_name : str
        File name to save the logs.
    file_location : str
        File location to save the logs.
    filemode : str, default 'a'
        Logging file mode, default using append mode.
    verbose : int, default LOG_PER_EPOCH
        Limit the granularity of metrics displayed during training process.
        verbose=LOG_PER_EPOCH: display metrics every epoch
        verbose=LOG_PER_BATCH: display metrics every batch
    train_metrics : list of EvalMetrics
        Training metrics to be logged, logged at batch end, epoch end, train end.
    val_metrics : list of EvalMetrics
        Validation metrics to be logged, logged at epoch end, train end.
    """

    LOG_PER_EPOCH = 1
    LOG_PER_BATCH = 2

    def __init__(self, file_name=None,
                 file_location=None,
                 filemode='a',
                 verbose=LOG_PER_EPOCH,
                 train_metrics=None,
                 val_metrics=None):
        super(LoggingHandler, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)
        # save logger to file only if file name or location is specified
        if file_name or file_location:
            file_name = file_name or 'estimator_log'
            file_location = file_location or './'
            file_handler = logging.FileHandler(os.path.join(file_location, file_name), mode=filemode)
            self.logger.addHandler(file_handler)
        if verbose not in [self.LOG_PER_EPOCH, self.LOG_PER_BATCH]:
            raise ValueError("verbose level must be either LOG_PER_EPOCH or "
                             "LOG_PER_BATCH, received %s. "
                             "E.g: LoggingHandler(verbose=LoggingHandler.LOG_PER_EPOCH)"
                             % verbose)
        self.verbose = verbose
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
        if estimator.max_epoch:
            self.logger.info("Train for %d epochs.", estimator.max_epoch)
        else:
            self.logger.info("Train for %d batches.", estimator.max_batch)
        # reset all counters
        self.current_epoch = 0
        self.batch_index = 0
        self.processed_samples = 0

    def train_end(self, estimator, *args, **kwargs):
        train_time = time.time() - self.train_start
        msg = 'Train finished using total %ds with %d epochs. ' % (train_time, self.current_epoch)
        # log every result in train stats including train/validation loss & metrics
        for metric in self.train_metrics + self.val_metrics:
            name, value = metric.get()
            msg += '%s: %.4f, ' % (name, value)
        self.logger.info(msg.rstrip(', '))
        # make a copy of handler list and remove one by one
        # as removing handler will edit the handler list
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        logging.shutdown()

    def batch_begin(self, estimator, *args, **kwargs):
        if self.verbose == self.LOG_PER_BATCH:
            self.batch_start = time.time()

    def batch_end(self, estimator, *args, **kwargs):
        if self.verbose == self.LOG_PER_BATCH:
            batch_time = time.time() - self.batch_start
            msg = '[Epoch %d][Batch %d]' % (self.current_epoch, self.batch_index)
            self.processed_samples += kwargs['batch'][0].shape[0]
            msg += '[Samples %s] ' % (self.processed_samples)
            msg += 'time/batch: %.3fs ' % batch_time
            for metric in self.train_metrics:
                # only log current training loss & metric after each batch
                name, value = metric.get()
                msg += '%s: %.4f, ' % (name, value)
            self.logger.info(msg.rstrip(', '))
        self.batch_index += 1

    def epoch_begin(self, estimator, *args, **kwargs):
        if self.verbose >= self.LOG_PER_EPOCH:
            self.epoch_start = time.time()
            self.logger.info("[Epoch %d] Begin, current learning rate: %.4f",
                             self.current_epoch, estimator.trainer.learning_rate)

    def epoch_end(self, estimator, *args, **kwargs):
        if self.verbose >= self.LOG_PER_EPOCH:
            epoch_time = time.time() - self.epoch_start
            msg = '[Epoch %d] Finished in %.3fs, ' % (self.current_epoch, epoch_time)
            for monitor in self.train_metrics + self.val_metrics:
                name, value = monitor.get()
                msg += '%s: %.4f, ' % (name, value)
            self.logger.info(msg.rstrip(', '))
        self.current_epoch += 1
        self.batch_index = 0


class CheckpointHandler(TrainBegin, BatchEnd, EpochEnd):
    """Save the model after user define period

    :py:class:`CheckpointHandler` saves the network architecture after first batch if the model
    can be fully hybridized, saves model parameters and trainer states after user defined period,
    default saves every epoch.

    Parameters
    ----------
    model_dir : str
        File directory to save all the model related files including model architecture,
        model parameters, and trainer states.
    model_prefix : str default 'model'
        Prefix to add for all checkpoint file names.
    monitor: EvalMetric, default None
        The metrics to monitor and determine if model has improved
    verbose: int, default 0
        Verbosity mode, 1 means inform user every time a checkpoint is saved
    save_best: bool, default False
        If True, monitor must not be None, :py:class:`CheckpointHandler` will save the
        model parameters and trainer states with the best monitored value.
    mode: str, default 'auto'
        One of {auto, min, max}, if `save_best=True`, the comparison to make
        and determine if the monitored value has improved. if 'auto' mode,
        :py:class:`CheckpointHandler` will try to use min or max based on
        the monitored metric name.
    epoch_period: int, default 1
        Epoch intervals between saving the network. By default, checkpoints are
        saved every epoch.
    batch_period: int, default None
        Batch intervals between saving the network.
        By default, checkpoints are not saved based on the number of batches.
    max_checkpoints : int, default 5
        Maximum number of checkpoint files to keep in the model_dir, older checkpoints
        will be removed. Best checkpoint file is not counted.
    resume_from_checkpoint : bool, default False
        Whether to resume training from checkpoint in model_dir. If True and checkpoints
        found, :py:class:`CheckpointHandler` will load net parameters and trainer states,
        and train the remaining of epochs and batches.
    """

    def __init__(self,
                 model_dir,
                 model_prefix='model',
                 monitor=None,
                 verbose=0,
                 save_best=False,
                 mode='auto',
                 epoch_period=1,
                 batch_period=None,
                 max_checkpoints=5,
                 resume_from_checkpoint=False):
        self.monitor = monitor
        self.verbose = verbose
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        self.save_best = save_best
        if self.save_best and not isinstance(self.monitor, EvalMetric):
            raise ValueError("To save best model only, please provide one of the metric objects as monitor, "
                             "You can get these objects using estimator.prepare_loss_and_metric()")
        self.epoch_period = epoch_period
        self.batch_period = batch_period
        self.current_batch = 0
        self.current_epoch = 0
        self.max_checkpoints = max_checkpoints
        self.resume_from_checkpoint = resume_from_checkpoint
        self.saved_checkpoints = []
        self.logger = logging.getLogger(__name__)
        if self.save_best:
            if mode not in ['auto', 'min', 'max']:
                warnings.warn('ModelCheckpoint mode %s is unknown, '
                              'fallback to auto mode. CheckpointHandler will use'
                              'max mode for f1 and accuracy metric comparison and '
                              'use min mode other wise' % (mode),
                              RuntimeWarning)
                mode = 'auto'

            if mode == 'min':
                self.monitor_op = np.less
                self.best = np.Inf
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                # use greater for accuracy and f1 and less otherwise
                if 'acc' or 'f1' in self.monitor.get()[0].lower():
                    self.logger.info("`greater` operator will be used to determine "
                                     "if %s has improved, please use `min` for mode "
                                     "if you want otherwise", self.monitor.get()[0])
                    self.monitor_op = np.greater
                else:
                    self.logger.info("`less` operator will be used to determine "
                                     "if %s has improved, please use `max` for mode "
                                     "if you want otherwise", self.monitor.get()[0])
                    self.monitor_op = np.less

    def train_begin(self, estimator, *args, **kwargs):
        # reset all counters
        self.current_epoch = 0
        self.current_batch = 0
        if self.save_best:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf # pylint: disable=comparison-with-callable
        if self.resume_from_checkpoint:
            error_msg = "To use resume from checkpoint, you must only specify " \
                        "the same type of period you used for training." \
                        "For example, if you are training based on number of epochs," \
                        "you must save only based on epochs, and set batch_period to None."
            if estimator.max_batch:
                assert self.batch_period, error_msg
                assert not self.epoch_period, error_msg
            if estimator.max_epoch:
                assert self.epoch_period, error_msg
                assert not self.batch_period, error_msg

            self._resume_from_checkpoint(estimator)

    def batch_end(self, estimator, *args, **kwargs):
        # only save symbol once after first batch
        if self.current_batch == 0:
            self._save_symbol(estimator)
        if self.batch_period and (self.current_batch + 1) % self.batch_period == 0:
            self._save_checkpoint(estimator)
        self.current_batch += 1

    def epoch_end(self, estimator, *args, **kwargs):
        if self.epoch_period and (self.current_epoch + 1) % self.epoch_period == 0:
            self._save_checkpoint(estimator)
        self.current_epoch += 1

    def _save_checkpoint(self, estimator):
        # if resumed from checkpoint, increment checkpoint number
        if self.resume_from_checkpoint:
            save_epoch_number = self.current_epoch + self.trained_epoch + 1
            if estimator.max_epoch:
                # checkpoint saved at epoch end, batch number already incremented
                save_batch_number = self.current_batch + self.trained_batch
            else:
                save_batch_number = self.current_batch + self.trained_batch + 1
        else:
            save_epoch_number = self.current_epoch
            save_batch_number = self.current_batch
        prefix = "%s-epoch%dbatch%d" % (self.model_prefix, save_epoch_number, save_batch_number)
        self._save_params_and_trainer(estimator, prefix)
        if self.verbose > 0:
            self.logger.info('[Epoch %d] CheckpointHandler: trained total %d batches, '
                             'saving model at %s with prefix: %s',
                             self.current_epoch, self.current_batch + 1, self.model_dir, prefix)

        if self.save_best:
            monitor_name, monitor_value = self.monitor.get()
            # check if monitor exists in train stats
            if np.isnan(monitor_value):
                warnings.warn(RuntimeWarning('Skipping save best because %s is not updated, make sure you '
                                             'pass one of the metric objects as monitor, '
                                             'you can use estimator.prepare_loss_and_metrics to'
                                             'create all metric objects', monitor_name))
            else:
                if self.monitor_op(monitor_value, self.best):
                    prefix = self.model_prefix + '-best'
                    self._save_params_and_trainer(estimator, prefix)
                    self.best = monitor_value
                    if self.verbose > 0:
                        self.logger.info('[Epoch %d] CheckpointHandler: '
                                         '%s improved from %0.5f to %0.5f, '
                                         'updating best model at %s with prefix: %s',
                                         self.current_epoch, monitor_name,
                                         self.best, monitor_value, self.model_dir, prefix)
                else:
                    if self.verbose > 0:
                        self.logger.info('[Epoch %d] CheckpointHandler: '
                                         '%s did not improve from %0.5f, '
                                         'skipping updating best model',
                                         self.current_batch, monitor_name,
                                         self.best)

    def _save_symbol(self, estimator):
        symbol_file = os.path.join(self.model_dir, self.model_prefix + '-symbol.json')
        if hasattr(estimator.net, '_cached_graph'):
            sym = estimator.net._cached_graph[1]
            sym.save(symbol_file)
        else:
            self.logger.info("Model architecture(symbol file) is not saved, please use HybridBlock"
                             "to construct your model, can call net.hybridize() before passing to"
                             "Estimator in order to save model architecture as %s.", symbol_file)

    def _save_params_and_trainer(self, estimator, file_prefix):
        param_file = os.path.join(self.model_dir, file_prefix + '.params')
        trainer_file = os.path.join(self.model_dir, file_prefix + '.states')
        estimator.net.save_parameters(param_file)
        estimator.trainer.save_states(trainer_file)

        # only count checkpoints with epoch or batch number in file name
        if 'best' not in file_prefix:
            self.saved_checkpoints.append(file_prefix)
        # remove old checkpoint when max number of checkpoints reached
        if len(self.saved_checkpoints) > self.max_checkpoints:
            prefix = self.saved_checkpoints.pop(0)
            for fname in os.listdir(self.model_dir):
                if fname.startswith(prefix):
                    os.remove(os.path.join(self.model_dir, fname))

    def _resume_from_checkpoint(self, estimator):
        prefix = self.model_prefix + '-epoch'
        self.trained_epoch = self._find_max_iteration(
            dir=self.model_dir,
            prefix=prefix,
            start='epoch',
            end='batch',
            saved_checkpoints=self.saved_checkpoints)
        prefix += str(self.trained_epoch)
        self.trained_batch = self._find_max_iteration(
            dir=self.model_dir,
            prefix=prefix,
            start='batch',
            end='.params')

        if self.trained_epoch == -1:
            msg = "CheckpointHandler: No checkpoint found, training from scratch for "
            if estimator.max_batch:
                msg += "%d batches" % estimator.max_batch
            else:
                msg += "%d epochs" % estimator.max_epoch
            self.logger.info(msg)
        else:
            msg = "CheckpointHandler: Checkpoint resumed from epoch %d batch %d, " \
                  "continue to train for " % (self.trained_epoch, self.trained_batch)
            # change maximum number of epoch or batch to train if resumed from epoch checkpoint
            if estimator.max_epoch:
                if self.trained_epoch >= estimator.max_epoch - 1:
                    raise ValueError("Found checkpoint with maximum number of epoch %d reached, please specify "
                                     "resume_from_checkpoint=False (default value) if you wan to train from scratch."
                                     % estimator.max_epoch)
                estimator.max_epoch = estimator.max_epoch - self.trained_epoch - 1
                msg += "%d epochs " % estimator.max_epoch
            if estimator.max_batch:
                if self.trained_batch >= estimator.max_batch - 1:
                    raise ValueError("Found checkpoint with maximum number of batch %d reached, please specify"
                                     "resume_from_checkpoint=False (default value) if you wan to train from scratch."
                                     % self.trained_batch)
                estimator.max_batch = estimator.max_batch - self.trained_batch - 1
                msg += "%d batches " % estimator.max_batch
            # load checkpoint
            param_file = "%s-epoch%dbatch%d.params" % (self.model_prefix, self.trained_epoch, self.trained_batch)
            param_file = os.path.join(self.model_dir, param_file)
            trainer_file = "%s-epoch%dbatch%d.states" % (self.model_prefix, self.trained_epoch, self.trained_batch)
            trainer_file = os.path.join(self.model_dir, trainer_file)
            assert os.path.exists(param_file), "Failed to load checkpoint, %s does not exist" % param_file
            assert os.path.exists(trainer_file), "Failed to load checkpoint, %s does not exist" % trainer_file
            estimator.net.load_parameters(param_file, ctx=estimator.context)
            estimator.trainer.load_states(trainer_file)
            self.logger.warning(msg)

    def _find_max_iteration(self, dir, prefix, start, end, saved_checkpoints=None):
        error_msg = "Error parsing checkpoint file, please check your " \
                    "checkpoints have the format: " \
                    "{model_name}-epoch{epoch_number}batch{batch_number}.params, " \
                    "there should also be a .states file for each .params file "
        max_iter = -1
        for fname in os.listdir(dir):
            if fname.startswith(prefix) and '.params' in fname:
                if saved_checkpoints:
                    # save prefix of existing checkpoints
                    saved_checkpoints.append(fname[:fname.find('.params')])
                try:
                    # find trained number of epoch
                    iter = int(fname[fname.find(start) + len(start): fname.find(end)])
                    if iter > max_iter:
                        max_iter = iter
                except ValueError:
                    raise ValueError(error_msg)
        return max_iter


class EarlyStoppingHandler(TrainBegin, EpochEnd, TrainEnd):
    """Early stop training if monitored value is not improving

    Parameters
    ----------
    monitor: EvalMetric
        The metric to monitor, and stop training if this metric does not improve.
    min_delta: float, default 0
        Minimal change in monitored value to be considered as an improvement.
    patience: int, default 0
        Number of epochs to wait for improvement before terminate training.
    mode: str, default 'auto'
        One of {auto, min, max}, if `save_best_only=True`, the comparison to make
        and determine if the monitored value has improved. if 'auto' mode, checkpoint
        handler will try to use min or max based on the monitored metric name.
    baseline: float
        Baseline value to compare the monitored value with.
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
        self.current_epoch = 0
        self.stop_training = False
        self.logger = logging.getLogger(__name__)

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode. CheckpointHandler will use'
                          'max mode for f1 and accuracy metric comparison and '
                          'use min mode other wise' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' or 'f1' in self.monitor.get()[0].lower():
                self.logger.info("`greater` operator is used to determine "
                                 "if %s has improved, please use `min` for mode "
                                 "if you want otherwise", self.monitor.get()[0])
                self.monitor_op = np.greater
            else:
                self.logger.info("`less` operator is used to determine "
                                 "if %s has improved, please use `max` for mode "
                                 "if you want otherwise", self.monitor.get()[0])
                self.monitor_op = np.less

        if self.monitor_op == np.greater: # pylint: disable=comparison-with-callable
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def train_begin(self, estimator, *args, **kwargs):
        self.wait = 0
        self.stopped_epoch = 0
        self.current_epoch = 0
        self.stop_training = False
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf # pylint: disable=comparison-with-callable

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
                    self.stopped_epoch = self.current_epoch
                    self.stop_training = True
        self.current_epoch += 1
        return self.stop_training

    def train_end(self, estimator, *args, **kwargs):
        if self.stopped_epoch > 0:
            self.logger.info('[Epoch %d] EarlyStoppingHanlder: early stopping due to %s not improving',
                             self.stopped_epoch, self.monitor.get()[0])
