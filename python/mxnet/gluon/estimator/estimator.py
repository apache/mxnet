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
"""Gluon Estimator"""

import warnings

from .event_handler import LoggingHandler
from ... import gluon, autograd
from ...context import Context, cpu, gpu, num_gpus
from ...io import DataIter
from ...metric import EvalMetric, Loss

__all__ = ['Estimator']


class Estimator(object):
    """Estimator Class for easy model training

    :py:class:`Estimator` can be used to facilitate the training & validation process


    Parameters
    ----------
    loss : Loss or list of Loss
        Loss(objective functions) to calculate during training
    metrics : EvalMetric or list of EvalMetric
        Metrics for evaluating models
    initializer : Initializer
        initializer to initialize the network
    trainer : Trainer
        Trainer to apply optimizer on network parameters
    context : Context or list of Context
        devices to run the training on
    """

    def __init__(self, net,
                 loss=None,
                 metrics=None,
                 initializer=None,
                 trainer=None,
                 context=None):

        self.net = net

        if isinstance(loss, gluon.loss.Loss):
            self.loss = [loss]
        else:
            self.loss = loss or []
            for l in self.loss:
                if not isinstance(l, gluon.loss.Loss):
                    raise ValueError("loss must be a Loss or a list of Loss, refer to gluon.loss.Loss")

        if isinstance(metrics, EvalMetric):
            self.metrics = [metrics]
        else:
            self.metrics = metrics or []
            for metric in self.metrics:
                if not isinstance(metric, EvalMetric):
                    raise ValueError("metrics must be a Metric or a list of Metric, refer to mxnet.metric.EvalMetric")

        self.initializer = initializer
        # store training statistics
        self.train_history = TrainHistory()
        self.loss_metrics = []
        # using the metric wrapper for loss to record loss value
        for l in self.loss:
            self.loss_metrics.append(Loss(l.name))


        # handle context
        if isinstance(context, Context):
            self.context = [context]
        if not context:
            if num_gpus() > 0:
                # only use 1 GPU by default
                if num_gpus() > 1:
                    warnings.warn("You have multiple GPUs, gpu(0) will be used by default."
                                  "To utilize all your GPUs, specify context as a list of gpus, "
                                  "e.g. context=[mx.gpu(0), mx.gpu(1)] ")
                self.context = [gpu(0)]
            else:
                self.context = [cpu()]

        # initialize the network
        if self.initializer:
            if self._is_initialized():
                # if already initialized, re-init with user specified initializer
                warnings.warn("Network already initialized, re-initializing with %s. "
                              "You don't need to pass initializer if you already "
                              "initialized your net."% type(self.initializer).__name__)
                self.net.initialize(init=self.initializer, ctx=self.context, force_reinit=True)
            else:
                # initialize with user specified initializer
                self.net.initialize(init=self.initializer, ctx=self.context, force_reinit=False)
        else:
            if not self._is_initialized():
                self.net.initialize(ctx=self.context)

        # handle trainer
        if not trainer:
            warnings.warn("No trainer specified, default SGD optimizer "
                          "with learning rate 0.001 is used.")
            self.trainer = gluon.Trainer(self.net.collect_params(),
                                           'sgd', {'learning_rate': 0.001})
        elif not isinstance(trainer, gluon.Trainer):
            raise ValueError("Trainer must be a Gluon Trainer instance, refer to gluon.trainer")
        else:
            self.trainer = trainer
        self.train_history.optimizer = self.trainer.optimizer


    def _is_initialized(self):
        param_dict = self.net.collect_params()
        for param in param_dict:
            try:
                param_dict[param].list_ctx()
            except RuntimeError:
                return False
        return True

    def _batch_fn(self, batch, ctx, is_iterator=False):
        if is_iterator:
            data = batch.data[0]
            label = batch.label[0]
        else:
            data = batch[0]
            label = batch[1]
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
        return data, label

    def fit(self, train_data,
            epochs=1,
            batch_size=None,
            event_handlers=None,
            batch_fn=None):
        """Main training loop

        Parameters
        ----------
        train_data : DataLoader or DataIter
            training data with data and labels
        val_data : DataLoader or DataIter
            validation data with data and labels
        epochs : int, default 1
            number of epochs to iterate on the training data.
        batch_size : int
            number of samples per gradient update.
            default will be 32 per device
        event_handlers : EventHandler or list of EventHandler
            list of EventHandlers to apply during training
        batch_fn : function
            custom batch function to extract data and label
            from a data batch and load into contexts(devices)
        """


        self.epochs = epochs
        if not batch_size:
            batch_size = 32 * len(self.context)

        self.train_history.max_epoch = epochs
        self.train_history.batch_size = batch_size

        event_handlers = event_handlers or []
        # provide default logging handler
        if not event_handlers or \
                not any(isinstance(handler, LoggingHandler) for handler in event_handlers):
            event_handlers.append(LoggingHandler())

        # training begin
        for handler in event_handlers:
            handler.train_history = self.train_history
            handler.net = self.net
            handler.train_begin()

        for epoch in range(epochs):
            # epoch begin
            self.train_history.epoch = epoch
            self.train_history.learning_rate = self.trainer.learning_rate

            for handler in event_handlers:
                handler.epoch_begin()

            for metric in self.metrics + self.loss_metrics:
                metric.reset()

            for i, batch in enumerate(train_data):
                if not batch_fn:
                    if isinstance(train_data, gluon.data.DataLoader):
                        data, label = self._batch_fn(batch, self.context)
                    elif isinstance(train_data, DataIter):
                        data, label = self._batch_fn(batch, self.context, is_iterator=True)
                    else:
                        raise ValueError("You are using a custom iterator, please also provide "
                                         "batch_fn to extract data and label")
                else:
                    data, label = batch_fn(batch, self.context)

                # batch begin
                for handler in event_handlers:
                    handler.batch_begin()

                with autograd.record():
                    pred = [self.net(x) for x in data]
                    losses = []
                    for loss in self.loss:
                        losses.append([loss(y_hat, y) for y_hat, y in zip(pred, label)])

                for loss in losses:
                    for l in loss:
                        l.backward()

                # update metrics
                for metric in self.metrics:
                    metric.update(label, pred)
                    # get metric name and current value and update train stats
                    self.train_history.set_batch_status(*metric.get())
                for loss, loss_metric, in zip(losses, self.loss_metrics):
                    loss_metric.update(0, [l for l in loss])
                    self.train_history.set_batch_status(*loss_metric.get())

                self.train_history.batch_idx = i
                # record trained samples v.s. total samples if using Gluon DataLoader
                if isinstance(train_data, gluon.data.DataLoader):
                    self.train_history.samples = "{}/{}".format(batch_size * (i + 1), len(train_data._dataset))

                self.trainer.step(batch_size)

                # batch end
                for handler in event_handlers:
                    handler.batch_end()

            for metric in self.metrics + self.loss_metrics:
                self.train_history.set_train_history(*metric.get())
            # epoch end
            for handler in event_handlers:
                handler.epoch_end()

            if self.train_history.stop_training:
                break

        # train end
        for handler in event_handlers:
            handler.train_end()


class TrainHistory(object):
    """TrainHistory class for holding hyper-parameters and training statistics

    """
    def __init__(self):
        self.max_epoch = 200
        self.epochs = []
        self.learning_rates = []
        self.stop_training = False
        self.batch_size = 0
        self.batch_idx = 0
        self.samples = ""
        self.optimizer = None

        # store current training status
        # each key will have only one current value
        self._status = {}
        # store a list of status
        # each key will have a list of values
        self._history = {}
        # store the current loss/metric value
        self._batch_status = {}
        # store list of training loss/metric value over epochs
        self._train_history = {}
        # store list of validation loss/metric value over epochs
        self._val_history = {}



    @property
    def epoch(self):
        return self.epochs[-1] if self.epochs else 0

    @epoch.setter
    def epoch(self, epoch):
        # record epochs as a list
        self.epochs.append(epoch)

    @property
    def learning_rate(self):
        if not self.learning_rates:
            if self.optimizer:
                return self.optimizer.learning_rate
            else:
                raise ValueError("Optimizer has not been initialized yet")
        else:
            return self.learning_rates[self.epoch]


    @learning_rate.setter
    def learning_rate(self, lr):
        # record learning rate history as a list
        self.learning_rates.append(lr)


    def get_status(self, key=None, status=None):
        """Get value from a status dictionary,

        If no key provided, return the entire dictionary
        """
        dict = self._status if not status else status
        if not key:
            return dict
        else:
            return dict.get(key)

    def set_status(self, key, value, status=None):
        """Set value for a status dictionary
        """

        dict = self._status if not status else status
        dict[key] = value

    def get_history(self, key=None, epoch=None, history=None):
        """ Get metric/loss values at certain epoch

        if epoch is None, return values at latest epoch
        if key is None, return a dictionary of all keys and values of the epoch
        """
        dict = self._history if not history else history
        # get the latest epoch
        index = self.epoch if not epoch else epoch

        if not key:
            single_history = {}
            for key in dict:
                if not dict.get(key):
                    # skip this key if no train stats recorded
                    warnings.warn("No stats recorded for %s at epoch %d" % (key, index), RuntimeWarning)
                else:
                    single_history[key] = dict.get(key)[index]
            return single_history
        else:
            if key in dict:
                if not dict.get(key):
                    raise ValueError("No stats recorded for %s at epoch %d" % (key, index))
                else:
                    return dict.get(key)[index]
            else:
                raise ValueError("%s not found in history, please make sure "
                                 "you passed the correct metric/loss name" % key)

    def set_history(self, key, value, history=None):
        dict = self._history if not history else history
        # record a list of stats over the epochs
        dict.setdefault(key, []).append(value)

    def get_batch_status(self, key=None):
        return self.get_status(key, status=self._batch_status)

    def set_batch_status(self, key, value):
        self.set_status(key, value, status=self._batch_status)

    def get_train_history(self, key=None, epoch=None):
        return self.get_history(key, epoch, history=self._train_history)

    def set_train_history(self, key, value):
        self.set_history(key, value, history=self._train_history)

    def get_val_history(self, key=None, epoch=None):
        return self.get_history(key, epoch, history=self._val_history)

    def set_val_history(self, key, value):
        self.set_history(key, value, history=self._val_s_val_history)