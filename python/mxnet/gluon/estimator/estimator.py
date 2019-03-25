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

import copy
import warnings

from .event_handler import LoggingHandler
from ... import gluon, autograd
from ...context import Context, cpu, gpu, num_gpus
from ...io import DataIter
from ...metric import EvalMetric, Loss, Accuracy

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
    trainers : Trainer or list of Trainer
        Trainers to apply optimizers on network parameters
    context : Context or list of Context
        devices to run the training on
    """

    def __init__(self, net,
                 loss=None,
                 metrics=None,
                 initializer=None,
                 trainers=None,
                 context=None):

        self.net = net
        self.stop_training = False

        if isinstance(loss, gluon.loss.Loss):
            self.loss = [loss]
        elif isinstance(loss, list) and all([isinstance(l, gluon.loss.Loss) for l in loss]):
            self.loss = loss
        else:
            raise ValueError("loss must be a Loss or a list of Loss, "
                             "refer to gluon.loss.Loss:{}".format(loss))

        if isinstance(metrics, EvalMetric):
            self.train_metrics = [metrics]
        else:
            self.train_metrics = metrics or []
            if not all([isinstance(metric, EvalMetric) for metric in self.train_metrics]):
                raise ValueError("metrics must be a Metric or a list of Metric, "
                                 "refer to mxnet.metric.EvalMetric:{}".format(metrics))

        # Use default mx.metric.Accuracy() for gluon.loss.SoftmaxCrossEntropyLoss()
        if not self.train_metrics and any([isinstance(l, gluon.loss.SoftmaxCrossEntropyLoss) for l in self.loss]):
            self.train_metrics = [Accuracy()]

        # Use same metrics for validation
        self.val_metrics = copy.deepcopy(self.train_metrics)

        # store training statistics
        self.train_stats = {}
        self.train_stats['epochs'] = []
        self.train_stats['learning_rate'] = []
        # current step of the epoch
        self.train_stats['step'] = ''
        for metric in self.train_metrics:
            # record a history of metrics over each epoch
            self.train_stats['train_' + metric.name] = []
            # only record the latest metric numbers after each batch
            self.train_stats['batch_' + metric.name] = 0.
        for metric in self.val_metrics:
            self.train_stats['val_' + metric.name] = []
        self.train_loss_metrics = []
        self.val_loss_metrics = []
        # using the metric wrapper for loss to record loss value
        for l in self.loss:
            self.train_loss_metrics.append(Loss(l.name))
            self.val_loss_metrics.append(Loss(l.name))
            self.train_stats['train_' + l.name] = []
            self.train_stats['val_' + l.name] = []
            # only record the latest loss numbers after each batch
            self.train_stats['batch_' + l.name] = 0.

        # handle context
        if isinstance(context, Context):
            self.context = [context]
        elif isinstance(context, list) and all([isinstance(c, Context) for c in context]):
            self.context = context
        elif not context:
            if num_gpus() > 0:
                # only use 1 GPU by default
                if num_gpus() > 1:
                    warnings.warn("You have multiple GPUs, gpu(0) will be used by default."
                                  "To utilize all your GPUs, specify context as a list of gpus, "
                                  "e.g. context=[mx.gpu(0), mx.gpu(1)] ")
                self.context = [gpu(0)]
            else:
                self.context = [cpu()]
        else:
            raise ValueError("context must be a Context or a list of Context, "
                             "refer to mxnet.Context:{}".format(context))


        # initialize the network
        self.initializer = initializer
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

        # handle trainers
        if isinstance(trainers, gluon.Trainer):
            self.trainers = [trainers]
        elif not trainers:
            warnings.warn("No trainer specified, default SGD optimizer "
                          "with learning rate 0.001 is used.")
            self.trainers = [gluon.Trainer(self.net.collect_params(),
                                           'sgd', {'learning_rate': 0.001})]
        else:
            raise ValueError("Invalid trainer specified, please provide a valid gluon.Trainer")

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

    def evaluate(self,
                 val_data,
                 batch_fn=None):
        """Evaluate model on validation data

         Parameters
         ----------
         val_data : DataLoader or DataIter
             validation data with data and labels
         batch_fn : function
             custom batch function to extract data and label
             from a data batch and load into contexts(devices)
         """

        for metric in self.val_metrics + self.val_loss_metrics:
            metric.reset()

        for _, batch in enumerate(val_data):
            if not batch_fn:
                if isinstance(val_data, gluon.data.DataLoader):
                    data, label = self._batch_fn(batch, self.context)
                elif isinstance(val_data, DataIter):
                    data, label = self._batch_fn(batch, self.context, is_iterator=True)
                else:
                    raise ValueError("You are using a custom iteration, please also provide "
                                     "batch_fn to extract data and label. Alternatively, you "
                                     "can provide the data as gluon.data.DataLoader or "
                                     "mx.io.DataIter")
            else:
                data, label = batch_fn(batch, self.context)
            pred = [self.net(x) for x in data]
            losses = []
            for loss in self.loss:
                losses.append([loss(y_hat, y) for y_hat, y in zip(pred, label)])
            # update metrics
            for metric in self.val_metrics:
                metric.update(label, pred)
            for loss, loss_metric, in zip(losses, self.val_loss_metrics):
                loss_metric.update(0, [l for l in loss])

    def fit(self, train_data,
            val_data=None,
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

        event_handlers = event_handlers or []
        # provide default logging handler
        if not event_handlers or \
                not any(isinstance(handler, LoggingHandler) for handler in event_handlers):
            event_handlers.append(LoggingHandler(self))

        # training begin
        for handler in event_handlers:
            handler.train_begin()

        for epoch in range(epochs):
            # epoch begin
            self.train_stats['epochs'].append(epoch)
            self.train_stats['learning_rate'].append(self.trainers[0].learning_rate)

            for handler in event_handlers:
                handler.epoch_begin()

            for metric in self.train_metrics + self.train_loss_metrics:
                metric.reset()

            for i, batch in enumerate(train_data):
                if not batch_fn:
                    if isinstance(train_data, gluon.data.DataLoader):
                        data, label = self._batch_fn(batch, self.context)
                    elif isinstance(train_data, DataIter):
                        data, label = self._batch_fn(batch, self.context, is_iterator=True)
                    else:
                        raise ValueError("You are using a custom iteration, please also provide "
                                         "batch_fn to extract data and label. Alternatively, you "
                                         "can provide the data as gluon.data.DataLoader or "
                                         "mx.io.DataIter")
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

                # update train metrics
                for metric in self.train_metrics:
                    metric.update(label, pred)
                    self.train_stats['batch_' + metric.name] = metric.get()[1]
                for loss, loss_metric, in zip(losses, self.train_loss_metrics):
                    loss_metric.update(0, [l for l in loss])
                    self.train_stats['batch_' + loss_metric.name] = loss_metric.get()[1]

                try:
                    completed_samples = len(train_data._dataset) if i == len(train_data._dataset) - 1 \
                                        else batch_size * (i + 1)
                    # We need to check if this is the last batch in the current epoch and select
                    # the value to print appropriately
                    self.train_stats['step'] = "{}/{}".format(completed_samples, len(train_data._dataset))
                except AttributeError:
                    self.train_stats['step'] = i

                for trainer in self.trainers:
                    trainer.step(batch_size)

                # batch end
                for handler in event_handlers:
                    handler.batch_end()

            if val_data:
                self.evaluate(val_data, batch_fn)

            for metric in self.train_metrics + self.train_loss_metrics:
                self.train_stats['train_' + metric.name].append(metric.get()[1])
            for metric in self.val_metrics + self.val_loss_metrics:
                self.train_stats['val_' + metric.name].append(metric.get()[1])

            # epoch end
            for handler in event_handlers:
                handler.epoch_end()

            if self.stop_training:
                break

        # train end
        for handler in event_handlers:
            handler.train_end()
