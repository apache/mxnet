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
# pylint: disable=wildcard-import, unused-variable
"""Gluon Estimator"""

import copy
import warnings
import weakref

from .event_handler import MetricHandler, ValidationHandler, LoggingHandler
from .event_handler import TrainBegin, EpochBegin, BatchBegin, BatchEnd, EpochEnd, TrainEnd
from .... import gluon, autograd
from ....context import Context, cpu, gpu, num_gpus
from ....metric import EvalMetric, Loss, Accuracy

__all__ = ['Estimator']


class Estimator(object):
    """Estimator Class for easy model training

    :py:class:`Estimator` can be used to facilitate the training & validation process


    Parameters
    ----------
    loss : gluon.loss.Loss or list of gluon.loss.Loss
        Loss(objective functions) to calculate during training
    metrics : EvalMetric or list of EvalMetric
        Metrics for evaluating models
    initializer : Initializer
        initializer to initialize the network
    trainer : Trainer
        Trainer to apply optimizer on network parameters
    context : Context or list of Context
        device(s) to run the training on
    """

    def __init__(self, net,
                 loss,
                 metrics=None,
                 initializer=None,
                 trainer=None,
                 context=None):

        self.net = net
        self.loss = self._check_loss(loss)
        self.train_metrics = self._check_metrics(metrics)

        self.context = self._check_context(context)
        self._initialize(initializer)
        self.trainer = self._check_trainer(trainer)

    def _check_loss(self, loss):
        if isinstance(loss, gluon.loss.Loss):
            loss = [loss]
        elif isinstance(loss, list) or all([isinstance(l, gluon.loss.Loss) for l in loss]):
            loss = loss
        else:
            raise ValueError("loss must be a Loss or a list of Loss, "
                             "refer to gluon.loss.Loss:{}".format(loss))
        return loss

    def _check_metrics(self, metrics):
        if isinstance(metrics, EvalMetric):
            metrics = [metrics]
        else:
            metrics = metrics or []
            if not all([isinstance(metric, EvalMetric) for metric in metrics]):
                raise ValueError("metrics must be a Metric or a list of Metric, "
                                 "refer to mxnet.metric.EvalMetric:{}".format(metrics))
        return metrics

    def _check_context(self, context):
        # handle context
        if isinstance(context, Context):
            context = [context]
        elif isinstance(context, list) and all([isinstance(c, Context) for c in context]):
            context = context
        elif not context:
            if num_gpus() > 0:
                # only use 1 GPU by default
                if num_gpus() > 1:
                    warnings.warn("You have multiple GPUs, gpu(0) will be used by default."
                                  "To utilize all your GPUs, specify context as a list of gpus, "
                                  "e.g. context=[mx.gpu(0), mx.gpu(1)] ")
                context = [gpu(0)]
            else:
                context = [cpu()]
        else:
            raise ValueError("context must be a Context or a list of Context, "
                             "refer to mxnet.Context:{}".format(context))
        return context

    def _initialize(self, initializer):
        # initialize the network
        if initializer:
            if self._is_initialized():
                # if already initialized, re-init with user specified initializer
                warnings.warn("Network already initialized, re-initializing with %s. "
                              "You don't need to pass initializer if you already "
                              "initialized your net." % type(initializer).__name__)
                self.net.initialize(init=initializer, ctx=self.context, force_reinit=True)
            else:
                # initialize with user specified initializer
                self.net.initialize(init=initializer, ctx=self.context, force_reinit=False)
        else:
            if not self._is_initialized():
                self.net.initialize(ctx=self.context)

    def _check_trainer(self, trainer):
        # handle trainer
        if not trainer:
            warnings.warn("No trainer specified, default SGD optimizer "
                          "with learning rate 0.001 is used.")
            trainer = gluon.Trainer(self.net.collect_params(),
                                    'sgd', {'learning_rate': 0.001})
        elif not isinstance(trainer, gluon.Trainer):
            raise ValueError("Trainer must be a Gluon Trainer instance, refer to "
                             "gluon.Trainer:{}".format(trainer))
        return trainer

    def _is_initialized(self):
        param_dict = self.net.collect_params()
        for param in param_dict:
            try:
                param_dict[param].list_ctx()
            except RuntimeError:
                return False
        return True

    def _get_data_and_label(self, batch, ctx):
        data = batch[0]
        label = batch[1]
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
        return data, label

    def prepare_loss_and_metrics(self):
        """
        Based on loss functions and training metrics in estimator
        Create metric wrappers to record loss values,
        Create copies of train loss/metric objects to record validation values
        """
        if any(not hasattr(self, attribute) for attribute in
               ['train_metrics', 'val_metrics']):
            # Use default mx.metric.Accuracy() for gluon.loss.SoftmaxCrossEntropyLoss()
            if not self.train_metrics and any([isinstance(l, gluon.loss.SoftmaxCrossEntropyLoss) for l in self.loss]):
                self.train_metrics = [Accuracy()]
            self.val_metrics = []
            for loss in self.loss:
                self.train_metrics.append(Loss("Train " + ''.join([i for i in loss.name if not i.isdigit()])))
                self.val_metrics.append(Loss("Validation " + ''.join([i for i in loss.name if not i.isdigit()])))
            for metric in self.train_metrics:
                val_metric = copy.deepcopy(metric)
                metric.name = "Train " + metric.name
                val_metric.name = "Validation " + val_metric.name
                self.val_metrics.append(val_metric)
        return self.train_metrics, self.val_metrics

    def evaluate(self,
                 val_data,
                 val_metrics):
        """Evaluate model on validation data

         Parameters
         ----------
         val_data : DataLoader
             validation data with data and labels
         val_metrics : EvalMetric or list of EvalMetrics
             metrics to update validation result
         """

        for metric in val_metrics:
            metric.reset()

        for _, batch in enumerate(val_data):
            if not isinstance(val_data, gluon.data.DataLoader):
                raise ValueError("Estimator only support input as Gluon DataLoader. Alternatively, you "
                                 "can transform your DataIter or any NDArray into Gluon DataLoader. "
                                 "Refer to gluon.data.dataloader")
            data, label = self._get_data_and_label(batch, self.context)
            pred = [self.net(x) for x in data]
            loss = [self.loss[0](y_hat, y) for y_hat, y in zip(pred, label)]
            # update metrics
            for metric in val_metrics:
                if isinstance(metric, Loss):
                    metric.update(0, loss)
                else:
                    metric.update(label, pred)

    def fit(self, train_data,
            val_data=None,
            epochs=1,
            event_handlers=None):
        """Trains the model on a given dataset for a specified
        number of epochs. Also, the batch size is inferred from the
        DataLoader's batch_size.

        Parameters
        ----------
        train_data : DataLoader
            training data with data and labels
        val_data : DataLoader
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
        self.max_epochs = epochs
        event_handlers = event_handlers or []
        # provide default logging handler
        if not event_handlers:
            train_metrics, val_metrics = self.prepare_loss_and_metrics()
            event_handlers.append(MetricHandler(train_metrics=train_metrics))
            if val_data:
                event_handlers.append(ValidationHandler(val_data=val_data, eval_fn=self.evaluate,
                                                        val_metrics=val_metrics))
            event_handlers.append(LoggingHandler(train_metrics=train_metrics,
                                                 val_metrics=val_metrics))
            warnings.warn("No Event Handler specified, default %s are used. "
                          "Please look at gluon.contrib.estimator.event_handler for more detail." %
                          ", ".join([handler.__class__.__name__ for handler in event_handlers]))

        event_handlers.sort(key=lambda handler: getattr(handler, 'rank', 0), reverse=True)

        train_begin, epoch_begin, batch_begin, \
        batch_end, epoch_end, train_end = self._categorize_handlers(event_handlers)

        # only pass a weak reference to all event handlers
        estimator_ref = weakref.proxy(self)
        # training begin
        for handler in train_begin:
            handler.train_begin(estimator_ref)

        for epoch in range(epochs):
            # epoch begin
            for handler in epoch_begin:
                handler.epoch_begin(estimator_ref)

            for i, batch in enumerate(train_data):
                if not isinstance(train_data, gluon.data.DataLoader):
                    raise ValueError("Estimator only support input as Gluon DataLoader. Alternatively, you "
                                     "can transform your DataIter or any NDArray into Gluon DataLoader. "
                                     "Refer to gluon.data.dataloader")
                data, label = self._get_data_and_label(batch, self.context)

                batch_size = batch[0].shape[0]

                # batch begin
                for handler in batch_begin:
                    handler.batch_begin(estimator_ref, batch=batch)

                with autograd.record():
                    pred = [self.net(x) for x in data]
                    loss = [self.loss[0](y_hat, y) for y_hat, y in zip(pred, label)]

                for l in loss:
                    l.backward()

                self.trainer.step(batch_size)
                # batch end
                for handler in batch_end:
                    if handler.batch_end(estimator_ref, batch=batch,
                                         pred=pred, label=label, loss=loss):
                        break

            # epoch end
            for handler in epoch_end:
                if handler.epoch_end(estimator_ref):
                    break

        # train end
        for handler in train_end:
            handler.train_end(estimator_ref)

    def _categorize_handlers(self, event_handlers):
        """
        categorize handlers into 6 event lists to avoid calling empty methods
        for example, only event handlers with train_begin method
        implemented will be called at train begin
        """

        train_begin = []
        epoch_begin = []
        batch_begin = []
        batch_end = []
        epoch_end = []
        train_end = []
        for handler in event_handlers:
            if isinstance(handler, TrainBegin):
                train_begin.append(handler)
            if isinstance(handler, EpochBegin):
                epoch_begin.append(handler)
            if isinstance(handler, BatchBegin):
                batch_begin.append(handler)
            if isinstance(handler, BatchEnd):
                batch_end.append(handler)
            if isinstance(handler, EpochEnd):
                epoch_end.append(handler)
            if isinstance(handler, TrainEnd):
                train_end.append(handler)
        return train_begin, epoch_begin, batch_begin, batch_end, epoch_end, train_end
