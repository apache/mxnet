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

from .event_handler import MetricHandler, ValidationHandler, LoggingHandler, StoppingHandler
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
    net : Block
        The model used for training.
    loss : gluon.loss.Loss or list of gluon.loss.Loss
        Loss(objective functions) to calculate during training.
    metrics : EvalMetric or list of EvalMetric
        Metrics for evaluating models.
    initializer : Initializer
        Initializer to initialize the network.
    trainer : Trainer
        Trainer to apply optimizer on network parameters.
    context : Context or list of Context
        Device(s) to run the training on.
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
        elif isinstance(loss, list) and all([isinstance(l, gluon.loss.Loss) for l in loss]):
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
        # infer available context
        gpus = num_gpus()
        available_gpus = [gpu(i) for i in range(gpus)]

        if context:
            # check context values, only accept Context or a list of Context
            if isinstance(context, Context):
                context = [context]
            elif isinstance(context, list) and all([isinstance(c, Context) for c in context]):
                context = context
            else:
                raise ValueError("context must be a Context or a list of Context, "
                                 "for example mx.cpu() or [mx.gpu(0), mx.gpu(1)], "
                                 "refer to mxnet.Context:{}".format(context))
            for ctx in context:
                assert ctx in available_gpus or str(ctx).startswith('cpu'), \
                    "%s is not available, please make sure " \
                    "your context is in one of: mx.cpu(), %s" % \
                    (ctx, ", ".join([str(ctx) for ctx in available_gpus]))
        else:
            # provide default context
            if gpus > 0:
                # only use 1 GPU by default
                if gpus > 1:
                    warnings.warn("You have multiple GPUs, gpu(0) will be used by default."
                                  "To utilize all your GPUs, specify context as a list of gpus, "
                                  "e.g. context=[mx.gpu(0), mx.gpu(1)] ")
                context = [gpu(0)]
            else:
                context = [cpu()]
        return context

    def _initialize(self, initializer):
        # initialize the network
        if not self._is_initialized():
            # net is partially or not initialized,
            # initialize with user specified initializer
            # if initializer is None, default initializer will be used
            # do not re-init layers already initialized
            if initializer:
                self.net.initialize(init=initializer, ctx=self.context)
            else:
                self.net.initialize(ctx=self.context)
        elif initializer:
            # net is fully initialized, and user passed not None initializer
            # do not force reinitialize, give warning
            warnings.warn("Network already fully initialized, skipping initialization. "
                          "You don't need to pass initializer if you already "
                          "initialized your net. "
                          "You can use net.initialize(init=your_initializer, force_reinit=True)"
                          "to force re-initialize.")

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

    def _get_data_and_label(self, batch, ctx, batch_axis=0):
        data = batch[0]
        label = batch[1]
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=batch_axis)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=batch_axis)
        return data, label

    def prepare_loss_and_metrics(self):
        """
        Based on loss functions and training metrics in estimator
        Create metric wrappers to record loss values,
        Create copies of train loss/metric objects to record validation values
        Returns train_metrics and val_metrics

        """
        if any(not hasattr(self, attribute) for attribute in
               ['train_metrics', 'val_metrics']):
            # Use default mx.metric.Accuracy() for gluon.loss.SoftmaxCrossEntropyLoss()
            if not self.train_metrics and any([isinstance(l, gluon.loss.SoftmaxCrossEntropyLoss) for l in self.loss]):
                self.train_metrics = [Accuracy()]
            self.val_metrics = []
            for loss in self.loss:
                # remove trailing numbers from loss name to avoid confusion
                self.train_metrics.append(Loss(loss.name.rstrip('1234567890')))
            for metric in self.train_metrics:
                val_metric = copy.deepcopy(metric)
                metric.name = "train " + metric.name
                val_metric.name = "validation " + val_metric.name
                self.val_metrics.append(val_metric)
        return self.train_metrics, self.val_metrics

    def evaluate(self,
                 val_data,
                 val_metrics,
                 batch_axis=0):
        """Evaluate model on validation data

         Parameters
         ----------
         val_data : DataLoader
             Validation data loader with data and labels.
         val_metrics : EvalMetric or list of EvalMetrics
             Metrics to update validation result.
         batch_axis : int, default 0
             Batch axis to split the validation data into devices.
         """
        if not isinstance(val_data, gluon.data.DataLoader):
            raise ValueError("Estimator only support input as Gluon DataLoader. Alternatively, you "
                             "can transform your DataIter or any NDArray into Gluon DataLoader. "
                             "Refer to gluon.data.dataloader")

        for metric in val_metrics:
            metric.reset()

        for _, batch in enumerate(val_data):
            data, label = self._get_data_and_label(batch, self.context, batch_axis)
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
            epochs=None,
            event_handlers=None,
            batches=None,
            batch_axis=0):
        """Trains the model with a given :py:class:`DataLoader` for a specified
        number of epochs or batches. The batch size is inferred from the
        data loader's batch_size.

        Parameters
        ----------
        train_data : DataLoader
            Training data loader with data and labels.
        val_data : DataLoader, default None
            Validation data loader with data and labels.
        epochs : int, default None
            Number of epochs to iterate on the training data.
            You can only specify one and only one type of iteration(epochs or batches).
        event_handlers : EventHandler or list of EventHandler
            List of :py:class:`EventHandlers` to apply during training.
        batches : int, default None
            Number of batches to iterate on the training data.
            You can only specify one and only one type of iteration(epochs or batches).
        batch_axis : int, default 0
            Batch axis to split the training data into devices.
        """
        if not isinstance(train_data, gluon.data.DataLoader):
            raise ValueError("Estimator only support input as Gluon DataLoader. Alternatively, you "
                             "can transform your DataIter or any NDArray into Gluon DataLoader. "
                             "Refer to gluon.data.dataloader")

        # must specify one and only one of epochs or batches
        if (not epochs) == (not batches):
            raise ValueError(
                "Fit only support exactly one type of iteration, "
                "train by number of epochs or number of batches."
                "Please specify one and only one of: epochs or batches.")

        self.max_epoch = epochs
        self.max_batch = batches

        # provide default handlers
        event_handlers = self._prepare_default_handlers(val_data, event_handlers)

        train_begin, epoch_begin, batch_begin, \
        batch_end, epoch_end, train_end = self._categorize_handlers(event_handlers)

        # pass a reference to all event handlers
        estimator_ref = self
        # training begin
        for handler in train_begin:
            handler.train_begin(estimator_ref)

        while True:
            # epoch begin
            for handler in epoch_begin:
                handler.epoch_begin(estimator_ref)

            for i, batch in enumerate(train_data):
                data, label = self._get_data_and_label(batch, self.context, batch_axis)

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

                batch_end_result = []
                for handler in batch_end:
                    batch_end_result.append(handler.batch_end(estimator_ref, batch=batch,
                                                              pred=pred, label=label, loss=loss))
                # if any handler signaled to stop
                if any(batch_end_result):
                    break

            # epoch end
            epoch_end_result = []
            for handler in epoch_end:
                epoch_end_result.append(handler.epoch_end(estimator_ref))
            # if any handler signaled to stop
            if any(epoch_end_result):
                break

        # train end
        for handler in train_end:
            handler.train_end(estimator_ref)

    def _prepare_default_handlers(self, val_data, event_handlers):
        event_handlers = event_handlers or []
        default_handlers = []
        train_metrics, val_metrics = self.prepare_loss_and_metrics()

        # no need to add to default handler check as StoppingHandler does not use metrics
        event_handlers.append(StoppingHandler(self.max_epoch, self.max_batch))

        if not any(isinstance(handler, MetricHandler) for handler in event_handlers):
            event_handlers.append(MetricHandler(train_metrics=train_metrics))
            default_handlers.append("MetricHandler")

        if val_data and not any(isinstance(handler, ValidationHandler) for handler in event_handlers):
            event_handlers.append(ValidationHandler(val_data=val_data, eval_fn=self.evaluate,
                                                    val_metrics=val_metrics))
            default_handlers.append("ValidationHandler")

        if not any(isinstance(handler, LoggingHandler) for handler in event_handlers):
            event_handlers.append(LoggingHandler(train_metrics=train_metrics,
                                                 val_metrics=val_metrics))
            default_handlers.append("LoggingHandler")

        # if there is a mix of user defined event handlers and default event handlers
        # they should have the same set of loss and metrics
        if default_handlers:
            msg = "You are training with the following default event handlers: %s. " \
                  "They use loss and metrics from estimator.prepare_loss_and_metrics(). " \
                  "Please use the same set of metrics for all your other handlers." % \
                  ", ".join(default_handlers)
            warnings.warn(msg)
            # check if all handlers has the same set of references to loss and metrics
            references = []
            for handler in event_handlers:
                for attribute in dir(handler):
                    if any(keyword in attribute for keyword in ['metric' or 'monitor']):
                        reference = getattr(handler, attribute)
                        if isinstance(reference, list):
                            references += reference
                        else:
                            references.append(reference)
            # remove None metric references
            references = set([ref for ref in references if ref])
            for metric in references:
                if metric not in train_metrics + val_metrics:
                    msg = "We have added following default handlers for you: %s and used " \
                          "estimator.prepare_loss_and_metrics() to pass metrics to " \
                          "those handlers. Please use the same set of metrics " \
                          "for all your handlers." % \
                          ", ".join(default_handlers)
                    raise ValueError(msg)

        event_handlers.sort(key=lambda handler: getattr(handler, 'priority', 0))
        return event_handlers

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
