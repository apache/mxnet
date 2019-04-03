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

import numpy as np

from .event_handler import EventHandler, LoggingHandler
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
    loss : gluon.loss.Loss or list of gluon.loss.Loss
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
                 loss,
                 metrics=None,
                 initializer=None,
                 trainer=None,
                 context=None):

        self.net = net

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

        # separate train and validation
        self.train_loss_metrics = []
        self.val_loss_metrics = []
        # using the metric wrapper for loss to record loss value
        for l in self.loss:
            self.train_loss_metrics.append(Loss(l.name))
            self.val_loss_metrics.append(Loss(l.name))

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
                              "initialized your net." % type(self.initializer).__name__)
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
            raise ValueError("Trainer must be a Gluon Trainer instance, refer to "
                             "gluon.Trainer:{}".format(trainer))
        else:
            self.trainer = trainer

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
                name, value = metric.get()
                self.train_stats['val_' + name] = value
            for loss, loss_metric, in zip(losses, self.val_loss_metrics):
                loss_metric.update(0, [l for l in loss])
                name, value = loss_metric.get()
                self.train_stats['val_' + name] = value

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

        self.max_epoch = epochs
        if not batch_size:
            self.batch_size = 32 * len(self.context)
        else:
            self.batch_size = batch_size
        self.stop_training = False
        self.samples = None
        self.batch_idx = 0

        event_handlers = event_handlers or []
        # provide default logging handler
        if not event_handlers or \
                not any(isinstance(handler, LoggingHandler) for handler in event_handlers):
            event_handlers.append(LoggingHandler())

        train_begin, epoch_begin, batch_begin, \
        batch_end, epoch_end, train_end = self._categorize_handlers(event_handlers)

        # passing estimator to event handlers so they can access estimator information
        # when a event is triggered
        for handler in event_handlers:
            handler.estimator = self

        # training begin
        for handler in train_begin:
            handler.train_begin()

        num_labels = self.infer_num_labels(self.loss)
        for i, batch in enumerate(train_data):
            first_batch = batch
            break
        num_inputs = self.infer_num_inputs(first_batch, num_labels)
        fit_helper = FitHelper(num_inputs, num_labels)

        for epoch in range(self.max_epoch):
            # epoch begin
            self.current_epoch = epoch

            for handler in epoch_begin:
                handler.epoch_begin()

            for metric in self.train_metrics + self.train_loss_metrics:
                metric.reset()

            for i, batch in enumerate(train_data):

                if not batch_fn:
                    if isinstance(train_data, gluon.data.DataLoader):
                        data, label = fit_helper.get_data_label(batch, self.context)
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
                for handler in batch_begin:
                    handler.batch_begin()

                with autograd.record():
                    pred =  fit_helper.forward_pass(self.net, data)
                    losses = fit_helper.calculate_loss(self.loss, pred, label)

                # backward loss per device
                for loss in losses:
                    loss.backward()

                fit_helper.update_metrics(self.train_metrics, pred, label, self.train_stats, 'train')

                # update loss
                for loss, loss_metric, in zip(losses, self.train_loss_metrics):
                    loss_metric.update(0, [l for l in loss])
                    name, value = loss_metric.get()
                    self.train_stats['train_' + name] = value

                self.batch_idx = i
                # record trained samples v.s. total samples if using Gluon DataLoader
                if isinstance(train_data, gluon.data.DataLoader):
                    self.samples = "{}/{}".format(self.batch_size * (i + 1), len(train_data._dataset))

                self.trainer.step(self.batch_size)
                # batch end
                for handler in batch_end:
                    handler.batch_end()

            if val_data:
                self.evaluate(val_data, batch_fn)

            # epoch end
            for handler in epoch_end:
                handler.epoch_end()

            if self.stop_training:
                break

        # train end
        for handler in train_end:
            handler.train_end()

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
            if not handler.__class__.train_begin == EventHandler.train_begin:
                train_begin.append(handler)
            if not handler.__class__.epoch_begin == EventHandler.epoch_begin:
                epoch_begin.append(handler)
            if not handler.__class__.batch_begin == EventHandler.batch_begin:
                batch_begin.append(handler)
            if not handler.__class__.batch_end == EventHandler.batch_end:
                batch_end.append(handler)
            if not handler.__class__.epoch_end == EventHandler.epoch_end:
                epoch_end.append(handler)
            if not handler.__class__.train_end == EventHandler.train_end:
                train_end.append(handler)
        return train_begin, epoch_begin, batch_begin, batch_end, epoch_end, train_end


    def infer_num_labels(self, loss):
        return len(loss)

    def infer_num_inputs(self, batch, num_labels):
        return len(batch) - num_labels

class FitHelper(object):
    def __init__(self, num_inputs, num_labels):
        self.num_inputs = num_inputs
        self.num_labels = num_labels

    def get_data_label(self, batch, context):
        if self.num_inputs > 1:
            data = self._get_multi_input(batch, self.num_inputs, context)
        else:
            data = self._get_single_input(batch, context)
        if self.num_labels > 1:
            label = self._get_multi_label(batch, self.num_inputs, self.num_labels, context)
        else:
            label = self._get_single_label(batch, self.num_inputs, context)
        return data, label

    def forward_pass(self, net, data):
        if self.num_inputs > 1:
            return self._forward_pass_multi_input(net, data)
        else:
            return self._forward_pass_single_input(net, data)

    def calculate_loss(self, loss, preds, labels, loss_weights=None):
        if self.num_labels > 1:
            return self._calculate_loss_multi_label(loss, loss_weights, preds, labels)
        else:
            return self._calculate_loss_single_label(loss, preds, labels)

    def update_metrics(self, metrics, preds, labels, train_stats, prefix):
        if self.num_labels > 1:
            self._update_metric_multi_label(self, metrics, preds, labels, train_stats, prefix)
        else:
            self._update_metric_single_label(metrics, preds, labels, train_stats, prefix)


    def _get_single_input(self, batch, ctx_list):
        return gluon.utils.split_and_load(batch[0], ctx_list=ctx_list, batch_axis=0)

    def _get_single_label(self, batch, num_inputs, ctx_list):
        return gluon.utils.split_and_load(batch[num_inputs], ctx_list=ctx_list, batch_axis=0)

    def _get_multi_input(self, batch, num_inputs, ctx_list):
        inputs = []
        for i in range(num_inputs):
            inputs.append(gluon.utils.split_and_load(batch[i], ctx_list=ctx_list, batch_axis=0))
        # convert inputs from inputs grouped to context grouped
        # before: [[input1_context1, input1_context2], [input2_context1, input2_context2]]
        # after:  [[input1_context1, input2_context1], [input1_context2, input2_context2]]
        return np.transpose(inputs)

    def _get_multi_label(self, batch, num_inputs, num_labels, ctx_list):
        labels = []
        for i in range(num_inputs, num_inputs + num_labels):
            labels.append(gluon.utils.split_and_load(batch[i], ctx_list=ctx_list, batch_axis=0))
        # convert labels from inputs grouped to context grouped
        return np.transpose(labels)

    def _forward_pass_single_input(self, net, data):
        return [net(x) for x in data]

    def _forward_pass_multi_input(self, net, data):
        return [net(*x) for x in data]

    def _calculate_loss_single_label(self, loss, preds, labels):
        return [loss[0](pred, label) for pred, label in zip(preds, labels)]

    def _calculate_loss_multi_label(self, loss, loss_weights, preds, labels):
        combined_loss_all_device = []
        # calculate combined loss per device
        for pred, label in preds, labels:
            combined_loss = 0
            for idx in range(len(pred)):
                combined_loss = combined_loss + loss_weights[idx] * loss[idx](pred[idx], label[idx])
            combined_loss_all_device.append(combined_loss)
        return combined_loss_all_device

    def _update_metric_single_label(self, metrics, preds, labels, train_stats, prefix):
        for metric in metrics:
            metric.update(labels, preds)
            name, value = metric.get()
            train_stats['%s_%s' % (prefix, name)] = value

    def _update_metric_multi_label(self, metrics, preds, labels, train_stats, prefix):
        pass
