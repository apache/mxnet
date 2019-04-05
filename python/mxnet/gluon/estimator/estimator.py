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
                 loss_weights=None,
                 metrics=None,
                 initializer=None,
                 trainer=None,
                 context=None):

        self.net = net
        self.loss, self.loss_weights = self._check_loss(loss, loss_weights)
        self.train_metrics = self._check_metrics(metrics)
        self.context = self._check_context(context)
        self._initialize(initializer)
        self.trainer = self._check_trainer(trainer)

        # record training statistics
        self.train_stats = {}
        # separate train and validation metrics and loss
        self.val_metrics = []
        self.train_loss = []
        self.val_loss = []

        # using the metric wrapper for loss to record loss value
        for l in self.loss:
            self.train_loss.append(Loss(l.name))

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

        for metric in self.val_metrics + self.val_loss:
            metric.reset()
        num_labels = self._infer_num_labels(self.loss)
        num_inputs = self._infer_num_inputs(val_data, num_labels)
        fit_helper = FitHelper(num_inputs, num_labels)
        for _, batch in enumerate(val_data):
            data, label = fit_helper.get_data_label(batch, self.context)
            pred = fit_helper.forward_pass(self.net, data)
            fit_helper.calculate_loss(self.loss, pred, label, self.loss_weights, self.val_loss)
            fit_helper.update_metrics(self.val_metrics, pred, label)
            self._update_trains_stats(self.val_metrics + self.val_loss, 'val')

    def fit(self, train_data,
            epochs=1,
            event_handlers=None):
        """Main training loop

        Parameters
        ----------
        train_data : DataLoader or DataIter
            training data with data and labels
        epochs : int, default 1
            number of epochs to iterate on the training data.
        event_handlers : EventHandler or list of EventHandler
            list of EventHandlers to apply during training
        """

        self.max_epoch = epochs
        self.stop_training = False
        self.samples = None
        self.batch_idx = 0
        self.total_steps = 0

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

        num_labels = self._infer_num_labels(self.loss)
        num_inputs = self._infer_num_inputs(train_data, num_labels)
        fit_helper = FitHelper(num_inputs, num_labels)

        for epoch in range(self.max_epoch):
            # epoch begin
            self.current_epoch = epoch

            for handler in epoch_begin:
                handler.epoch_begin()

            for metric in self.train_metrics + self.train_loss:
                metric.reset()

            for i, batch in enumerate(train_data):

                data, label = fit_helper.get_data_label(batch, self.context)

                # batch begin
                for handler in batch_begin:
                    handler.batch_begin()

                with autograd.record():
                    pred =  fit_helper.forward_pass(self.net, data)
                    losses = fit_helper.calculate_loss(self.loss, pred, label, self.loss_weights, self.train_loss)

                # backward loss per device
                for loss in losses:
                    loss.backward()

                fit_helper.update_metrics(self.train_metrics, pred, label)

                self.batch_idx = i
                # record trained samples v.s. total samples if using Gluon DataLoader
                if isinstance(train_data, gluon.data.DataLoader):
                    self.samples = "{}/{}".format(batch[0].shape[0] * (i + 1), len(train_data._dataset))

                self.trainer.step(batch[0].shape[0])
                self.total_steps += 1
                self._update_trains_stats(self.train_loss + self.train_metrics, 'train')
                # batch end
                for handler in batch_end:
                    handler.batch_end()

            # epoch end
            for handler in epoch_end:
                handler.epoch_end()

            if self.stop_training:
                break

        # train end
        for handler in train_end:
            handler.train_end()

    def list_loss_and_metrics(self):
        # print and return all loss and metrics been recorded
        print("Available loss and metrics:")
        for name in self.train_stats:
            print(name)

    def _check_loss(self, loss, loss_weights):
        if isinstance(loss, gluon.loss.Loss):
            loss = [loss]
        elif isinstance(loss, list) and not all([isinstance(l, gluon.loss.Loss) for l in loss]):
            raise ValueError("loss must be a Loss or a list of Loss, "
                             "refer to gluon.loss.Loss:{}".format(loss))

        if len(loss) > 1:
            if not loss_weights:
                loss_weights = [1.0 / len(loss) for _ in loss]
            if not isinstance(loss_weights, list):
                raise ValueError("Please provide loss weights as a list to match the number of loss")
            if len(loss) != len(loss_weights):
                raise ValueError("Number of loss weights must match number of loss")
        return loss, loss_weights

    def _check_metrics(self, metrics):
        if isinstance(metrics, EvalMetric):
            metrics= [metrics]
        elif not metrics:
            # Use default mx.metric.Accuracy() for gluon.loss.SoftmaxCrossEntropyLoss()
            if any([isinstance(l, gluon.loss.SoftmaxCrossEntropyLoss) for l in self.loss]):
                metrics = [Accuracy()]
            else:
                # allow empty list of metrics
                metrics = []
        if not all([isinstance(metric, EvalMetric) for metric in metrics]):
                raise ValueError("metrics must be a Metric or a list of Metric, "
                                 "refer to mxnet.metric.EvalMetric:{}".format(metrics))
        return metrics

    def _check_context(self, context):
        # handle context
        if isinstance(context, Context):
            context = [context]
        elif isinstance(context, list) and not all([isinstance(c, Context) for c in context]):
            raise ValueError("context must be a Context or a list of Context, "
                             "refer to mxnet.Context:{}".format(context))
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

    def _infer_num_labels(self, loss):
        return len(loss)

    def _infer_num_inputs(self, train_data, num_labels):
        for i, batch in enumerate(train_data):
            first_batch = batch
            break
        return len(first_batch) - num_labels

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

    def _update_trains_stats(self, metrics, prefix):
        for metric in metrics:
            name, value = metric.get()
            self.train_stats["%s_%s" % (prefix, name)] = value

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

    def calculate_loss(self, loss, preds, labels, loss_weights, loss_metrics):
        if self.num_labels > 1:
            return self._calculate_loss_multi_label(loss, loss_weights, preds, labels, loss_metrics)
        else:
            return self._calculate_loss_single_label(loss[0], preds, labels, loss_metrics[0])

    def update_metrics(self, metrics, preds, labels):
        if self.num_labels > 1:
            self._update_metric_multi_label(metrics, preds, labels)
        else:
            self._update_metric_single_label(metrics, preds, labels)

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
        return list(map(list, zip(*inputs)))

    def _get_multi_label(self, batch, num_inputs, num_labels, ctx_list):
        labels = []
        for i in range(num_inputs, num_inputs + num_labels):
            labels.append(gluon.utils.split_and_load(batch[i], ctx_list=ctx_list, batch_axis=0))
        # convert labels from inputs grouped to context grouped
        return list(map(list, zip(*labels)))

    def _forward_pass_single_input(self, net, data):
        return [net(x) for x in data]

    def _forward_pass_multi_input(self, net, data):
        return [net(*x) for x in data]

    def _calculate_loss_single_label(self, loss, preds, labels, loss_metric):
        loss_all_device = [loss(pred, label) for pred, label in zip(preds, labels)]
        loss_metric.update(0, loss_all_device)
        return loss_all_device

    def _calculate_loss_multi_label(self, loss, loss_weights, preds, labels, loss_metrics):
        combined_loss_all_device = []
        # calculate combined loss per device
        for pred, label in zip(preds, labels):
            combined_loss = 0
            for idx in range(self.num_labels):
                single_loss = loss[idx](pred[idx], label[idx])
                combined_loss = combined_loss + loss_weights[idx] * single_loss
                loss_metrics[idx].update(0, single_loss)
            combined_loss_all_device.append(combined_loss)
        return combined_loss_all_device

    def _update_metric_single_label(self, metrics, preds, labels):
        for metric in metrics:
            metric.update(labels, preds)

    def _update_metric_multi_label(self, metrics, preds, labels):
        for pred, label in zip(preds, labels):
            for idx in range(self.num_labels):
                metrics[idx].update(label[idx], pred[idx])
