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
"""TensorBoard functions that can be used to log various status during epoch."""
from __future__ import absolute_import

import logging



class LogMetricsCallback(object):
    """Log metrics periodically in TensorBoard.
    This callback works almost same as `callback.Speedometer`, but write TensorBoard event file
    for visualization. For more usage, please refer https://github.com/dmlc/tensorboard

    Parameters
    ----------
    logging_dir : str
        TensorBoard event file directory.
        After that, use `tensorboard --logdir=path/to/logs` to launch TensorBoard visualization.
    prefix : str
        Prefix for a metric name of `scalar` value.
        You might want to use this param to leverage TensorBoard plot feature,
        where TensorBoard plots different curves in one graph when they have same `name`.
        The follow example shows the usage(how to compare a train and eval metric in a same graph).

    Examples
    --------
    >>> # log train and eval metrics under different directories.
    >>> training_log = 'logs/train'
    >>> evaluation_log = 'logs/eval'
    >>> # in this case, each training and evaluation metric pairs has same name,
    >>> # you can add a prefix to make it separate.
    >>> batch_end_callbacks = [mx.contrib.tensorboard.LogMetricsCallback(training_log)]
    >>> eval_end_callbacks = [mx.contrib.tensorboard.LogMetricsCallback(evaluation_log)]
    >>> # run
    >>> model.fit(train,
    >>>     ...
    >>>     batch_end_callback = batch_end_callbacks,
    >>>     eval_end_callback  = eval_end_callbacks)
    >>> # Then use `tensorboard --logdir=logs/` to launch TensorBoard visualization.
    """
    def __init__(self, logging_dir, prefix=None):
        self.prefix = prefix
        try:
            from tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(logging_dir)
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to log training speed and metrics in TensorBoard."""
        if param.eval_metric is None:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_scalar(name, value)

    def node_histogram_visualization(self, prefix=None, node_names=None, bins="auto"):
        """Node histogram visualization in TensorBoard.
        This callback works almost same as `callback.module_checkpoint`, but write TensorBoard event file
        for visualization. For more usage, please refer https://github.com/dmlc/tensorboard

        Parameters
        ----------
        prefix : str
            Prefix for a metric name of `histograms` and `distributions` value.
        node_names : list of str, optional
            Name of nodes list you want to visualize.
            If set 'None', this callback visualize all nodes histogram and distributions.
            Default node_names = None.
        bins : str
            one of {'tensorflow','auto', 'fd', ...}, this determines how the bins are made. 
            You can find other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            Default bins = 'auto'
        """
        self.histogram_prefix = prefix
        self.node_names = node_names
        self.bins = bins

        def _callback(iter_no, sym, arg, aux):
            """Callback to log node histogram visualization in TensorBoard."""
            for k,v in arg.items():
                if self.node_names is None or k in self.node_names:
                    name = k
                    if self.histogram_prefix is not None:
                        name = '%s-%s' % (self.histogram_prefix, k)
                    self.summary_writer.add_histogram(name, v, global_step=iter_no, bins=self.bins)
        return _callback
