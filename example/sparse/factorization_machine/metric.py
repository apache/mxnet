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

import mxnet as mx
import numpy as np
from operator import itemgetter

@mx.metric.register
@mx.metric.alias('log_loss')
class LogLossMetric(mx.metric.EvalMetric):
    """Computes the negative log-likelihood loss.

    The negative log-likelihoodd loss over a batch of sample size :math:`N` is given by

    .. math::
       -\\sum_{n=1}^{N}\\sum_{k=1}^{K}t_{nk}\\log (y_{nk}),

    where :math:`K` is the number of classes, :math:`y_{nk}` is the prediceted probability for
    :math:`k`-th class for :math:`n`-th sample. :math:`t_{nk}=1` if and only if sample
    :math:`n` belongs to class :math:`k`.

    Parameters
    ----------
    eps : float
        Negative log-likelihood loss is undefined for predicted value is 0,
        so predicted values are added with the small constant.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3], [0], [0.4]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> log_loss= mx.metric.NegativeLogLikelihood()
    >>> log_loss.update(labels, predicts)
    >>> print(log_loss.get())
    ('log-loss', 0.57159948348999023)
    """
    def __init__(self, eps=1e-12, name='log-loss',
                 output_names=None, label_names=None):
        super(LogLossMetric, self).__init__(
            name, eps=eps,
            output_names=output_names, label_names=label_names)
        self.eps = eps

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        mx.metric.check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()
            pred = np.column_stack((1 - pred, pred))

            label = label.ravel()
            num_examples = pred.shape[0]
            assert label.shape[0] == num_examples, (label.shape[0], num_examples)
            prob = pred[np.arange(num_examples, dtype=np.int64), np.int64(label)]
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += num_examples

@mx.metric.register
@mx.metric.alias('auc')
class AUCMetric(mx.metric.EvalMetric):
    def __init__(self, eps=1e-12):
        super(AUCMetric, self).__init__(
            'auc')
        self.eps = eps

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        label_weight = labels[0].asnumpy()
        preds = preds[0].asnumpy()
        tmp = []
        for i in range(preds.shape[0]):
            tmp.append((label_weight[i], preds[i]))
        tmp = sorted(tmp, key=itemgetter(1), reverse=True)
        label_sum = label_weight.sum()
        if label_sum == 0 or label_sum == label_weight.size:
            raise Exception("AUC with one class is undefined")

        label_one_num = np.count_nonzero(label_weight)
        label_zero_num = len(label_weight) - label_one_num
        total_area = label_zero_num * label_one_num
        height = 0
        width = 0
        area = 0
        for a, _ in tmp:
            if a == 1.0:
                height += 1.0
            else:
                width += 1.0
                area += height

        self.sum_metric += area / total_area
        self.num_inst += 1
