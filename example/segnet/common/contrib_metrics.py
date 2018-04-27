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

import numpy
import mxnet as mx

class Accuracy(mx.metric.EvalMetric):
    """Computes accuracy classification score.

    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    ignore_label : int
        Number of label that should not be computed.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> acc = mx.metric.Accuracy()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None, ignore_label=-1):
        super(Accuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis
        self.ignore_label = ignore_label

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

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)
            self.sum_metric -= len(pred_label[pred_label == self.ignore_label])
            self.num_inst -= len(label[label == self.ignore_label])

class CrossEntropy(mx.metric.EvalMetric):
    """Computes Cross Entropy loss.

    The cross entropy over a batch of sample size :math:`N` is given by

    .. math::
       -\\sum_{n=1}^{N}\\sum_{k=1}^{K}t_{nk}\\log (y_{nk}),

    where :math:`t_{nk}=1` if and only if sample :math:`n` belongs to class :math:`k`.
    :math:`y_{nk}` denotes the probability of sample :math:`n` belonging to
    class :math:`k`.

    Parameters
    ----------
    eps : float
        Cross Entropy loss is undefined for predicted value is 0 or 1,
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
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> ce = mx.metric.CrossEntropy()
    >>> ce.update(labels, predicts)
    >>> print ce.get()
    ('cross-entropy', 0.57159948348999023)
    """
    def __init__(self, eps=1e-12, name='cross-entropy',
                 output_names=None, label_names=None):
        super(CrossEntropy, self).__init__(
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
            if len(pred.shape) > 2:
                for i in range(len(pred.shape) - 1, 1, -1):
                    pred = pred.swapaxes(1, i)
            pred = pred.reshape((-1, pred.shape[-1]))
            label = label.asnumpy()
            pred = pred.asnumpy()

            label = label.ravel()
            assert label.shape[0] == pred.size / pred.shape[-1]
            while label.max() >= pred.shape[1]:
                pred = numpy.append(pred, numpy.ones((pred.shape[0], 1)), 1)
            prob = pred[numpy.arange(label.shape[0]), numpy.int64(label)]
            self.sum_metric += (-numpy.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]
            