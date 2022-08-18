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
# pylint: disable=no-member, too-many-lines

"""Online evaluation metric module."""
import math
from collections import OrderedDict

from .. import numpy
from ..util import use_np

from ..base import numeric_types, string_types
from .. import ndarray, npx
from .. import registry


def check_label_shapes(labels, preds, wrap=False, shape=False):
    """Helper function for checking shape of label and prediction

    Parameters
    ----------
    labels : list of `NDArray`
        The labels of the data.

    preds : list of `NDArray`
        Predicted values.

    wrap : boolean
        If True, wrap labels/preds in a list if they are single NDArray

    shape : boolean
        If True, check the shape of labels and preds;
        Otherwise only check their length.
    """
    if not shape:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

    if wrap:
        if isinstance(labels, ndarray.ndarray.NDArray):
            labels = [labels]
        if isinstance(preds, ndarray.ndarray.NDArray):
            preds = [preds]

    return labels, preds

class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, name, output_names=None,
                 label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(``**config``)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred

        Parameters
        ----------
        labels : OrderedDict of str -> NDArray
            name to array mapping for labels.

        preds : OrderedDict of str -> NDArray
            name to array mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            res = self.sum_metric / self.num_inst
            if isinstance(res, numpy.ndarray) and len(res.shape) == 0:
                # currently calling ' c = mxnet.numpy.array([1,2,3]).sum() ' would get
                # ' array(6.) ', a ndarray with shape ()
                # In this case, returning a 'float' in .get() is more explicit.
                res = res.item()
            return (self.name, res)

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

# pylint: disable=invalid-name
register = registry.get_register_func(EvalMetric, 'metric')
alias = registry.get_alias_func(EvalMetric, 'metric')
_create = registry.get_create_func(EvalMetric, 'metric')
# pylint: enable=invalid-name


def create(metric, *args, **kwargs):
    """Creates evaluation metric from metric names or instances of EvalMetric
    or a custom metric function.

    Parameters
    ----------
    metric : str or callable
        Specifies the metric to create.
        This argument must be one of the below:

        - Name of a metric.
        - An instance of `EvalMetric`.
        - A list, each element of which is a metric or a metric name.
        - An evaluation function that computes custom metric for a given batch of
          labels and predictions.
    *args : list
        Additional arguments to metric constructor.
        Only used when metric is str.
    **kwargs : dict
        Additional arguments to metric constructor.
        Only used when metric is str

    Examples
    --------
    >>> def custom_metric(label, pred):
    ...     return np.mean(np.abs(label - pred))
    ...
    >>> metric1 = mx.gluon.metric.create('acc')
    >>> metric2 = mx.gluon.metric.create(custom_metric)
    >>> metric3 = mx.gluon.metric.create([metric1, metric2, 'rmse'])
    """
    if callable(metric):
        return CustomMetric(metric, *args, **kwargs)
    elif isinstance(metric, list):
        composite_metric = CompositeEvalMetric()
        for child_metric in metric:
            composite_metric.add(create(child_metric, *args, **kwargs))
        return composite_metric

    return _create(metric, *args, **kwargs)


@register
@alias('composite')
class CompositeEvalMetric(EvalMetric):
    """Manages multiple evaluation metrics.

    Parameters
    ----------
    metrics : list of EvalMetric
        List of child metrics.
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
    >>> predicts = [mx.np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.np.array([0, 1, 1])]
    >>> eval_metrics_1 = mx.gluon.metric.Accuracy()
    >>> eval_metrics_2 = mx.gluon.metric.F1()
    >>> eval_metrics = mx.gluon.metric.CompositeEvalMetric()
    >>> for child_metric in [eval_metrics_1, eval_metrics_2]:
    >>>     eval_metrics.add(child_metric)
    >>> eval_metrics.update(labels = labels, preds = predicts)
    >>> eval_metrics.get()
    (['accuracy', 'f1'], [0.6666666666666666, 0.8])
    """

    def __init__(self, metrics=None, name='composite',
                 output_names=None, label_names=None):
        super(CompositeEvalMetric, self).__init__(
            name, output_names=output_names, label_names=label_names)
        if metrics is None:
            metrics = []
        self.metrics = [create(i) for i in metrics]

    def add(self, metric):
        """Adds a child metric.

        Parameters
        ----------
        metric
            A metric instance.
        """
        self.metrics.append(create(metric))

    def get_metric(self, index):
        """Returns a child metric.

        Parameters
        ----------
        index : int
            Index of child metric in the list of metrics.
        """
        try:
            return self.metrics[index]
        except IndexError:
            return ValueError("Metric index {} is out of range 0 and {}".format(
                index, len(self.metrics)))

    def update_dict(self, labels, preds): # pylint: disable=arguments-differ
        if self.label_names is not None:
            labels = OrderedDict([i for i in labels.items()
                                  if i[0] in self.label_names])
        if self.output_names is not None:
            preds = OrderedDict([i for i in preds.items()
                                 if i[0] in self.output_names])

        for metric in self.metrics:
            metric.update_dict(labels, preds)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def get(self):
        """Returns the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get()
            if isinstance(name, string_types):
                name = [name]
            if isinstance(value, numeric_types):
                value = [value]
            names.extend(name)
            values.extend(value)
        return (names, values)

    def get_config(self):
        config = super(CompositeEvalMetric, self).get_config()
        config.update({'metrics': [i.get_config() for i in self.metrics]})
        return config


########################
# CLASSIFICATION METRICS
########################


@register
@alias('acc')
@use_np
class Accuracy(EvalMetric):
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

    Examples
    --------
    >>> predicts = [mx.np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.np.array([0, 1, 1])]
    >>> acc = mx.gluon.metric.Accuracy()
    >>> acc.update(preds = predicts, labels = labels)
    >>> acc.get()
    ('accuracy', 0.6666666666666666)
    """
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None):
        super(Accuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.

        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            pred_label = pred_label.as_np_ndarray().to_device(label.device)
            label = label.as_np_ndarray()
            if pred_label.shape != label.shape:
                pred_label = pred_label.argmax(axis=self.axis)
            pred_label = pred_label.astype('int32')
            label = label.astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.reshape(-1)
            pred_label = pred_label.reshape(-1)

            check_label_shapes(label, pred_label)

            num_correct = (pred_label == label).sum().astype('float64')
            self.sum_metric += num_correct
            self.num_inst += len(pred_label)


@register
@alias('top_k_accuracy', 'top_k_acc')
@use_np
class TopKAccuracy(EvalMetric):
    """Computes top k predictions accuracy.

    `TopKAccuracy` differs from Accuracy in that it considers the prediction
    to be ``True`` as long as the ground truth label is in the top K
    predicated labels.

    If `top_k` = ``1``, then `TopKAccuracy` is identical to `Accuracy`.

    Parameters
    ----------
    top_k : int
        Whether targets are in top k predictions.
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
    >>> np.random.seed(999)
    >>> top_k = 3
    >>> labels = [mx.np.array([2, 6, 9, 2, 3, 4, 7, 8, 9, 6])]
    >>> predicts = [mx.np.array(np.random.rand(10, 10))]
    >>> acc = mx.gluon.metric.TopKAccuracy(top_k=top_k)
    >>> acc.update(labels, predicts)
    >>> acc.get()
    ('top_k_accuracy', 0.3)
    """

    def __init__(self, top_k=1, name='top_k_accuracy',
                 output_names=None, label_names=None):
        super(TopKAccuracy, self).__init__(
            name, top_k=top_k,
            output_names=output_names, label_names=label_names)
        self.top_k = top_k
        assert(self.top_k > 1), 'Please use Accuracy if top_k is no more than 1'
        self.name += f'_{self.top_k}'

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            assert(len(pred_label.shape) <= 2), 'Predictions should be no more than 2 dims'
            # Using argpartition here instead of argsort is safe because
            # we do not care about the order of top k elements. It is
            # much faster, which is important since that computation is
            # single-threaded due to Python GIL.
            pred_label = pred_label.as_np_ndarray().to_device(label.device).astype('float32')
            pred_label = numpy.argpartition(pred_label, -self.top_k).to_device(label.device)
            label = label.as_np_ndarray().astype('int32')
            check_label_shapes(label, pred_label)
            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                num_correct = (pred_label.reshape(-1) == label.reshape(-1)).sum()
                self.sum_metric += num_correct.astype('float64')
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    num_correct = (pred_label[:, num_classes - 1 - j].reshape(-1) == label.reshape(-1)).sum()
                    self.sum_metric += num_correct.astype('float64')
            self.num_inst += num_samples


def predict_with_threshold(pred, threshold=0.5):
    """Do thresholding of predictions in binary and multilabel cases.

    Parameters
    ----------
    preds : ndarray
        predictions in shape of (batch_size, ...) or (batch_size, ..., num_categories)

    preds : float or ndarray
        threshold（s) in shape of float or (num_categories)
    """
    if isinstance(threshold, float):
        return pred > threshold
    elif isinstance(threshold, (numpy.ndarray, ndarray.ndarray.NDArray)):
        num_classes = pred.shape[-1]
        assert threshold.shape[-1] == num_classes, \
                f"shape mismatch: {pred.shape[-1]} vs. {threshold.shape[-1]}"
        return pred > threshold
    else:
        raise ValueError("{} is a wrong type for threshold!".format(type(threshold)))


def one_hot(idx, num):
    return (numpy.arange(num).astype(idx) == idx[:, None]).astype('int32')


@use_np
class _ClassificationMetrics(object):
    """Private container class for classification metric statistics.

    True/false positive and true/false negative counts are sufficient statistics for various classification metrics.
    This class provides the machinery to track those statistics across mini-batches of
    (label, prediction) pairs.

    Parameters
    ----------
    class_type : str, default "binary"
        "binary": f1 for binary classification.
        "multiclass": f1 for multiclassification problem.
        "multilabel": f1 for multilabel classification.
    beta : float, default 1
        weight of precision in harmonic mean.
    threshold : float, default 0.5
        threshold for deciding whether the predictions are positive or negative.

    """

    def __init__(self, class_type="binary", threshold=0.5, beta=1):
        self.class_type = class_type
        self.threshold = threshold
        self.beta = beta
        self.reset_stats()

    def _set(self, num, device):
        if self.num_classes is None:
            self.num_classes = num
            self.true_positives = numpy.zeros(num, dtype='float64').to_device(device)
            self.false_negatives = numpy.zeros(num, dtype='float64').to_device(device)
            self.false_positives = numpy.zeros(num, dtype='float64').to_device(device)
            self.true_negatives = numpy.zeros(num, dtype='float64').to_device(device)
        else:
            assert self.num_classes == num, \
                "Input number of classes has changed from {} to {}".format(self.num_classes, num)

    def update_stats(self, label, pred):
        """Update various binary classification counts for a single (label, pred) pair.

        Parameters
        ----------
        label : `NDArray`
            The labels of the data.

        pred : `NDArray`
            Predicted values.
        """
        pred = pred.as_np_ndarray().to_device(label.device)
        label = label.as_np_ndarray().astype('int32')
        if self.class_type == "binary":
            self._set(1, label.device)
            if label.max() > 1:
                raise ValueError("Wrong label for binary classification.")
            if pred.shape == label.shape:
                pass
            elif pred.shape[-1] > 2:
                raise ValueError("The shape of prediction {} is wrong for binary classification.".format(pred.shape))
            elif pred.shape[-1] == 2:
                pred = pred.reshape(-1, 2)[:, 1]
            pred_label = predict_with_threshold(pred, self.threshold).reshape(-1)
            label = label.reshape(-1)

        elif self.class_type == "multiclass":
            num = pred.shape[-1]
            self._set(num, label.device)
            assert label.max() < num, "pred contains fewer classes than label!"
            pred_label = one_hot(pred.argmax(axis=-1).reshape(-1), num)
            label = one_hot(label.reshape(-1), num)

        elif self.class_type == "multilabel":
            num = pred.shape[-1]
            self._set(num, label.device)
            assert pred.shape == label.shape, \
                "The shape of label should be same as that of prediction for multilabel classification."
            pred_label = predict_with_threshold(pred, self.threshold).reshape(-1, num)
            label = label.reshape(-1, num)
        else:
            raise ValueError(
                "Wrong class_type {}! Only supports ['binary', 'multiclass', 'multilabel']".format(self.class_type))

        check_label_shapes(label, pred_label)

        pred_true = (pred_label == 1)
        pred_false = (pred_label == 0)
        label_true = (label == 1)
        label_false = (label == 0)

        true_pos = (pred_true * label_true).sum(0)
        false_pos = (pred_true * label_false).sum(0)
        false_neg = (pred_false * label_true).sum(0)
        true_neg = (pred_false * label_false).sum(0)
        self.true_positives += true_pos
        self.false_positives += false_pos
        self.false_negatives += false_neg
        self.true_negatives += true_neg

    @property
    def precision(self):
        if self.num_classes is not None:
            return self.true_positives / numpy.maximum(self.true_positives + self.false_positives, 1e-12)
        else:
            return 0.

    @property
    def micro_precision(self):
        if self.num_classes is not None:
            return self.true_positives.sum() / \
                numpy.maximum(self.true_positives.sum() + self.false_positives.sum(), 1e-12)
        else:
            return 0.

    @property
    def recall(self):
        if self.num_classes is not None:
            return self.true_positives / numpy.maximum(self.true_positives + self.false_negatives, 1e-12)
        else:
            return 0.

    @property
    def micro_recall(self):
        if self.num_classes is not None:
            return self.true_positives.sum() / \
                numpy.maximum(self.true_positives.sum() + self.false_negatives.sum(), 1e-12)
        else:
            return 0.

    @property
    def fscore(self):
        return (1 + self.beta ** 2) * self.precision * self.recall / \
            numpy.maximum(self.beta ** 2 * self.precision + self.recall, 1e-12)

    @property
    def micro_fscore(self):
        if self.micro_precision + self.micro_recall > 0:
            return (1 + self.beta ** 2) * self.micro_precision * self.micro_recall / \
                (self.beta ** 2 * self.micro_precision + self.micro_recall)
        else:
            return 0.

    def binary_matthewscc(self):
        """Calculate the Matthew's Correlation Coefficent"""
        if not self.total_examples:
            return 0.

        true_pos = float(self.true_positives)
        false_pos = float(self.false_positives)
        false_neg = float(self.false_negatives)
        true_neg = float(self.true_negatives)

        terms = [(true_pos + false_pos),
                 (true_pos + false_neg),
                 (true_neg + false_pos),
                 (true_neg + false_neg)]
        denom = 1.
        for t in filter(lambda t: t != 0., terms):
            denom *= t
        return ((true_pos * true_neg) - (false_pos * false_neg)) / math.sqrt(denom)

    @property
    def total_examples(self):
        if self.num_classes is None:
            return 0
        return int(self.false_negatives[0] + self.false_positives[0] + \
               self.true_negatives[0] + self.true_positives[0])

    def reset_stats(self):
        self.num_classes = None
        self.true_positives = None
        self.false_negatives = None
        self.false_positives = None
        self.true_negatives = None


@register
@use_np
class F1(EvalMetric):
    """Computes the F1 score of a binary classification problem.

    The F1 score is equivalent to harmonic mean of the precision and recall,
    where the best value is 1.0 and the worst value is 0.0. The formula for F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    The formula for precision and recall is::

        precision = true_positives / (true_positives + false_positives)
        recall    = true_positives / (true_positives + false_negatives)

    .. note::

        This F1 score only supports binary classification.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    class_type : str, default "binary"
        "binary": f1 for binary classification.
        "multiclass": f1 for multiclassification problem.
        "multilabel": f1 for multilabel classification.
    threshold : float, default 0.5
        threshold for postive confidence value.
    average : str, default 'micro'
        Strategy to be used for aggregating across mini-batches.
            "macro": Calculate metrics for each label and return unweighted mean of f1.
            "micro": Calculate metrics globally by counting the total TP, FN and FP.
            None: Return f1 scores for each class (numpy.ndarray) .

    Examples
    --------
    >>> predicts = [mx.np.array([[0.3, 0.7], [0., 1.], [0.4, 0.6]])]
    >>> labels   = [mx.np.array([0., 1., 1.])]
    >>> f1 = mx.gluon.metric.F1()
    >>> f1.update(preds = predicts, labels = labels)
    >>> f1.get()
    ('f1', 0.8)
    """

    def __init__(self, name='f1',
                 output_names=None, label_names=None, class_type="binary", threshold=0.5, average="micro"):
        self.average = average
        self.metrics = _ClassificationMetrics(class_type=class_type, threshold=threshold)
        EvalMetric.__init__(self, name=name,
                            output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            self.metrics.update_stats(label, pred)

        if self.average == "micro":
            self.sum_metric = self.metrics.micro_fscore * self.metrics.total_examples
        elif self.average == "macro":
            self.sum_metric = self.metrics.fscore.mean() * self.metrics.total_examples
        else:
            self.sum_metric = self.metrics.fscore * self.metrics.total_examples
        self.num_inst = self.metrics.total_examples

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.sum_metric = 0.
        self.num_inst = 0
        self.metrics.reset_stats()


@register
@use_np
class Fbeta(F1):
    """Computes the Fbeta score of a binary classification problem.

    The Fbeta score is equivalent to harmonic mean of the precision and recall,
    where the best value is 1.0 and the worst value is 0.0. The formula for Fbeta score is::

        Fbeta = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)

    The formula for precision and recall is::

        precision = true_positives / (true_positives + false_positives)
        recall    = true_positives / (true_positives + false_negatives)

    .. note::

        This Fbeta score only supports binary classification.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    class_type : str, default "binary"
        "binary": f1 for binary classification.
        "multiclass": f1 for multiclassification problem.
        "multilabel": f1 for multilabel classification.
    beta : float, default 1
        weight of precision in harmonic mean.
    threshold : float, default 0.5
        threshold for postive confidence value.
    average : str, default 'micro'
        Strategy to be used for aggregating across mini-batches.
            "macro": Calculate metrics for each label and return unweighted mean of f1.
            "micro": Calculate metrics globally by counting the total TP, FN and FP.
            None: Return f1 scores for each class.

    Examples
    --------
    >>> predicts = [mx.np.array([[0.3, 0.7], [0., 1.], [0.4, 0.6]])]
    >>> labels   = [mx.np.array([0., 1., 1.])]
    >>> fbeta = mx.gluon.metric.Fbeta(beta=2)
    >>> fbeta.update(preds = predicts, labels = labels)
    >>> fbeta.get()
    ('fbeta', 0.9090909090909091)
    """

    def __init__(self, name='fbeta',
                 output_names=None, label_names=None, class_type="binary", beta=1, threshold=0.5, average="micro"):
        super(Fbeta, self).__init__(
            name=name, output_names=output_names, label_names=label_names,
            class_type=class_type, threshold=threshold, average=average)
        self.metrics = _ClassificationMetrics(class_type=class_type, threshold=threshold, beta=beta)


@register
@use_np
class BinaryAccuracy(EvalMetric):
    """Computes the accuracy of a binary or multilabel classification problem.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    threshold : float or ndarray, default 0.5
        threshold for deciding whether the predictions are positive or negative.

    Examples
    --------
    >>> predicts = [mx.np.array([0.7, 1, 0.55])]
    >>> labels   = [mx.np.array([0., 1., 0.])]
    >>> bacc = mx.gluon.metric.BinaryAccuracy(threshold=0.6)
    >>> bacc.update(preds = predicts, labels = labels)
    >>> bacc.get()
    ('binary_accuracy', 0.6666666666666666)
    """

    def __init__(self, name='binary_accuracy',
                 output_names=None, label_names=None, threshold=0.5):
        self.threshold = threshold
        EvalMetric.__init__(self, name=name,
                            output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            Each label denotes positive/negative for each class.

        preds : list of `NDArray`
            Each prediction value is a confidence value of being positive for each class.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            pred_label = predict_with_threshold(pred_label, self.threshold)

            pred_label = pred_label.as_np_ndarray().astype('int32').to_device(label.device)
            label = label.as_np_ndarray().astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.reshape(-1)
            pred_label = pred_label.reshape(-1)

            check_label_shapes(label, pred_label)

            num_correct = (pred_label == label).sum().astype('float64')
            self.sum_metric += num_correct
            self.num_inst += len(pred_label)


@register
@use_np
class MCC(EvalMetric):
    """Computes the Matthews Correlation Coefficient of a binary classification problem.

    While slower to compute than F1 the MCC can give insight that F1 or Accuracy cannot.
    For instance, if the network always predicts the same result
    then the MCC will immeadiately show this. The MCC is also symetric with respect
    to positive and negative categorization, however, there needs to be both
    positive and negative examples in the labels or it will always return 0.
    MCC of 0 is uncorrelated, 1 is completely correlated, and -1 is negatively correlated.

    .. math::

        \\text{MCC} = \\frac{ TP \\times TN - FP \\times FN }
        {\\sqrt{ (TP + FP) ( TP + FN ) ( TN + FP ) ( TN + FN ) } }

    where 0 terms in the denominator are replaced by 1.

    .. note::

        This version of MCC only supports binary classification.  See PCC.

    Parameters
    ----------
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
    >>> # In this example the network almost always predicts positive
    >>> false_positives = 1000
    >>> false_negatives = 1
    >>> true_positives = 10000
    >>> true_negatives = 1
    >>> predicts = [mx.np.array(
        [[.3, .7]]*false_positives +
        [[.7, .3]]*true_negatives +
        [[.7, .3]]*false_negatives +
        [[.3, .7]]*true_positives
    )]
    >>> labels  = [mx.np.array(
        [0.]*(false_positives + true_negatives) +
        [1.]*(false_negatives + true_positives)
    )]
    >>> f1 = mx.gluon.metric.F1()
    >>> f1.update(preds = predicts, labels = labels)
    >>> mcc = mx.gluon.metric.MCC()
    >>> mcc.update(preds = predicts, labels = labels)
    >>> f1.get()
    ('f1', 0.95233560306652054)
    >>> mcc.get()
    ('mcc', 0.01917751877733392)
    """

    def __init__(self, name='mcc',
                 output_names=None, label_names=None):
        self._metrics = _ClassificationMetrics()
        EvalMetric.__init__(self, name=name,
                            output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            self._metrics.update_stats(label, pred)

        self.sum_metric = self._metrics.binary_matthewscc() * self._metrics.total_examples
        self.num_inst = self._metrics.total_examples

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.sum_metric = 0.
        self.num_inst = 0.
        self._metrics.reset_stats()


####################
# REGRESSION METRICS
####################


@register
@use_np
class MAE(EvalMetric):
    """Computes Mean Absolute Error (MAE) loss.

    The mean absolute error is given by

    .. math::

        \\frac{\\sum_i^n |y_i - \\hat{y}_i|}{n}

    Parameters
    ----------
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
    >>> predicts = [mx.np.array([3, -0.5, 2, 7])]
    >>> labels = [mx.np.array([2.5, 0.0, 2, 8])]
    >>> mean_absolute_error = mx.gluon.metric.MAE()
    >>> mean_absolute_error.update(labels = labels, preds = predicts)
    >>> mean_absolute_error.get()
    ('mae', 0.5)
    """

    def __init__(self, name='mae',
                 output_names=None, label_names=None):
        super(MAE, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            label = label.as_np_ndarray()
            pred = pred.as_np_ndarray().to_device(label.device)

            num_inst = label.shape[0]
            mae = numpy.abs(label - pred).reshape(num_inst, -1).mean(axis=-1).sum()

            self.sum_metric += mae
            self.num_inst += num_inst


@register
@use_np
class MSE(EvalMetric):
    """Computes Mean Squared Error (MSE) loss.

    The mean squared error is given by

    .. math::

        \\frac{\\sum_i^n (y_i - \\hat{y}_i)^2}{n}

    Parameters
    ----------
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
    >>> predicts = [mx.np.array([3, -0.5, 2, 7])]
    >>> labels = [mx.np.array([2.5, 0.0, 2, 8])]
    >>> mean_squared_error = mx.gluon.metric.MSE()
    >>> mean_squared_error.update(labels = labels, preds = predicts)
    >>> mean_squared_error.get()
    ('mse', 0.375)
    """
    def __init__(self, name='mse',
                 output_names=None, label_names=None):
        super(MSE, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            label = label.as_np_ndarray()
            pred = pred.as_np_ndarray().to_device(label.device)

            num_inst = label.shape[0]
            mse = ((label - pred)**2.0).reshape(num_inst, -1).mean(axis=-1).sum()

            self.sum_metric += mse
            self.num_inst += num_inst


@register
@use_np
class RMSE(MSE):
    """Computes Root Mean Squred Error (RMSE) loss.

    The root mean squared error is given by

    .. math::

        \\sqrt{\\frac{\\sum_i^n (y_i - \\hat{y}_i)^2}{n}}

    Parameters
    ----------
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
    >>> predicts = [mx.np.array([3, -0.5, 2, 7])]
    >>> labels = [mx.np.array([2.5, 0.0, 2, 8])]
    >>> root_mean_squared_error = mx.gluon.metric.RMSE()
    >>> root_mean_squared_error.update(labels = labels, preds = predicts)
    >>> root_mean_squared_error.get()
    ('rmse', 0.612372457981)
    """
    def __init__(self, name='rmse',
                 output_names=None, label_names=None):
        super(RMSE, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, math.sqrt(self.sum_metric / self.num_inst))


@register
@use_np
class MeanPairwiseDistance(EvalMetric):
    """Computes Mean Pairwise Distance.

    The mean pairwise distance is given by

    .. math::

        \\sqrt{\\frac{(\\sum_i^n (y_i - \\hat{y}_i)^p)^\\frac{1}{p}}{n}}

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    p : float, default 2
        calculating distance using the p-norm

    Examples
    --------
    >>> predicts = [mx.np.array([[1., 2.], [3., 4.]])]
    >>> labels = [mx.np.array([[1., 0.], [4., 2.]])]
    >>> mpd = mx.gluon.metric.MeanPairwiseDistance()
    >>> mpd.update(labels = labels, preds = predicts)
    >>> mpd.get()
    ('mpd', 2.1180338859558105)
    """
    def __init__(self, name='mpd',
                 output_names=None, label_names=None, p=2):
        super(MeanPairwiseDistance, self).__init__(
            name, output_names=output_names, label_names=label_names)
        self.p = p

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            label = label.as_np_ndarray()
            pred = pred.as_np_ndarray().to_device(label.device)

            label = label.reshape(label.shape[0], -1)
            pred = pred.reshape(pred.shape[0], -1)

            dis = (((label - pred) ** self.p).sum(axis=-1)) ** (1./self.p)
            dis = dis.sum()
            num_inst = label.shape[0]

            self.sum_metric += dis
            self.num_inst += num_inst


@register
@use_np
class MeanCosineSimilarity(EvalMetric):
    r"""Computes Mean Cosine Similarity.

    The mean cosine similarity is given by

    .. math::

        cos_sim(label, pred) = \frac{{label}.{pred}}{max(||label||.||pred||, eps)}

    Calculation happens on the last dimension of label and pred.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    eps : float, default 1e-8
        small vale to avoid division by zero.

    Examples
    --------
    >>> predicts = [mx.np.array([[1., 0.], [1., 1.]])]
    >>> labels = [mx.np.array([[3., 4.], [2., 2.]])]
    >>> mcs = mx.gluon.metric.MeanCosineSimilarity()
    >>> mcs.update(labels = labels, preds = predicts)
    >>> mcs.get()
    ('cos_sim', 0.8)
    """
    def __init__(self, name='cos_sim',
                 output_names=None, label_names=None, eps=1e-8):
        super(MeanCosineSimilarity, self).__init__(
            name, output_names=output_names, label_names=label_names)
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
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            label = label.as_np_ndarray()
            pred = pred.as_np_ndarray().to_device(label.device)

            if len(label.shape) == 1:
                label = label.reshape(1, label.shape[0])
            if len(pred.shape) == 1:
                pred = pred.reshape(1, pred.shape[0])

            sim = (label * pred).sum(axis=-1)
            n_p = numpy.linalg.norm(pred, axis=-1)
            n_l = numpy.linalg.norm(label, axis=-1)
            sim = sim / numpy.maximum(n_l * n_p, self.eps)
            sim = sim.sum()
            num_inst = len(label.reshape(-1, label.shape[-1])) # numpy.prod(label.shape[:-1]) is not supported
            self.sum_metric += sim
            self.num_inst += num_inst


@register
@alias('ce')
@use_np
class CrossEntropy(EvalMetric):
    """Computes Cross Entropy loss.

    The cross entropy over a batch of sample size :math:`N` is given by

    .. math::

       -\\sum_{n=1}^{N}\\sum_{k=1}^{K}t_{nk}\\log (y_{nk}),

    where :math:`t_{nk}=1` if and only if sample :math:`n` belongs to class :math:`k`.
    :math:`y_{nk}` denotes the probability of sample :math:`n` belonging to
    class :math:`k`.

    Parameters
    ----------
    eps : float, default 1e-12
        Use small constant for the case that predicted value is 0.
    ignore_label : int or None, default None
        Index of invalid label to ignore when
        counting. By default, sets to -1.
        If set to `None`, it will include all entries.
    axis : int, default -1
        The axis from prediction that was used to
        compute softmax. By default use the last axis.
    from_logits : boolean, default False
        Whether `pred` is expected to be a logits tensor.
        By default, we assume that `pred` encodes a probability distribution.
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
    >>> predicts = [mx.np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.np.array([0, 1, 1])]
    >>> ce = mx.gluon.metric.CrossEntropy()
    >>> ce.update(labels, predicts)
    >>> ce.get()
    ('cross-entropy', 0.57159948348999023)
    """
    def __init__(self, eps=1e-12, ignore_label=None, axis=-1, from_logits=False,
                 name='cross-entropy', output_names=None, label_names=None):
        super(CrossEntropy, self).__init__(
            name, output_names=output_names, label_names=label_names)
        self.ignore_label = ignore_label
        self.axis = axis
        self.from_logits = from_logits
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
        labels, preds = check_label_shapes(labels, preds, True)

        loss = 0.
        num = 0
        for label, pred in zip(labels, preds):
            assert label.size == pred.size/pred.shape[-1], \
                f"shape mismatch: {label.shape} vs. {pred.shape}"
            label = label.reshape((label.size,))
            if self.from_logits:
                pred = npx.softmax(pred, axis=self.axis)
            pred = npx.pick(pred.to_device(label.device), label.astype(dtype='int32'), axis=self.axis)
            if self.ignore_label is not None:
                ignore = (label == self.ignore_label).astype(pred.dtype)
                num -= ignore.sum()
                pred = pred * (1 - ignore) + ignore
            loss -= numpy.log(numpy.maximum(self.eps, pred)).sum()
            num += pred.size
        self.sum_metric += loss
        self.num_inst += num


@register
@use_np
class Perplexity(CrossEntropy):
    """Computes perplexity.

    Perplexity is a measurement of how well a probability distribution
    or model predicts a sample. A low perplexity indicates the model
    is good at predicting the sample.

    The perplexity of a model q is defined as

    .. math::

        b^{\\big(-\\frac{1}{N} \\sum_{i=1}^N \\log_b q(x_i) \\big)}
        = \\exp \\big(-\\frac{1}{N} \\sum_{i=1}^N \\log q(x_i)\\big)

    where we let `b = e`.

    :math:`q(x_i)` is the predicted value of its ground truth
    label on sample :math:`x_i`.

    For example, we have three samples :math:`x_1, x_2, x_3` and their labels
    are :math:`[0, 1, 1]`.
    Suppose our model predicts :math:`q(x_1) = p(y_1 = 0 | x_1) = 0.3`
    and :math:`q(x_2) = 1.0`,
    :math:`q(x_3) = 0.6`. The perplexity of model q is
    :math:`exp\\big(-(\\log 0.3 + \\log 1.0 + \\log 0.6) / 3\\big) = 1.77109762852`.

    Parameters
    ----------
    eps : float, default 1e-12
        Use small constant for the case that predicted value is 0.
    ignore_label : int or None, default None
        Index of invalid label to ignore when
        counting. By default, sets to -1.
        If set to `None`, it will include all entries.
    axis : int (default -1)
        The axis from prediction that was used to
        compute softmax. By default use the last axis.
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
    >>> predicts = [mx.np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.np.array([0, 1, 1])]
    >>> perp = mx.gluon.metric.Perplexity(ignore_label=None)
    >>> perp.update(labels, predicts)
    >>> perp.get()
    ('Perplexity', 1.7710976285155853)
    """
    def __init__(self, eps=1e-12, ignore_label=None, axis=-1, from_logits=False,
                 name='perplexity', output_names=None, label_names=None):
        super(Perplexity, self).__init__(
            eps=eps, ignore_label=ignore_label, axis=axis, from_logits=from_logits,
            name=name, output_names=output_names, label_names=label_names)

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, math.exp(self.sum_metric/self.num_inst))


@register
@alias('pearsonr')
@use_np
class PearsonCorrelation(EvalMetric):
    """Computes Pearson correlation.

    The pearson correlation is given by

    .. math::

        \\frac{cov(y, \\hat{y})}{\\sigma{y}\\sigma{\\hat{y}}}

    Parameters
    ----------
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
    >>> predicts = [mx.np.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.np.array([[1, 0], [0, 1], [0, 1]])]
    >>> pr = mx.gluon.metric.PearsonCorrelation()
    >>> pr.update(labels, predicts)
    >>> pr.get()
    ('pearsonr', 0.42163704544016178)
    """
    def __init__(self, name='pearsonr',
                 output_names=None, label_names=None):
        super(PearsonCorrelation, self).__init__(
            name, output_names=output_names, label_names=label_names)
        self.reset()

    def reset(self):
        self._sse_p = 0
        self._mean_p = 0
        self._sse_l = 0
        self._mean_l = 0
        self._pred_nums = 0
        self._label_nums = 0
        self._conv = 0

        self.num_inst = 0
        self.sum_metric = 0.0

    def update_variance(self, new_values, *aggregate):
        #Welford's online algorithm for variance update
        count, mean, m_2 = aggregate
        count += len(new_values)
        delta = new_values - mean
        mean += numpy.sum(delta / count)
        delta_2 = new_values - mean
        m_2 += numpy.sum(delta * delta_2)
        return count, mean, m_2

    def update_cov(self, label, pred):
        self._conv = self._conv + numpy.sum((label - self._mean_l) * (pred - self._mean_p))

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)
        for label, pred in zip(labels, preds):
            check_label_shapes(label, pred, False, True)
            label = label.as_np_ndarray().reshape(-1).astype(numpy.float64)
            pred = pred.as_np_ndarray().to_device(label.device).reshape(-1).astype(numpy.float64)

            self.num_inst += 1
            self._label_nums, self._mean_l, self._sse_l = \
                self.update_variance(label, self._label_nums, self._mean_l, self._sse_l)
            self.update_cov(label, pred)
            self._pred_nums, self._mean_p, self._sse_p = \
                self.update_variance(pred, self._pred_nums, self._mean_p, self._sse_p)

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))

        n = self._label_nums
        pearsonr = self._conv / ((n-1) * numpy.sqrt(self._sse_p / (n - 1)) * numpy.sqrt(self._sse_l / (n - 1)))
        return (self.name, float(pearsonr))

@register
@use_np
class PCC(EvalMetric):
    """PCC is a multiclass equivalent for the Matthews correlation coefficient derived
    from a discrete solution to the Pearson correlation coefficient.

    .. math::

        \\text{PCC} = \\frac {\\sum _{k}\\sum _{l}\\sum _{m}C_{kk}C_{lm}-C_{kl}C_{mk}}
        {{\\sqrt {\\sum _{k}(\\sum _{l}C_{kl})(\\sum _{k'|k'\\neq k}\\sum _{l'}C_{k'l'})}}
         {\\sqrt {\\sum _{k}(\\sum _{l}C_{lk})(\\sum _{k'|k'\\neq k}\\sum _{l'}C_{l'k'})}}}

    defined in terms of a K x K confusion matrix C.

    When there are more than two labels the PCC will no longer range between -1 and +1.
    Instead the minimum value will be between -1 and 0 depending on the true distribution.
    The maximum value is always +1.

    Parameters
    ----------
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
    >>> # In this example the network almost always predicts positive
    >>> false_positives = 1000
    >>> false_negatives = 1
    >>> true_positives = 10000
    >>> true_negatives = 1
    >>> predicts = [mx.np.array(
        [[.3, .7]]*false_positives +
        [[.7, .3]]*true_negatives +
        [[.7, .3]]*false_negatives +
        [[.3, .7]]*true_positives
    )]
    >>> labels  = [mx.np.array(
        [0]*(false_positives + true_negatives) +
        [1]*(false_negatives + true_positives)
    )]
    >>> f1 = mx.gluon.metric.F1()
    >>> f1.update(preds = predicts, labels = labels)
    >>> pcc = mx.gluon.metric.PCC()
    >>> pcc.update(preds = predicts, labels = labels)
    >>> f1.get()
    ('f1', 0.95233560306652054)
    >>> pcc.get()
    ('pcc', 0.01917751877733392)
    """
    def __init__(self, name='pcc',
                 output_names=None, label_names=None):
        self.k = 2
        super(PCC, self).__init__(
            name=name, output_names=output_names, label_names=label_names)

    def _grow(self, inc):
        self.lcm = numpy.pad(
            self.lcm, ((0, inc), (0, inc)), 'constant', constant_values=(0))
        self.k += inc

    def _calc_mcc(self, cmat):
        n = cmat.sum()
        x = cmat.sum(axis=1)
        y = cmat.sum(axis=0)
        cov_xx = numpy.sum(x * (n - x))
        cov_yy = numpy.sum(y * (n - y))
        if cov_xx == 0 or cov_yy == 0:
            return float('nan')
        # i = cmat.diagonal() # mxnet.numpy.ndarray.diagonal() is currently not available.
        i = cmat[numpy.arange(self.k), numpy.arange(self.k)]
        cov_xy = numpy.sum(i * n - x * y)
        return cov_xy / (cov_xx * cov_yy) ** 0.5

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        # update the confusion matrix
        for label, pred in zip(labels, preds):
            label = label.astype('int32', copy=False).as_np_ndarray()
            pred = pred.as_np_ndarray().to_device(label.device)
            if pred.shape != label.shape:
                pred = pred.argmax(axis=1).astype(label, copy=False)
            else:
                pred = pred.astype('int32', copy=False)
            n = int(max(pred.max(), label.max()))
            if n >= self.k:
                self._grow(n + 1 - self.k)
            bcm = numpy.zeros((self.k, self.k), dtype='float64')
            for i, j in zip(pred, label):
                bcm[i, j] += 1
            self.lcm += bcm
        self.num_inst += 1

    @property
    def sum_metric(self):
        return self._calc_mcc(self.lcm) * self.num_inst

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0.
        self.lcm = numpy.zeros((self.k, self.k), dtype='float64')


@register
@use_np
class Loss(EvalMetric):
    """Dummy metric for directly printing loss.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, name='loss',
                 output_names=None, label_names=None):
        super(Loss, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, _, preds):

        if isinstance(preds, ndarray.ndarray.NDArray):
            preds = [preds]

        for pred in preds:
            loss = pred.sum().item()
            self.sum_metric += loss
            self.num_inst += pred.size


@register
class Torch(Loss):
    """Dummy metric for torch criterions."""
    def __init__(self, name='torch',
                 output_names=None, label_names=None):
        super(Torch, self).__init__(
            name, output_names=output_names, label_names=label_names)


@register
@use_np
class CustomMetric(EvalMetric):
    """Computes a customized evaluation metric.

    The `feval` function can return a `tuple` of (sum_metric, num_inst) or return
    an `int` sum_metric.

    Parameters
    ----------
    feval : callable(label, pred)
        Customized evaluation function.
    name : str, optional
        The name of the metric. (the default is None).
    allow_extra_outputs : bool, optional
        If true, the prediction outputs can have extra outputs.
        This is useful in RNN, where the states are also produced
        in outputs for forwarding. (the default is False).
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
    >>> predicts = [mx.np.array(np.array([3, -0.5, 2, 7]).reshape(4,1))]
    >>> labels = [mx.np.array(np.array([2.5, 0.0, 2, 8]).reshape(4,1))]
    >>> feval = lambda x, y : (x + y).mean()
    >>> eval_metrics = mx.gluon.metric.CustomMetric(feval=feval)
    >>> eval_metrics.update(labels, predicts)
    >>> eval_metrics.get()
    ('custom(<lambda>)', 6.0)
    """
    def __init__(self, feval, name=None, allow_extra_outputs=False,
                 output_names=None, label_names=None):
        if name is None:
            name = feval.__name__
            if name.find('<') != -1:
                name = f'custom({name})'
        super(CustomMetric, self).__init__(
            name, feval=feval,
            allow_extra_outputs=allow_extra_outputs,
            output_names=output_names, label_names=label_names)
        self._feval = feval
        self._allow_extra_outputs = allow_extra_outputs

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        if not self._allow_extra_outputs:
            labels, preds = check_label_shapes(labels, preds, True)

        for pred, label in zip(preds, labels):
            label = label.as_np_ndarray()
            pred = pred.as_np_ndarray().to_device(label.device)

            reval = self._feval(label, pred)
            if isinstance(reval, tuple):
                (sum_metric, num_inst) = reval
                self.sum_metric += sum_metric
                self.num_inst += num_inst
            else:
                self.sum_metric += reval
                self.num_inst += 1

    def get_config(self):
        raise NotImplementedError("CustomMetric cannot be serialized")


# pylint: disable=invalid-name
def np(numpy_feval, name=None, allow_extra_outputs=False):
    """Creates a custom evaluation metric that receives its inputs as numpy arrays.

    Parameters
    ----------
    numpy_feval : callable(label, pred)
        Custom evaluation function that receives labels and predictions for a minibatch
        as numpy arrays and returns the corresponding custom metric as a floating point number.
    name : str, optional
        Name of the custom metric.
    allow_extra_outputs : bool, optional
        Whether prediction output is allowed to have extra outputs. This is useful in cases
        like RNN where states are also part of output which can then be fed back to the RNN
        in the next step. By default, extra outputs are not allowed.

    Returns
    -------
    float
        Custom metric corresponding to the provided labels and predictions.

    Example
    -------
    >>> def custom_metric(label, pred):
    ...     return np.mean(np.abs(label-pred))
    ...
    >>> metric = mx.gluon.metric.np(custom_metric)
    """
    def feval(label, pred):
        """Internal eval function."""
        return numpy_feval(label, pred)
    feval.__name__ = numpy_feval.__name__
    return CustomMetric(feval, name, allow_extra_outputs)
# pylint: enable=invalid-name
