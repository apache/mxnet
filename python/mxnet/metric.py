# coding: utf-8
# pylint: disable=no-member

"""Online evaluation metric module."""
from __future__ import absolute_import
import math
import numpy
from . import ndarray

def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.
    """

    def __init__(self, name, num=None):
        self.name = name
        self.num = num
        self.reset()

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
        if self.num is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s_%d'%(self.name, i) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

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
        return zip(name, value)

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))


class CompositeEvalMetric(EvalMetric):
    """Manages multiple evaluation metrics.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> eval_metrics_1 = mx.metric.Accuracy()
    >>> eval_metrics_2 = mx.metric.F1()
    >>> eval_metrics = mx.metric.CompositeEvalMetric()
    >>> for child_metric in [eval_metrics_1, eval_metrics_2]:
    >>>     eval_metrics.add(child_metric)
    >>> eval_metrics.update(labels = labels, preds = predicts)
    >>> print eval_metrics.get()
    (['accuracy', 'f1'], [0.6666666666666666, 0.8])
    """

    def __init__(self, **kwargs):
        super(CompositeEvalMetric, self).__init__('composite')
        try:
            self.metrics = kwargs['metrics']
        except KeyError:
            self.metrics = []

    def add(self, metric):
        """Adds a child metric.

        Parameters
        ----------
        metric
            A metric instance.
        """
        self.metrics.append(metric)

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
        results = []
        for metric in self.metrics:
            result = metric.get()
            names.append(result[0])
            results.append(result[1])
        return (names, results)

########################
# CLASSIFICATION METRICS
########################

class Accuracy(EvalMetric):
    """Computes accuracy classification score.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> acc = mx.metric.Accuracy()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """

    def __init__(self):
        super(Accuracy, self).__init__('accuracy')

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax_channel(pred_label)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

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

    Examples
    --------
    >>> np.random.seed(999)
    >>> top_k = 3
    >>> labels = [mx.nd.array([2, 6, 9, 2, 3, 4, 7, 8, 9, 6])]
    >>> predicts = [mx.nd.array(np.random.rand(10, 10))]
    >>> acc = mx.metric.TopKAccuracy(top_k=top_k)
    >>> acc.update(labels, predicts)
    >>> print acc.get()
    ('top_k_accuracy', 0.3)
    """

    def __init__(self, **kwargs):
        super(TopKAccuracy, self).__init__('top_k_accuracy')
        try:
            self.top_k = kwargs['top_k']
        except KeyError:
            self.top_k = 1
        assert(self.top_k > 1), 'Please use Accuracy if top_k is no more than 1'
        self.name += '_%d' % self.top_k

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            assert(len(pred_label.shape) <= 2), 'Predictions should be no more than 2 dims'
            pred_label = numpy.argsort(pred_label.asnumpy().astype('float32'), axis=1)
            label = label.asnumpy().astype('int32')
            check_label_shapes(label, pred_label)
            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric += (pred_label.flat == label.flat).sum()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    self.sum_metric += (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
            self.num_inst += num_samples

class F1(EvalMetric):
    """Computes the F1 score of a binary classification problem.

    The F1 score is equvalent to weighted average of the precision and recall,
    where the best value is 1.0 and the worst value is 0.0. The formula for F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    The formula for precision and recall is::

        precision = true_positives / (true_positives + false_positives)
        recall    = true_positives / (true_positives + false_negatives)

    .. note::

        This F1 score only supports binary classification.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0., 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0., 1., 1.])]
    >>> acc = mx.metric.F1()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('f1', 0.8)
    """

    def __init__(self):
        super(F1, self).__init__('f1')

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            pred = pred.asnumpy()
            label = label.asnumpy().astype('int32')
            pred_label = numpy.argmax(pred, axis=1)

            check_label_shapes(label, pred)
            if len(numpy.unique(label)) > 2:
                raise ValueError("F1 currently only supports binary classification.")

            true_positives, false_positives, false_negatives = 0., 0., 0.

            for y_pred, y_true in zip(pred_label, label):
                if y_pred == 1 and y_true == 1:
                    true_positives += 1.
                elif y_pred == 1 and y_true == 0:
                    false_positives += 1.
                elif y_pred == 0 and y_true == 1:
                    false_negatives += 1.

            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0.

            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.

            if precision + recall > 0:
                f1_score = 2 * precision * recall / (precision + recall)
            else:
                f1_score = 0.

            self.sum_metric += f1_score
            self.num_inst += 1


class Perplexity(EvalMetric):
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
    ignore_label : int or None
        Index of invalid label to ignore when
        counting. By default, sets to -1.
        If set to `None`, it will include all entries.
    axis : int (default -1)
        The axis from prediction that was used to
        compute softmax. By default use the last
        axis.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> perp = mx.metric.Perplexity(ignore_label=None)
    >>> perp.update(labels, predicts)
    >>> print perp.get()
    ('Perplexity', 1.7710976285155853)
    """
    def __init__(self, ignore_label, axis=-1):
        super(Perplexity, self).__init__('Perplexity')
        self.ignore_label = ignore_label
        self.axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        assert len(labels) == len(preds)
        loss = 0.
        num = 0
        for label, pred in zip(labels, preds):
            assert label.size == pred.size/pred.shape[-1], \
                "shape mismatch: %s vs. %s"%(label.shape, pred.shape)
            label = label.as_in_context(pred.context).reshape((label.size,))
            pred = ndarray.pick(pred, label.astype(dtype='int32'), axis=self.axis)
            if self.ignore_label is not None:
                ignore = label == self.ignore_label
                num -= ndarray.sum(ignore).asscalar()
                pred = pred*(1-ignore) + ignore
            loss -= ndarray.sum(ndarray.log(ndarray.maximum(1e-10, pred))).asscalar()
            num += pred.size
        self.sum_metric += loss
        self.num_inst += num

    def get(self):
        """Returns the current evaluation result.

        Returns
        -------
        Tuple of (str, float)
            Representing name of the metric and evaluation result.
        """
        return (self.name, math.exp(self.sum_metric/self.num_inst))

####################
# REGRESSION METRICS
####################

class MAE(EvalMetric):
    """Computes Mean Absolute Error (MAE) loss.

    The mean absolute error is given by

    .. math::
        \\frac{\\sum_i^n |y_i - \\hat{y}_i|}{n}

    Examples
    --------
    >>> predicts = [mx.nd.array(np.array([3, -0.5, 2, 7]).reshape(4,1))]
    >>> labels = [mx.nd.array(np.array([2.5, 0.0, 2, 8]).reshape(4,1))]
    >>> mean_absolute_error = mx.metric.MAE()
    >>> mean_absolute_error.update(labels = labels, preds = predicts)
    >>> print mean_absolute_error.get()
    ('mae', 0.5)
    """

    def __init__(self):
        super(MAE, self).__init__('mae')

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            self.sum_metric += numpy.abs(label - pred).mean()
            self.num_inst += 1 # numpy.prod(label.shape)


class MSE(EvalMetric):
    """Computes Mean Squared Error (MSE) loss.

    The mean squared error is given by

    .. math::
        \\frac{\\sum_i^n (y_i - \\hat{y}_i)^2}{n}

    Examples
    --------
    >>> predicts = [mx.nd.array(np.array([3, -0.5, 2, 7]).reshape(4,1))]
    >>> labels = [mx.nd.array(np.array([2.5, 0.0, 2, 8]).reshape(4,1))]
    >>> mean_squared_error = mx.metric.MSE()
    >>> mean_squared_error.update(labels = labels, preds = predicts)
    >>> print mean_squared_error.get()
    ('mse', 0.375)
    """
    def __init__(self):
        super(MSE, self).__init__('mse')

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            self.sum_metric += ((label - pred)**2.0).mean()
            self.num_inst += 1 # numpy.prod(label.shape)

class RMSE(EvalMetric):
    """Computes Root Mean Squred Error (RMSE) loss.

    The root mean squared error is given by

    .. math::
        \\sqrt{\\frac{\\sum_i^n (y_i - \\hat{y}_i)^2}{n}}

    Examples
    --------
    >>> predicts = [mx.nd.array(np.array([3, -0.5, 2, 7]).reshape(4,1))]
    >>> labels = [mx.nd.array(np.array([2.5, 0.0, 2, 8]).reshape(4,1))]
    >>> root_mean_squared_error = mx.metric.RMSE()
    >>> root_mean_squared_error.update(labels = labels, preds = predicts)
    >>> print root_mean_squared_error.get()
    ('rmse', 0.612372457981)
    """
    def __init__(self):
        super(RMSE, self).__init__('rmse')

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            self.sum_metric += numpy.sqrt(((label - pred)**2.0).mean())
            self.num_inst += 1

class CrossEntropy(EvalMetric):
    """Computes Cross Entropy loss.

    The cross entropy is given by

    .. math::
        -y\\log \\hat{y} + (1-y)\\log (1-\\hat{y})

    Parameters
    ----------
    eps : float
        Cross Entropy loss is undefined for predicted value is 0 or 1,
        so predicted values are added with the small constant.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> ce = mx.metric.CrossEntropy()
    >>> ce.update(labels, predicts)
    >>> print ce.get()
    ('cross-entropy', 0.57159948348999023)
    """
    def __init__(self, eps=1e-8):
        super(CrossEntropy, self).__init__('cross-entropy')
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
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[numpy.arange(label.shape[0]), numpy.int64(label)]
            self.sum_metric += (-numpy.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]

class Torch(EvalMetric):
    """Dummy metric for torch criterions."""
    def __init__(self, name='torch'):
        super(Torch, self).__init__(name)

    def update(self, _, preds):
        for pred in preds:
            self.sum_metric += pred.asnumpy().mean()
        self.num_inst += 1

class Caffe(Torch):
    """Dummy metric for caffe criterions"""
    def __init__(self):
        super(Caffe, self).__init__('caffe')

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

    Examples
    --------
    >>> predicts = [mx.nd.array(np.array([3, -0.5, 2, 7]).reshape(4,1))]
    >>> labels = [mx.nd.array(np.array([2.5, 0.0, 2, 8]).reshape(4,1))]
    >>> feval = lambda x, y : (x + y).mean()
    >>> eval_metrics = mx.metric.CustomMetric(feval=feval)
    >>> eval_metrics.update(labels, predicts)
    >>> print eval_metrics.get()
    ('custom(<lambda>)', 6.0)
    """
    def __init__(self, feval, name=None, allow_extra_outputs=False):
        if name is None:
            name = feval.__name__
            if name.find('<') != -1:
                name = 'custom(%s)' % name
        super(CustomMetric, self).__init__(name)
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
            check_label_shapes(labels, preds)

        for pred, label in zip(preds, labels):
            label = label.asnumpy()
            pred = pred.asnumpy()

            reval = self._feval(label, pred)
            if isinstance(reval, tuple):
                (sum_metric, num_inst) = reval
                self.sum_metric += sum_metric
                self.num_inst += num_inst
            else:
                self.sum_metric += reval
                self.num_inst += 1

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
    >>> metric = mx.metric.np(custom_metric)
    """
    def feval(label, pred):
        """Internal eval function."""
        return numpy_feval(label, pred)
    feval.__name__ = numpy_feval.__name__
    return CustomMetric(feval, name, allow_extra_outputs)
# pylint: enable=invalid-name

def create(metric, **kwargs):
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

    Examples
    --------
    >>> def custom_metric(label, pred):
    ...     return np.mean(np.abs(label - pred))
    ...
    >>> metric1 = mx.metric.create('acc')
    >>> metric2 = mx.metric.create(custom_metric)
    >>> metric3 = mx.metric.create([metric1, metric2, 'rmse'])
    """

    if callable(metric):
        return CustomMetric(metric)
    elif isinstance(metric, EvalMetric):
        return metric
    elif isinstance(metric, list):
        composite_metric = CompositeEvalMetric()
        for child_metric in metric:
            composite_metric.add(create(child_metric, **kwargs))
        return composite_metric

    metrics = {
        'acc': Accuracy,
        'accuracy': Accuracy,
        'ce': CrossEntropy,
        'f1': F1,
        'mae': MAE,
        'mse': MSE,
        'rmse': RMSE,
        'top_k_accuracy': TopKAccuracy
    }

    try:
        return metrics[metric.lower()](**kwargs)
    except:
        raise ValueError("Metric must be either callable or in {}".format(
            metrics.keys()))
