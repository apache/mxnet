# pylint: disable=invalid-name
"""Online evaluation metric module."""
from .base import string_types
import numpy as np

class EvalMetric(object):
    """Base class of all evaluation metrics."""
    def __init__(self, name):
        self.name = name
        self.reset()

    def update(self, pred, label):
        """Update the internal evaluation.

        Parameters
        ----------
        pred : NDArray
            Predicted value.

        label : NDArray
            The label of the data.
        """
        raise NotImplementedError()

    def reset(self):
        """Clear the internal statistics to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        return (self.name, self.sum_metric / self.num_inst)


class Accuracy(EvalMetric):
    """Calculate accuracy"""
    def __init__(self):
        super(Accuracy, self).__init__('accuracy')

    def update(self, pred, label):
        pred = pred.asnumpy()
        label = label.asnumpy().astype('int32')
        py = np.argmax(pred, axis=1)
        self.sum_metric += np.sum(py == label)
        self.num_inst += label.size

class LogLoss(EvalMetric):
    """Calculate logloss"""
    def __init__(self):
        self.eps = 1e-15
        super(LogLoss, self).__init__('logloss')

    def update(self, label, pred):
        pred = pred.asnumpy()
        label = label.asnumpy().astype('int32')
        for i in range(label.size):
            p = pred[i][label[i]]
            p = max(min(p, 1 - self.eps), self.eps)
            self.sum_metric += -np.log(p)
            self.num_inst += label.size

class CustomMetric(EvalMetric):
    """Calculate accuracy"""
    def __init__(self, feval):
        name = feval.__name__
        if name.find('<') != -1:
            name = 'custom(%s)' % name
        super(CustomMetric, self).__init__(name)
        self._feval = feval

    def update(self, pred, label):
        self.sum_metric += self._feval(pred, label)
        self.num_inst += 1


def create(metric):
    """Create an evaluation metric.

    Parameters
    ----------
    metric : str or callable
        The name of the metric, or a function
        providing statistics given pred, label NDArray.
    """
    if callable(metric):
        return CustomMetric(metric)
    if not isinstance(metric, string_types):
        raise TypeError('metric should either be callable or str')
    if metric == 'acc' or metric == 'accuracy':
        return Accuracy()
    elif metric == 'logloss':
        return LogLoss()
    else:
        raise ValueError('Cannot find metric %s' % metric)
