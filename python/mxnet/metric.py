# coding: utf-8
"""Online evaluation metric module."""
from __future__ import absolute_import

from .base import string_types
import numpy

class EvalMetric(object):
    """Base class of all evaluation metrics."""
    def __init__(self, name):
        self.name = name
        self.reset()

    def update(self, label, pred):
        """Update the internal evaluation.

        Parameters
        ----------
        label : NDArray
            The label of the data.

        pred : NDArray
            Predicted value.
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

    def update(self, label, pred):
        pred = pred.asnumpy()
        label = label.asnumpy().astype('int32')
        pred_label = numpy.argmax(pred, axis=1)
        self.sum_metric += numpy.sum(pred_label == label)
        self.num_inst += label.size


class CustomMetric(EvalMetric):
    """Custom evaluation metric that takes a NDArray function.

    Parameters
    ----------
    feval : callable(label, pred)
        Customized evaluation function.

    name : str, optional
        The name of the metric
    """
    def __init__(self, feval, name=None):
        if name is None:
            name = feval.__name__
            if name.find('<') != -1:
                name = 'custom(%s)' % name
        super(CustomMetric, self).__init__(name)
        self._feval = feval

    def update(self, label, pred):
        self.sum_metric += self._feval(label, pred)
        self.num_inst += 1

# pylint: disable=invalid-name
def np(numpy_feval, name=None):
    """Create a customized metric from numpy function.

    Parameters
    ----------
    numpy_feval : callable(label, pred)
        Customized evaluation function.

    name : str, optional
        The name of the metric.
    """
    def feval(label, pred):
        """Internal eval function."""
        return numpy_feval(label.asnumpy(), pred.asnumpy())
    feval.__name__ = numpy_feval.__name__
    return CustomMetric(feval, name)
# pylint: enable=invalid-name

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
    else:
        raise ValueError('Cannot find metric %s' % metric)
