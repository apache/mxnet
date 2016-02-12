# coding: utf-8
# pylint: disable=no-member

"""Online evaluation metric module."""
from __future__ import absolute_import
from . import ndarray
import numpy

def check_label_shapes(labels, preds, shape=0):
    """Check to see if the two arrays are the same size."""

    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

class EvalMetric(object):
    """Base class of all evaluation metrics."""

    def __init__(self, name):
        self.name = name
        self.reset()

    def update(self, label, pred):
        """Update the internal evaluation.

        Parameters
        ----------
        labels : list of NDArray
            The labels of the data.

        preds : list of NDArray
            Predicted values.
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
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

########################
# CLASSIFICATION METRICS
########################

class Accuracy(EvalMetric):
    """Calculate accuracy"""

    def __init__(self):
        super(Accuracy, self).__init__('accuracy')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred_label = ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)

class F1(EvalMetric):
    """Calculate the F1 score of a binary classification problem."""

    def __init__(self):
        super(F1, self).__init__('f1')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for i in range(len(labels)):
            pred = preds[i].asnumpy()
            label = labels[i].asnumpy().astype('int32')
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

####################
# REGRESSION METRICS
####################

class MAE(EvalMetric):
    """Calculate Mean Absolute Error loss"""

    def __init__(self):
        super(MAE, self).__init__('mae')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            check_label_shapes(label, pred, shape=1)

            self.sum_metric += numpy.abs(label - pred).sum()
            self.num_inst += numpy.prod(label.shape)

class MSE(EvalMetric):
    """Calculate Mean Squared Error loss"""
    def __init__(self):
        super(MSE, self).__init__('mse')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            check_label_shapes(label, pred, shape=1)

            self.sum_metric += ((label - pred)**2.0).mean()
            self.num_inst += numpy.prod(label.shape)

class RMSE(EvalMetric):
    """Calculate Root Mean Squred Error loss"""
    def __init__(self):
        super(RMSE, self).__init__('rmse')

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            check_label_shapes(label, pred, shape=1)

            self.sum_metric += numpy.sqrt(((label - pred)**2.0).mean())
        self.num_inst += 1

class Torch(EvalMetric):
    """Dummy metric for torch criterions"""
    def __init__(self):
        super(Torch, self).__init__('torch')

    def update(self, _, preds):
        for pred in preds:
            self.sum_metric += pred.asnumpy().mean()
        self.num_inst += 1

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

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for pred, label in zip(preds, labels):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if pred.shape[1] == 2:
                pred = pred[:, 1]

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
        return numpy_feval(label, pred)
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

    metrics = {
        'accuracy' : Accuracy(),
        'f1' : F1(),
        'acc' : Accuracy(),
        'rmse' : RMSE(),
        'mae' : MAE(),
        'mse' : MSE()
    }

    if callable(metric):
        return CustomMetric(metric)
    try:
        return metrics[metric.lower()]
    except:
        raise ValueError("Metric must be either callable or in {}".format(
            metrics.keys()))
