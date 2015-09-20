# pylint: disable=invalid-name
"""Online evaluation metric module."""
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
        y = np.argmax(pred, axis=1)
        self.sum_metric += np.sum(y == label)
        self.num_inst += label.size


def create(name):
    """Create an evaluation metric.

    Parameters
    ----------
    name : str
        The name of the metric
    """
    if name == 'acc' or name == 'accuracy':
        return Accuracy()
    else:
        raise ValueError('Cannot find metric %s' % name)
