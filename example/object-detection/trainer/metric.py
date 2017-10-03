import mxnet as mx
from mxnet import ndarray
from mxnet.metric import check_label_shapes
import numpy as np


class Accuracy(mx.metric.EvalMetric):
    """

    """
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None,
                 ignore_label=-1):
        super(Accuracy, self).__init__(
            name, axis=axis,
            output_names=output_names,
            label_names=label_names)
        self.axis = axis
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        """

        """
        check_label_shapes(labels, preds)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += np.sum(label != self.ignore_label)


class SmoothL1(mx.metric.EvalMetric):
    """
    """
    def __init__(self, name='smoothl1', output_names=None, label_names=None):
        super(SmoothL1, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """

        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            smoothl1 = ndarray.smooth_l1(label - pred, scalar=1.0).asnumpy()
            label = label.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            self.sum_metric += np.sum(smoothl1)
            self.num_inst += np.sum(label > 0) # numpy.prod(label.shape)
