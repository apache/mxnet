import mxnet as mx
from mxnet import ndarray
from mxnet.metric import check_label_shapes
import numpy as np


class LossRecorder(mx.metric.EvalMetric):
    """

    """
    def __init__(self, name):
        super(LossRecorder, self).__init__(name)

    def update(self, labels, preds=0):
        """
        """
        for loss in labels:
            if isinstance(loss, mx.nd.NDArray):
                loss = loss.asnumpy()
            self.sum_metric += loss.sum()
            self.num_inst += 1

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
            correct = np.logical_and(
                pred_label.flat == label.flat,
                pred_label.flat != self.ignore_label)

            self.sum_metric += correct.sum()
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


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MultiBoxMetric, self).__init__('MultiBox')
        self.eps = eps
        self.num = 2
        self.name = ['CrossEntropy', 'SmoothL1']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        valid_count = np.sum(cls_label >= 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += valid_count

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)
