import mxnet as mx
import numpy as np


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self):
        super(MultiBoxMetric, self).__init__(['Acc', 'ObjectAcc', 'SmoothL1'], 3)

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        mask = np.where(label >= 0)[0]
        p = np.argmax(cls_prob, axis=1).flatten()
        self.sum_metric[0] += np.sum(p[mask] == label[mask])
        self.num_inst[0] += mask.size
        mask = np.where(label > 0)[0]
        self.sum_metric[1] += np.sum(p[mask] == label[mask])
        self.num_inst[1] += mask.size
        # smoothl1loss
        self.sum_metric[2] += np.sum(loc_loss)
        self.num_inst[2] += loc_loss.shape[0]

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
