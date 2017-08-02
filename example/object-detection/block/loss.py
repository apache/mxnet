"""Custom losses for object detection.
Losses are used to penalize incorrect classification and inaccurate box regression.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
from mxnet.gluon import loss


class FocalLoss(loss.Loss):
    """Focal Loss for inbalanced classification.
    Focal loss was described in https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    pending
    """
    def __init__(self, axis=-1, alpha=0.25, gamma=2, sparse_label=True,
                 from_logits=False, batch_axis=0, weight=None, num_class=None,
                 **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._sparse_label = sparse_label
        if sparse_label and (not isinstance(num_class, int) or (num_class < 1)):
            raise ValueError("Number of class > 0 must be provided if sparse label is used.")
        self._num_class = num_class
        self._from_logits = from_logits

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if not self._from_logits:
            output = F.softmax(output)
        if self._sparse_label:
            one_hot = F.one_hot(label, self._num_class)
        else:
            one_hot = label > 0
        pt = F.where(one_hot, output, 1 - output)
        loss = -self._alpha * ((1 - pt) ** self._gamma) * F.log(pt)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
