"""Custom losses for object detection.
Losses are used to penalize incorrect classification and inaccurate box regression.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
from mxnet.gluon import loss
import numpy as np

def find_inf(x, mark='null'):
    pos = np.where(x.asnumpy().flat == np.inf)[0]
    print(mark, pos)

class FocalLoss(loss.Loss):
    """Focal Loss for inbalanced classification.
    Focal loss was described in https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    pending
    """
    def __init__(self, axis=-1, alpha=0.25, gamma=2, sparse_label=True,
                 from_logits=False, batch_axis=0, weight=None, num_class=None,
                 eps=1e-12, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._sparse_label = sparse_label
        if sparse_label and (not isinstance(num_class, int) or (num_class < 1)):
            raise ValueError("Number of class > 0 must be provided if sparse label is used.")
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps

    def hybrid_forward(self, F, output, label, sample_weight=None):
        find_inf(output, 'output')
        backup = output.asnumpy()
        if not self._from_logits:
            output = F.sigmoid(output)
        if self._sparse_label:
            one_hot = F.one_hot(label, self._num_class)
        else:
            one_hot = label > 0
        pt = F.where(one_hot, output, 1 - output)
        print('pt', pt)
        find_inf(pt, 'pt')
        t = F.ones_like(one_hot)
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        find_inf(alpha, 'alpha')
        tmp1 = (1-pt) ** self._gamma
        print('tmp1',tmp1)
        find_inf(tmp1, 'tmp1')
        tmp2 = F.log(pt)
        find_inf(tmp2, 'tmp2')
        tmp3 = -alpha * tmp1
        find_inf(tmp3, 'tmp3')
        tmp4 = tmp1 * tmp2
        find_inf(tmp4, 'tmp4')
        loss = -alpha * ((1 - pt) ** self._gamma) * F.log(F.minimum(pt + self._eps, 1))
        # print('pt again', pt)
        # find_inf(loss, 'loss')
        # import numpy as np
        # np.set_printoptions(threshold=np.inf)
        # temp = loss.asnumpy()
        # pos = np.where(temp.flat == np.inf)[0]
        # print(temp.dtype)
        # print(alpha.asnumpy().flat[pos], 'alpha pos')
        # print(tmp2.asnumpy().flat[pos], 'log results pos')
        # print(tmp1.asnumpy().flat[pos], 'power')
        # print(pt.asnumpy().flat[pos], 'in pt')
        # print(output.asnumpy().flat[pos], 'in output')
        # print(backup.flat[pos], 'in backup')
        # print(pos)
        # print(temp.flat[pos])
        # print(np.sum(temp))
        # print(F.sum(loss,axis=self._batch_axis, exclude=True))
        # raise
        return F.mean(loss, axis=self._batch_axis, exclude=True)
