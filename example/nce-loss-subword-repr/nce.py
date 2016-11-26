# pylint:skip-file
import sys
import mxnet as mx
import numpy as np
from operator import itemgetter


def nce_loss(data, label, label_mask, label_weight, embed_weight, vocab_size, num_hidden, num_label):
    label_units_embed = mx.sym.Embedding(data = label, input_dim = vocab_size,
                                         weight = embed_weight,
                                         output_dim = num_hidden)
    label_units_embed = mx.sym.broadcast_mul(lhs = label_units_embed, rhs = label_mask,
                                             name = 'label_units_embed')
    label_units_embed = mx.sym.sum(label_units_embed, axis=2)

    label_embed = mx.sym.SliceChannel(data = label_units_embed,
                                      num_outputs = num_label,
                                      squeeze_axis = 1, name = 'label_slice')
    label_weight = mx.sym.SliceChannel(data = label_weight,
                                       num_outputs = num_label,
                                       squeeze_axis = 1)
    preds = []
    for i in range(num_label):
        vec = label_embed[i]
        vec = vec * data
        vec = mx.sym.sum(vec, axis = 1)
        pred = mx.sym.LogisticRegressionOutput(data = vec,
                                               label = label_weight[i])
        preds.append(pred)
    return preds


class NceAuc(mx.metric.EvalMetric):
    def __init__(self):
        super(NceAuc, self).__init__('nce-auc')

    def update(self, labels, preds):
        label_weight = labels[1].asnumpy()
        preds = np.array([x.asnumpy() for x in preds]).transpose()
        tmp = []
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                tmp.append((label_weight[i][j], preds[i][j]))
        tmp = sorted(tmp, key = itemgetter(1), reverse = True)
        m = 0.0
        n = 0.0
        z = 0.0
        k = 0
        for a, b in tmp:
            if a > 0.5:
                m += 1.0
                z += len(tmp) - k
            else:
                n += 1.0
            k += 1
        z -= m * (m + 1.0) / 2.0
        z /= m
        z /= n
        self.sum_metric += z
        self.num_inst += 1