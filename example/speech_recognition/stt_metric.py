# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import numpy as np

from label_util import LabelUtil
from log_util import LogUtil


def check_label_shapes(labels, preds, shape=0):
    """Check to see if the two arrays are the same size."""

    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))


class STTMetric(mx.metric.EvalMetric):
    def __init__(self, batch_size, num_gpu, is_epoch_end=False, is_logging=True):
        super(STTMetric, self).__init__('STTMetric')

        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.total_n_label = 0
        self.total_l_dist = 0
        self.is_epoch_end = is_epoch_end
        self.total_ctc_loss = 0.
        self.batch_loss = 0.
        self.is_logging = is_logging

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        if self.is_logging:
            log = LogUtil.getInstance().getlogger()
            labelUtil = LabelUtil.getInstance()
        self.batch_loss = 0.

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            seq_length = len(pred) / int(int(self.batch_size) / int(self.num_gpu))

            for i in range(int(int(self.batch_size) / int(self.num_gpu))):
                l = remove_blank(label[i])
                p = []
                for k in range(int(seq_length)):
                    p.append(np.argmax(pred[k * int(int(self.batch_size) / int(self.num_gpu)) + i]))
                p = pred_best(p)

                l_distance = levenshtein_distance(l, p)
                self.total_n_label += len(l)
                self.total_l_dist += l_distance
                this_cer = float(l_distance) / float(len(l))
                if self.is_logging:
                    log.info("label: %s " % (labelUtil.convert_num_to_word(l)))
                    log.info("pred : %s , cer: %f (distance: %d/ label length: %d)" % (
                        labelUtil.convert_num_to_word(p), this_cer, l_distance, len(l)))
                self.num_inst += 1
                self.sum_metric += this_cer
                if self.is_epoch_end:
                    loss = ctc_loss(l, pred, i, int(seq_length), int(self.batch_size), int(self.num_gpu))
                    self.batch_loss += loss
                    if self.is_logging:
                        log.info("loss: %f " % loss)
        self.total_ctc_loss += self.batch_loss

    def get_batch_loss(self):
        return self.batch_loss

    def get_name_value(self):
        try:
            total_cer = float(self.total_l_dist) / float(self.total_n_label)
        except ZeroDivisionError:
            total_cer = float('inf')

        return total_cer, self.total_n_label, self.total_l_dist, self.total_ctc_loss

    def reset(self):
        self.total_n_label = 0
        self.total_l_dist = 0
        self.num_inst = 0
        self.sum_metric = 0.0
        self.total_ctc_loss = 0.0


def pred_best(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i + 1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret


def remove_blank(l):
    ret = []
    for i in range(l.size):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret


def remove_space(l):
    labelUtil = LabelUtil.getInstance()
    ret = []
    for i in range(len(l)):
        if l[i] != labelUtil.get_space_index():
            ret.append(l[i])
    return ret


def ctc_loss(label, prob, remainder, seq_length, batch_size, num_gpu=1, big_num=1e10):
    label_ = [0, 0]
    prob[prob < 1 / big_num] = 1 / big_num
    log_prob = np.log(prob)

    l = len(label)
    for i in range(l):
        label_.append(int(label[i]))
        label_.append(0)

    l_ = 2 * l + 1
    a = np.full((seq_length, l_ + 1), -big_num)
    a[0][1] = log_prob[remainder][0]
    a[0][2] = log_prob[remainder][label_[2]]
    for i in range(1, seq_length):
        row = i * int(batch_size / num_gpu) + remainder
        a[i][1] = a[i - 1][1] + log_prob[row][0]
        a[i][2] = np.logaddexp(a[i - 1][2], a[i - 1][1]) + log_prob[row][label_[2]]
        for j in range(3, l_ + 1):
            a[i][j] = np.logaddexp(a[i - 1][j], a[i - 1][j - 1])
            if label_[j] != 0 and label_[j] != label_[j - 2]:
                a[i][j] = np.logaddexp(a[i][j], a[i - 1][j - 2])
            a[i][j] += log_prob[row][label_[j]]

    return -np.logaddexp(a[seq_length - 1][l_], a[seq_length - 1][l_ - 1])


# label is done with remove_blank
# pred is got from pred_best
def levenshtein_distance(label, pred):
    n_label = len(label) + 1
    n_pred = len(pred) + 1
    if (label == pred):
        return 0
    if (len(label) == 0):
        return len(pred)
    if (len(pred) == 0):
        return len(label)

    v0 = [i for i in range(n_label)]
    v1 = [0 for i in range(n_label)]

    for i in range(len(pred)):
        v1[0] = i + 1

        for j in range(len(label)):
            cost = 0 if label[j] == pred[i] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)

        for j in range(n_label):
            v0[j] = v1[j]

    return v1[len(label)]


def char_match_1way(char_label, char_pred, criteria, n_whole_label):
    n_label = len(char_label)
    n_pred = len(char_pred)

    pred_pos = 0
    accuracy = 0.
    next_accu = 0.
    n_matched = 0.
    next_n_matched = 0.

    for i_index in range(n_label):
        tail_label = n_label - 1 - i_index
        c_label = char_label[i_index]

        for j_index in range(pred_pos, n_pred):
            tail_pred = n_pred - 1 - j_index
            c_pred = char_pred[j_index]

            if tail_label < tail_pred * criteria or tail_pred < tail_label * criteria:
                break
            if c_label == c_pred:
                n_matched += 1.0
                pred_pos = j_index + 1
                break

    accuracy = n_matched / n_whole_label

    if n_label > 0.7 * n_whole_label:
        next_label = char_label[1:]
        next_accu, next_n_matched = char_match_1way(next_label, char_pred, criteria, n_whole_label)

    if next_accu > accuracy:
        accuracy = next_accu
        n_matched = next_n_matched
    return accuracy, n_matched


def char_match_2way(label, pred):
    criterias = [0.98, 0.96, 0.93, 0.9, 0.85, 0.8, 0.7]
    r_pred = pred[::-1]
    r_label = label[::-1]
    n_whole_label = len(remove_space(label))

    val1_max = 0.
    val2_max = 0.
    val1_max_matched = 0.
    val2_max_matched = 0.
    for criteria in criterias:
        val1, val1_matched = char_match_1way(label, pred, criteria, n_whole_label)
        val2, val2_matched = char_match_1way(r_label, r_pred, criteria, n_whole_label)

        if val1 > val1_max:
            val1_max = val1
            val1_max_matched = val1_matched
        if val2 > val2_max:
            val2_max = val2
            val2_max_matched = val2_matched

    val = val1_max if val1_max > val2_max else val2_max
    val_matched = val1_max_matched if val1_max > val2_max else val2_max_matched
    return val, val_matched, n_whole_label
