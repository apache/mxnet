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
from mxnet.test_utils import use_np
import numpy as np
import scipy
from scipy.stats import pearsonr
import json
import math
from common import xfail_when_nonstandard_decimal_separator
from copy import deepcopy

def check_metric(metric, *args, **kwargs):
    metric = mx.gluon.metric.create(metric, *args, **kwargs)
    str_metric = json.dumps(metric.get_config())
    metric2 = mx.gluon.metric.create(str_metric)

    assert metric.get_config() == metric2.get_config()

def test_metrics():
    check_metric('acc', axis=0)
    check_metric('f1')
    check_metric('mcc')
    check_metric('perplexity', axis=-1)
    check_metric('pearsonr')
    check_metric('pcc')
    check_metric('ce')
    check_metric('loss')
    composite = mx.gluon.metric.create(['acc', 'f1'])
    check_metric(composite)

def test_ce():
    metric = mx.gluon.metric.create('ce')
    pred = mx.nd.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    label = mx.nd.array([2, 1])
    metric.update([label], [pred])
    _, loss = metric.get()
    expected_loss = -(np.log(pred[0][2].asscalar()) + np.log(pred[1][1].asscalar())) / 2
    assert loss == expected_loss
    metric = mx.gluon.metric.create('ce', from_logits=True)
    pred = mx.nd.log(pred)
    metric.update([label], [pred])
    _, loss = metric.get()
    np.testing.assert_almost_equal(loss, expected_loss)


def test_acc():
    pred = mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])
    label = mx.nd.array([0, 1, 1])
    metric = mx.gluon.metric.create('acc')
    metric.update([label], [pred])
    _, acc = metric.get()
    expected_acc = (np.argmax(pred, axis=1) == label).sum().asscalar() / label.size
    np.testing.assert_almost_equal(acc, expected_acc)

def test_acc_2d_label():
    # label maybe provided in 2d arrays in custom data iterator
    pred = mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6], [0.8, 0.2], [0.3, 0.5], [0.6, 0.4]])
    label = mx.nd.array([[0, 1, 1], [1, 0, 1]])
    metric = mx.gluon.metric.create('acc')
    metric.update([label], [pred])
    _, acc = metric.get()
    expected_acc = (np.argmax(pred, axis=1).asnumpy() == label.asnumpy().ravel()).sum() / \
                   float(label.asnumpy().ravel().size)
    np.testing.assert_almost_equal(acc, expected_acc)

def test_loss_update():
    pred = mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])
    metric1 = mx.gluon.metric.create('loss')
    metric2 = mx.gluon.metric.create('loss')
    metric1.update(None, [pred])
    metric2.update(None, pred)
    _, acc1 = metric1.get()
    _, acc2 = metric2.get()
    assert acc1 == acc2

@xfail_when_nonstandard_decimal_separator
def test_binary_f1():
    microF1 = mx.gluon.metric.create("f1", average="micro")
    macroF1 = mx.gluon.metric.F1(average="macro")

    assert np.isnan(macroF1.get()[1])
    assert np.isnan(microF1.get()[1])

    # check divide by zero
    pred = mx.nd.array([[0.9, 0.1],
                        [0.8, 0.2]])
    label = mx.nd.array([0, 0])
    macroF1.update([label], [pred])
    microF1.update([label], [pred])
    assert macroF1.get()[1] == 0.0
    assert microF1.get()[1] == 0.0
    macroF1.reset()
    microF1.reset()

    pred11 = mx.nd.array([[0.1, 0.9],
                          [0.5, 0.5]])
    label11 = mx.nd.array([1, 0])
    pred12 = mx.nd.array([[0.85, 0.15],
                          [1.0, 0.0]])
    label12 = mx.nd.array([1, 0])
    pred21 = mx.nd.array([[0.6, 0.4]])
    label21 = mx.nd.array([0])
    pred22 = mx.nd.array([[0.2, 0.8]])
    label22 = mx.nd.array([1])

    microF1.update([label11, label12], [pred11, pred12])
    macroF1.update([label11, label12], [pred11, pred12])
    assert microF1.num_inst == 4
    assert macroF1.num_inst == 4
    # f1 = 2 * tp / (2 * tp + fp + fn)
    fscore1 = 2. * (1) / (2 * 1 + 1 + 0)
    np.testing.assert_almost_equal(microF1.get()[1], fscore1)
    np.testing.assert_almost_equal(macroF1.get()[1], fscore1)

    microF1.update([label21, label22], [pred21, pred22])
    macroF1.update([label21, label22], [pred21, pred22])
    assert microF1.num_inst == 6
    assert macroF1.num_inst == 6
    fscore2 = 2. * (1) / (2 * 1 + 0 + 0)
    fscore_total = 2. * (1 + 1) / (2 * (1 + 1) + (1 + 0) + (0 + 0))
    np.testing.assert_almost_equal(microF1.get()[1], fscore_total)
    np.testing.assert_almost_equal(macroF1.get()[1], fscore_total)

def test_multiclass_f1():
    microF1 = mx.gluon.metric.create("f1", class_type="multiclass", average="micro")
    macroF1 = mx.gluon.metric.F1(class_type="multiclass", average="macro")

    assert np.isnan(macroF1.get()[1])
    assert np.isnan(microF1.get()[1])

    # check one class is zero
    pred = mx.nd.array([[0.9, 0.1],
                        [0.8, 0.2]])
    label = mx.nd.array([0, 0])
    macroF1.update([label], [pred])
    microF1.update([label], [pred])
    assert macroF1.get()[1] == 0.5 # one class is 1.0, the other is 0. (divided by 0)
    assert microF1.get()[1] == 1.0 # globally f1 is 1.0
    macroF1.reset()
    microF1.reset()

    # test case from sklearn, here pred is probabilistic distributions instead of predicted labels
    pred11 = mx.nd.array([[1, 0, 0], [0, 1, 0]])
    label11 = mx.nd.array([0, 2])
    pred12 = mx.nd.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    label12 = mx.nd.array([1, 0, 0, 1])

    microF1.update([label11, label12], [pred11, pred12])
    macroF1.update([label11, label12], [pred11, pred12])
    assert microF1.num_inst == 6
    assert macroF1.num_inst == 6

    # from sklearn.metrics import f1_score
    # overall_pred = [0, 1, 2, 0, 1, 2]
    # overall_label = [0, 2, 1, 0, 0, 1]
    fmacro = 0.26666666666666666 #f1_score(overall_label, overall_pred, average="macro")
    fmicro = 0.3333333333333333 #f1_score(overall_label, overall_pred, average="micro")
    np.testing.assert_almost_equal(microF1.get()[1], fmicro)
    np.testing.assert_almost_equal(macroF1.get()[1], fmacro)

@xfail_when_nonstandard_decimal_separator
def test_multilabel_f1():
    microF1 = mx.gluon.metric.create("f1", class_type="multilabel", average="micro")
    macroF1 = mx.gluon.metric.F1(class_type="multilabel", average="macro")

    assert np.isnan(macroF1.get()[1])
    assert np.isnan(microF1.get()[1])

    # check one class is zero
    pred = mx.nd.array([[0.9, 0.1],
                        [0.8, 0.2]])
    label = mx.nd.array([[1, 1], [1, 1]])
    macroF1.update([label], [pred])
    microF1.update([label], [pred])
    assert macroF1.get()[1] == 0.5 # one class is 1.0, the other is 0. (divided by 0)
    np.testing.assert_almost_equal(microF1.get()[1], 2.0 / 3)
    macroF1.reset()
    microF1.reset()

    pred11 = mx.nd.array([[0.9, 0.4, 0.3], [0.2, 0.7, 0.8]])
    label11 = mx.nd.array([[1, 0, 1], [0, 0, 1]])
    pred12 = mx.nd.array([[0.6, 0.6, 0.7]])
    label12 = mx.nd.array([[0, 1, 1]])

    microF1.update([label11, label12], [pred11, pred12])
    macroF1.update([label11, label12], [pred11, pred12])
    assert microF1.num_inst == 3
    assert macroF1.num_inst == 3
    #from sklearn.metrics import f1_score
    #overall_pred = [[1, 0, 0], [0, 1, 1], [1, 1, 1]]
    #overall_label = [[1, 0, 1], [0, 0, 1], [0, 1, 1]]
    fmacro = 0.7111111111111111  #f1_score(overall_label, overall_pred, average="macro")
    fmicro = 0.7272727272727272  #f1_score(overall_label, overall_pred, average="micro")
    np.testing.assert_almost_equal(microF1.get()[1], fmicro)
    np.testing.assert_almost_equal(macroF1.get()[1], fmacro)

@xfail_when_nonstandard_decimal_separator
def test_mcc():
    microMCC = mx.gluon.metric.create("mcc")

    assert np.isnan(microMCC.get()[1])

    # check divide by zero
    pred = mx.nd.array([[0.9, 0.1],
                        [0.8, 0.2]])
    label = mx.nd.array([0, 0])
    microMCC.update([label], [pred])
    assert microMCC.get()[1] == 0.0
    microMCC.reset()

    pred11 = mx.nd.array([[0.1, 0.9],
                        [0.5, 0.5]])
    label11 = mx.nd.array([1, 0])
    pred12 = mx.nd.array([[0.85, 0.15],
                        [1.0, 0.0]])
    label12 = mx.nd.array([1, 0])
    pred21 = mx.nd.array([[0.6, 0.4]])
    label21 = mx.nd.array([0])
    pred22 = mx.nd.array([[0.2, 0.8]])
    label22 = mx.nd.array([1])
    microMCC.update([label11, label12], [pred11, pred12])
    assert microMCC.num_inst == 4
    tp1 = 1; fp1 = 0; fn1 = 1; tn1=2
    mcc1 = (tp1*tn1 - fp1*fn1) / np.sqrt((tp1+fp1)*(tp1+fn1)*(tn1+fp1)*(tn1+fn1))
    np.testing.assert_almost_equal(microMCC.get()[1], mcc1)

    microMCC.update([label21, label22], [pred21, pred22])
    assert microMCC.num_inst == 6
    tp2 = 1; fp2 = 0; fn2 = 0; tn2=1
    mcc2 = (tp2*tn2 - fp2*fn2) / np.sqrt((tp2+fp2)*(tp2+fn2)*(tn2+fp2)*(tn2+fn2))
    tpT = tp1+tp2; fpT = fp1+fp2; fnT = fn1+fn2; tnT = tn1+tn2;
    mccT = (tpT*tnT - fpT*fnT) / np.sqrt((tpT+fpT)*(tpT+fnT)*(tnT+fpT)*(tnT+fnT))
    np.testing.assert_almost_equal(microMCC.get()[1], mccT)

def test_perplexity():
    pred = mx.nd.array([[0.8, 0.2], [0.2, 0.8], [0, 1.]])
    label = mx.nd.array([0, 1, 1])
    p = pred.asnumpy()[np.arange(label.size), label.asnumpy().astype('int32')]
    perplexity_expected = np.exp(-np.log(p).sum()/label.size)
    metric = mx.gluon.metric.create('perplexity', axis=-1)
    metric.update([label], [pred])
    _, perplexity = metric.get()
    np.testing.assert_almost_equal(perplexity, perplexity_expected)

def test_pearsonr():
    pred1 = mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])
    label1 = mx.nd.array([[1, 0], [0, 1], [0, 1]])
    pearsonr_expected_np = np.corrcoef(pred1.asnumpy().ravel(), label1.asnumpy().ravel())[0, 1]
    pearsonr_expected_scipy, _ = pearsonr(pred1.asnumpy().ravel(), label1.asnumpy().ravel())
    micro_pr = mx.gluon.metric.create('pearsonr')

    assert np.isnan(micro_pr.get()[1])

    micro_pr.update([label1], [pred1])

    np.testing.assert_almost_equal(micro_pr.get()[1], pearsonr_expected_np)
    np.testing.assert_almost_equal(micro_pr.get()[1], pearsonr_expected_scipy)

    pred2 = mx.nd.array([[1, 2], [3, 2], [4, 6]])
    label2 = mx.nd.array([[1, 0], [0, 1], [0, 1]])
    # Note that pred12 = pred1 + pred2; label12 = label1 + label2
    pred12 = mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6],[1, 2], [3, 2], [4, 6]])
    label12 = mx.nd.array([[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1]])

    pearsonr_expected_np = np.corrcoef(pred12.asnumpy().ravel(), label12.asnumpy().ravel())[0, 1]
    pearsonr_expected_scipy, _ = pearsonr(pred12.asnumpy().ravel(), label12.asnumpy().ravel())

    micro_pr.update([label2], [pred2])
    np.testing.assert_almost_equal(micro_pr.get()[1], pearsonr_expected_np)
    np.testing.assert_almost_equal(micro_pr.get()[1], pearsonr_expected_scipy)

def cm_batch(cm):
    # generate a batch yielding a given confusion matrix
    n = len(cm)
    ident = np.identity(n)
    labels = []
    preds = []
    for i in range(n):
        for j in range(n):
            labels += [ i ] * cm[i][j]
            preds += [ ident[j] ] * cm[i][j]
    return ([ mx.nd.array(labels, dtype='int32') ], [ mx.nd.array(preds) ])

def test_pcc():
    labels, preds = cm_batch([
        [ 7, 3 ],
        [ 2, 5 ],
    ])
    met_pcc = mx.gluon.metric.create('pcc')
    met_pcc.update(labels, preds)
    _, pcc = met_pcc.get()

    # pcc should agree with mcc for binary classification
    met_mcc = mx.gluon.metric.create('mcc')
    met_mcc.update(labels, preds)
    _, mcc = met_mcc.get()
    np.testing.assert_almost_equal(pcc, mcc)

    # pcc should agree with Pearson for binary classification
    met_pear = mx.gluon.metric.create('pearsonr')
    met_pear.update(labels, [p.argmax(axis=1) for p in preds])
    _, pear = met_pear.get()
    np.testing.assert_almost_equal(pcc, pear)

    # pcc should also accept pred as scalar rather than softmax vector
    # like acc does
    met_pcc.reset()
    met_pcc.update(labels, [p.argmax(axis=1) for p in preds])
    _, chk = met_pcc.get()
    np.testing.assert_almost_equal(pcc, chk)

    # check multiclass case against reference implementation
    CM = [
        [ 23, 13,  3 ],
        [  7, 19, 11 ],
        [  2,  5, 17 ],
    ]
    K = 3
    ref = sum(
        CM[k][k] * CM[l][m] - CM[k][l] * CM[m][k]
        for k in range(K)
        for l in range(K)
        for m in range(K)
    ) / (sum(
        sum(CM[k][l] for l in range(K)) * sum(
            sum(CM[f][g] for g in range(K))
            for f in range(K)
            if f != k
        )
        for k in range(K)
    ) * sum(
        sum(CM[l][k] for l in range(K)) * sum(
            sum(CM[f][g] for f in range(K))
            for g in range(K)
            if g != k
        )
        for k in range(K)
    )) ** 0.5
    labels, preds = cm_batch(CM)
    met_pcc.reset()
    met_pcc.update(labels, preds)
    _, pcc = met_pcc.get()
    np.testing.assert_almost_equal(pcc, ref)

    # things that should not change metric score:
    # * order
    # * batch size
    # * update frequency
    labels = [ [ i.reshape(-1) ] for i in labels[0] ]
    labels.reverse()
    preds = [ [ i.reshape((1, -1)) ] for i in preds[0] ]
    preds.reverse()

    met_pcc.reset()
    for l, p in zip(labels, preds):
        met_pcc.update(l, p)
    assert pcc == met_pcc.get()[1]

@xfail_when_nonstandard_decimal_separator
def test_single_array_input():
    pred = mx.nd.array([[1,2,3,4]])
    label = pred + 0.1

    mse = mx.gluon.metric.create('mse')
    mse.update(label, pred)
    _, mse_res = mse.get()
    np.testing.assert_almost_equal(mse_res, 0.01)

    mae = mx.gluon.metric.create('mae')
    mae.update(label, pred)
    mae.get()
    _, mae_res = mae.get()
    np.testing.assert_almost_equal(mae_res, 0.1)

    rmse = mx.gluon.metric.create('rmse')
    rmse.update(label, pred)
    rmse.get()
    _, rmse_res = rmse.get()
    np.testing.assert_almost_equal(rmse_res, 0.1)

