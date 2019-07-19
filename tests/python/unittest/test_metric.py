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
import json
from common import with_seed
from copy import deepcopy

def check_metric(metric, *args, **kwargs):
    metric = mx.metric.create(metric, *args, **kwargs)
    str_metric = json.dumps(metric.get_config())
    metric2 = mx.metric.create(str_metric)

    assert metric.get_config() == metric2.get_config()

def test_metrics():
    check_metric('acc', axis=0)
    check_metric('f1')
    check_metric('mcc')
    check_metric('perplexity', -1)
    check_metric('pearsonr')
    check_metric('pcc')
    check_metric('nll_loss')
    check_metric('loss')
    composite = mx.metric.create(['acc', 'f1'])
    check_metric(composite)

def _check_global_metric(metric, *args, **kwargs):
    def _create_pred_label():
        if use_same_shape:
            pred = mx.nd.random.uniform(0, 1, shape=shape)
            label = mx.nd.random.uniform(0, 1, shape=shape)
        else:
            # Make a random prediction
            idx = np.random.rand(*shape).argsort(1)
            pred = mx.nd.array(1 - 0.1 * idx)
            # Label is half 1 and half 0
            # Setting all 0s or all 1s would make either
            # MCC or F1 metrics always produce 0
            label = mx.nd.ones(shape[0])
            label[:shape[0] // 2] = 0
        return pred, label

    shape = kwargs.pop('shape', (10,10))
    use_same_shape = kwargs.pop('use_same_shape', False)
    m1 = mx.metric.create(metric, *args, **kwargs)
    m2 = deepcopy(m1)
    # check that global stats are not reset when calling
    # reset_local()
    for i in range(10):
        pred, label = _create_pred_label()
        m1.update([label], [pred])
        m1.reset_local()
        m2.update([label], [pred])
    assert m1.get_global() == m2.get()

    # check that reset_local() properly resets the local state
    m1.reset_local()
    m2.reset()
    pred, label = _create_pred_label()
    m1.update([label], [pred])
    m1.reset_local()
    pred, label = _create_pred_label()
    m1.update([label], [pred])
    m2.update([label], [pred])
    assert m1.get() == m2.get()

@with_seed()
def test_global_metric():
    _check_global_metric('acc')
    _check_global_metric('TopKAccuracy', top_k=3)
    _check_global_metric('f1', shape=(10,2))
    _check_global_metric('f1', shape=(10,2), average='micro')
    _check_global_metric('mcc', shape=(10,2))
    _check_global_metric('mcc', shape=(10,2), average='micro')
    _check_global_metric('perplexity', -1)
    _check_global_metric('pearsonr', use_same_shape=True)
    _check_global_metric('pcc', shape=(10,2))
    _check_global_metric('nll_loss')
    _check_global_metric('loss')
    _check_global_metric('ce')
    _check_global_metric('mae', use_same_shape=True)
    _check_global_metric('mse', use_same_shape=True)
    _check_global_metric('rmse', use_same_shape=True)
    def custom_metric(label, pred):
        return np.mean(np.abs(label-pred))
    _check_global_metric(custom_metric, use_same_shape=True)
    _check_global_metric(['acc', 'f1'], shape=(10,2))

def test_nll_loss():
    metric = mx.metric.create('nll_loss')
    pred = mx.nd.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    label = mx.nd.array([2, 1])
    metric.update([label], [pred])
    _, loss = metric.get()
    expected_loss = -(np.log(pred[0][2].asscalar()) + np.log(pred[1][1].asscalar())) / 2
    assert loss == expected_loss

def test_acc():
    pred = mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])
    label = mx.nd.array([0, 1, 1])
    metric = mx.metric.create('acc')
    metric.update([label], [pred])
    _, acc = metric.get()
    expected_acc = (np.argmax(pred, axis=1) == label).sum().asscalar() / label.size
    assert acc == expected_acc

def test_acc_2d_label():
    # label maybe provided in 2d arrays in custom data iterator
    pred = mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6], [0.8, 0.2], [0.3, 0.5], [0.6, 0.4]])
    label = mx.nd.array([[0, 1, 1], [1, 0, 1]])
    metric = mx.metric.create('acc')
    metric.update([label], [pred])
    _, acc = metric.get()
    expected_acc = (np.argmax(pred, axis=1).asnumpy() == label.asnumpy().ravel()).sum() / \
                   float(label.asnumpy().ravel().size)
    assert acc == expected_acc

def test_loss_update():
    pred = mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])
    metric1 = mx.metric.create('loss')
    metric2 = mx.metric.create('loss')
    metric1.update(None, [pred])
    metric2.update(None, pred)
    _, acc1 = metric1.get()
    _, acc2 = metric2.get()
    assert acc1 == acc2

def test_f1():
    microF1 = mx.metric.create("f1", average="micro")
    macroF1 = mx.metric.F1(average="macro")

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
    assert macroF1.num_inst == 1
    # f1 = 2 * tp / (2 * tp + fp + fn)
    fscore1 = 2. * (1) / (2 * 1 + 1 + 0)
    np.testing.assert_almost_equal(microF1.get()[1], fscore1)
    np.testing.assert_almost_equal(macroF1.get()[1], fscore1)

    microF1.update([label21, label22], [pred21, pred22])
    macroF1.update([label21, label22], [pred21, pred22])
    assert microF1.num_inst == 6
    assert macroF1.num_inst == 2
    fscore2 = 2. * (1) / (2 * 1 + 0 + 0)
    fscore_total = 2. * (1 + 1) / (2 * (1 + 1) + (1 + 0) + (0 + 0))
    np.testing.assert_almost_equal(microF1.get()[1], fscore_total)
    np.testing.assert_almost_equal(macroF1.get()[1], (fscore1 + fscore2) / 2.)

def test_mcc():
    microMCC = mx.metric.create("mcc", average="micro")
    macroMCC = mx.metric.MCC(average="macro")

    assert np.isnan(microMCC.get()[1])
    assert np.isnan(macroMCC.get()[1])

    # check divide by zero
    pred = mx.nd.array([[0.9, 0.1],
                        [0.8, 0.2]])
    label = mx.nd.array([0, 0])
    microMCC.update([label], [pred])
    macroMCC.update([label], [pred])
    assert microMCC.get()[1] == 0.0
    assert macroMCC.get()[1] == 0.0
    microMCC.reset()
    macroMCC.reset()

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
    macroMCC.update([label11, label12], [pred11, pred12])
    assert microMCC.num_inst == 4
    assert macroMCC.num_inst == 1
    tp1 = 1; fp1 = 0; fn1 = 1; tn1=2
    mcc1 = (tp1*tn1 - fp1*fn1) / np.sqrt((tp1+fp1)*(tp1+fn1)*(tn1+fp1)*(tn1+fn1))
    np.testing.assert_almost_equal(microMCC.get()[1], mcc1)
    np.testing.assert_almost_equal(macroMCC.get()[1], mcc1)

    microMCC.update([label21, label22], [pred21, pred22])
    macroMCC.update([label21, label22], [pred21, pred22])
    assert microMCC.num_inst == 6
    assert macroMCC.num_inst == 2
    tp2 = 1; fp2 = 0; fn2 = 0; tn2=1
    mcc2 = (tp2*tn2 - fp2*fn2) / np.sqrt((tp2+fp2)*(tp2+fn2)*(tn2+fp2)*(tn2+fn2))
    tpT = tp1+tp2; fpT = fp1+fp2; fnT = fn1+fn2; tnT = tn1+tn2;
    mccT = (tpT*tnT - fpT*fnT) / np.sqrt((tpT+fpT)*(tpT+fnT)*(tnT+fpT)*(tnT+fnT))
    np.testing.assert_almost_equal(microMCC.get()[1], mccT)
    np.testing.assert_almost_equal(macroMCC.get()[1], .5*(mcc1+mcc2))

def test_perplexity():
    pred = mx.nd.array([[0.8, 0.2], [0.2, 0.8], [0, 1.]])
    label = mx.nd.array([0, 1, 1])
    p = pred.asnumpy()[np.arange(label.size), label.asnumpy().astype('int32')]
    perplexity_expected = np.exp(-np.log(p).sum()/label.size)
    metric = mx.metric.create('perplexity', -1)
    metric.update([label], [pred])
    _, perplexity = metric.get()
    assert perplexity == perplexity_expected

def test_pearsonr():
    pred = mx.nd.array([[0.7, 0.3], [0.1, 0.9], [1., 0]])
    label = mx.nd.array([[0, 1], [1, 0], [1, 0]])
    pearsonr_expected = np.corrcoef(pred.asnumpy().ravel(), label.asnumpy().ravel())[0, 1]
    metric = mx.metric.create('pearsonr')
    metric.update([label], [pred])
    _, pearsonr = metric.get()
    assert pearsonr == pearsonr_expected

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
    met_pcc = mx.metric.create('pcc')
    met_pcc.update(labels, preds)
    _, pcc = met_pcc.get()

    # pcc should agree with mcc for binary classification
    met_mcc = mx.metric.create('mcc')
    met_mcc.update(labels, preds)
    _, mcc = met_mcc.get()
    np.testing.assert_almost_equal(pcc, mcc)

    # pcc should agree with Pearson for binary classification
    met_pear = mx.metric.create('pearsonr')
    met_pear.update(labels, [p.argmax(axis=1) for p in preds])
    _, pear = met_pear.get()
    np.testing.assert_almost_equal(pcc, pear)

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
    labels = [ [ i ] for i in labels[0] ]
    labels.reverse()
    preds = [ [ i.reshape((1, -1)) ] for i in preds[0] ]
    preds.reverse()

    met_pcc.reset()
    for l, p in zip(labels, preds):
        met_pcc.update(l, p)
    assert pcc == met_pcc.get()[1]

def test_single_array_input():
    pred = mx.nd.array([[1,2,3,4]])
    label = pred + 0.1

    mse = mx.metric.create('mse')
    mse.update(label, pred)
    _, mse_res = mse.get()
    np.testing.assert_almost_equal(mse_res, 0.01)

    mae = mx.metric.create('mae')
    mae.update(label, pred)
    mae.get()
    _, mae_res = mae.get()
    np.testing.assert_almost_equal(mae_res, 0.1)

    rmse = mx.metric.create('rmse')
    rmse.update(label, pred)
    rmse.get()
    _, rmse_res = rmse.get()
    np.testing.assert_almost_equal(rmse_res, 0.1)

if __name__ == '__main__':
    import nose
    nose.runmodule()
