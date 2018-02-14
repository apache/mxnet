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

from sklearn.metrics import f1_score as scikit_f1

def check_metric(metric, *args, **kwargs):
    metric = mx.metric.create(metric, *args, **kwargs)
    str_metric = json.dumps(metric.get_config())
    metric2 = mx.metric.create(str_metric)

    assert metric.get_config() == metric2.get_config()

def test_metrics():
    check_metric('acc', axis=0)
    check_metric('f1')
    check_metric('perplexity', -1)
    check_metric('pearsonr')
    check_metric('nll_loss')
    composite = mx.metric.create(['acc', 'f1'])
    check_metric(composite)

def test_nll_loss():
    metric = mx.metric.create('nll_loss')
    pred = mx.nd.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    label = mx.nd.array([2, 1])
    metric.update([label], [pred])
    _, loss = metric.get()
    expected_loss = 0.0
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

def test_f1():
    micro_f1 = mx.metric.create("f1", average="micro")
    macro_f1 = mx.metric.F1(average="macro")

    assert np.isnan(macro_f1.get()[1])
    assert np.isnan(micro_f1.get()[1])

    # check divide by zero
    pred = mx.nd.array([[0.9, 0.1],
                        [0.8, 0.2]])
    label = mx.nd.array([0, 0])
    macro_f1.update([label], [pred])
    micro_f1.update([label], [pred])
    assert macro_f1.get()[1] == 0.0
    assert micro_f1.get()[1] == 0.0
    macro_f1.reset()
    micro_f1.reset()

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

    micro_f1.update([label11, label12], [pred11, pred12])
    macro_f1.update([label11, label12], [pred11, pred12])
    assert micro_f1.num_inst == 4
    assert macro_f1.num_inst == 1
    np_pred1 = np.concatenate([mx.nd.argmax(pred11, axis=1).asnumpy(),
                              mx.nd.argmax(pred12, axis=1).asnumpy()])
    np_label1 = np.concatenate([label11.asnumpy(), label12.asnumpy()])
    np.testing.assert_almost_equal(micro_f1.get()[1], scikit_f1(np_label1, np_pred1))
    np.testing.assert_almost_equal(macro_f1.get()[1], scikit_f1(np_label1, np_pred1))

    micro_f1.update([label21, label22], [pred21, pred22])
    macro_f1.update([label21, label22], [pred21, pred22])
    assert micro_f1.num_inst == 6
    assert macro_f1.num_inst == 2
    np_pred2 = np.concatenate([mx.nd.argmax(pred21, axis=1).asnumpy(),
                               mx.nd.argmax(pred22, axis=1).asnumpy()])
    np_pred_total = np.concatenate([np_pred1, np_pred2])
    np_label2 = np.concatenate([label21.asnumpy(), label22.asnumpy()])
    np_label_total = np.concatenate([np_label1, np_label2])
    np.testing.assert_almost_equal(micro_f1.get()[1], scikit_f1(np_label_total, np_pred_total))
    np.testing.assert_almost_equal(macro_f1.get()[1], (scikit_f1(np_label1, np_pred1) +
                                                      scikit_f1(np_label2, np_pred2)) / 2)

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

if __name__ == '__main__':
    import nose
    nose.runmodule()
