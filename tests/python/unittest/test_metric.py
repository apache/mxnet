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
    pred = mx.nd.array([[0.3, 0.7], [1., 0], [0.4, 0.6], [0.6, 0.4], [0.9, 0.1]])
    label = mx.nd.array([0, 1, 1, 1, 1])
    positives = np.argmax(pred, axis=1).sum().asscalar()
    true_positives = (np.argmax(pred, axis=1) == label).sum().asscalar()
    precision = true_positives / positives
    overall_positives = label.sum().asscalar()
    recall = true_positives / overall_positives
    f1_expected = 2 * (precision * recall) / (precision + recall)
    metric = mx.metric.create('f1')
    metric.update([label], [pred])
    _, f1 = metric.get()
    assert f1 == f1_expected

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
