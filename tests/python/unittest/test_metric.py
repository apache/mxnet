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
    check_metric('top_k_precision')
    check_metric('top_k_recall')
    composite = mx.metric.create(['acc', 'f1'])
    check_metric(composite)

def test_top_k_precision():
    ytrue = [[1.,0.,1.,0.],[0.,1.,1.,0.]]
    ytrue = mx.nd.array(ytrue)
    yhat = [[0.4,0.8,0.1,0.1],[0.4,0.8,0.8,0.4]]
    yhat = mx.nd.array(yhat)
    pre = mx.metric.create('top_k_precision',top_k=2)
    pre.update(preds = [yhat], labels = [ytrue])
    assert(pre.get()[1]==0.75), pre.get()


def test_top_k_recall():
    ytrue = [[1.,0.,1.,0.],[0.,1.,1.,0.]]
    ytrue = mx.nd.array(ytrue)
    yhat = [[0.4,0.1,0.1,0.4],[0.4,0.8,0.8,0.4]]
    yhat = mx.nd.array(yhat)
    rec = mx.metric.create('top_k_recall',top_k=2)
    rec.update(preds = [yhat], labels = [ytrue])
    assert(rec.get()[1]==0.75), rec.get()




if __name__ == '__main__':
    import nose
    nose.runmodule()
