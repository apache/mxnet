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

from __future__ import print_function

import itertools
import mxnet as mx
import sys
import time


class MetricDataGen(object):
    """ Base class for generating random data for metric benchmarking """
    def __init__(self, n, c, pred_ctx, label_ctx):
        self.n = n
        self.c = c
        self.pred_ctx = pred_ctx
        self.label_ctx = label_ctx

    def data(self):
        mx.random.seed(0)
        pred = mx.nd.random_uniform(0.0, 1.0, (self.n, self.c), ctx=self.pred_ctx)
        label = mx.nd.random_uniform(0.0, self.c - 1, (self.n,), ctx=self.label_ctx).round()
        return label, pred

    @property
    def batch_size(self):
        return self.n

    @property
    def output_dim(self):
        return self.c


class F1MetricDataGen(MetricDataGen):
    """ Class for generating random data for F1 metric benchmarking """
    def __init__(self, n, c, pred_ctx, label_ctx):
        super(F1MetricDataGen, self).__init__(n, 2, pred_ctx, label_ctx)


class PearsonMetricDataGen(MetricDataGen):
    """ Class for generating random data for Pearson Correlation metric benchmarking """
    def __init__(self, n, c, pred_ctx, label_ctx):
        super(PearsonMetricDataGen, self).__init__(n, c, pred_ctx, label_ctx)

    def data(self):
        mx.random.seed(0)
        pred = mx.nd.random_uniform(0.0, 1.0, (self.n, self.c), ctx=self.pred_ctx)
        label = mx.nd.random_uniform(0.0, 1.0, (self.n, self.c), ctx=self.label_ctx)
        return label, pred


def run_metric(name, data_gen_cls, i, n, c, pred_ctx, label_ctx, **kwargs):
    """ Helper function for running one metric benchmark """
    metric = mx.gluon.metric.create(name, **kwargs)
    data_gen = data_gen_cls(n, c, pred_ctx, label_ctx)
    try:
        label, pred = data_gen.data()
        mx.nd.waitall()
        before = time.time()
        metric.update([label] * i, [pred] * i)
        mx.nd.waitall()
        elapsed = time.time() - before
        elapsed_str = f"{elapsed:<.5}"
    except mx.MXNetError:
        elapsed_str = "FAILED"
    print(f"{name:<15}{pred_ctx:<10}{label_ctx:<12}{i * n:<12}{data_gen.batch_size:<15}{data_gen.output_dim:<15}{elapsed_str:<}", file=sys.stderr)


def test_metric_performance():
    """ unittest entry for metric performance benchmarking """
    # Each dictionary entry is (metric_name:(kwargs, DataGenClass))
    metrics = [
        ('acc', ({}, MetricDataGen)),
        ('top_k_acc', ({'top_k': 5}, MetricDataGen)),
        ('F1', ({}, F1MetricDataGen)),
        ('Perplexity', ({'ignore_label': -1}, MetricDataGen)),
        ('MAE', ({}, MetricDataGen)),
        ('MSE', ({}, MetricDataGen)),
        ('RMSE', ({}, MetricDataGen)),
        ('ce', ({}, MetricDataGen)),
        ('nll_loss', ({}, MetricDataGen)),
        ('pearsonr', ({}, PearsonMetricDataGen)),
    ]

    data_size = 1024 * 128

    batch_sizes = [16, 64, 256, 1024]
    output_dims = [128, 1024, 8192]
    ctxs = [mx.cpu(), mx.gpu()]

    print("\nmx.gluon.metric benchmarks", file=sys.stderr)
    print(
        f"{'Metric':15}{'Data-Ctx':10}{'Label-Ctx':12}{'Data Size':12}{'Batch Size':15}{'Output Dim':15}{'Elapsed Time'}",
        file=sys.stderr)
    print(f"{'':-^90}", file=sys.stderr)
    for k, v in metrics:
        for c in output_dims:
            for n in batch_sizes:
                for pred_ctx, label_ctx in itertools.product(ctxs, ctxs):
                    run_metric(k, v[1], (data_size * 128), (n * c), n, c, pred_ctx, label_ctx, **v[0])
                print(f"{'':-^90}", file=sys.stderr)
