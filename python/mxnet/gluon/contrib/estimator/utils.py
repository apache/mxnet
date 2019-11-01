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

# coding: utf-8
# pylint: disable=wildcard-import, unused-variable
"""Gluon Estimator Utility Functions"""

from ....metric import EvalMetric, CompositeEvalMetric

def _check_metrics(metrics):
    if isinstance(metrics, CompositeEvalMetric):
        metrics = [m for metric in metrics.metrics for m in _check_metrics(metric)]
    elif isinstance(metrics, EvalMetric):
        metrics = [metrics]
    else:
        metrics = metrics or []
        if not all([isinstance(metric, EvalMetric) for metric in metrics]):
            raise ValueError("metrics must be a Metric or a list of Metric, "
                             "refer to mxnet.metric.EvalMetric:{}".format(metrics))
    return metrics
