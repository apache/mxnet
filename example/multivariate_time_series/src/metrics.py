# !/usr/bin/env python

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

# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx

def rse(label, pred):
    """computes the root relative squared error (condensed using standard deviation formula)"""
    numerator = np.sqrt(np.mean(np.square(label - pred), axis = None))
    denominator = np.std(label, axis = None)
    return numerator / denominator

def rae(label, pred):
    """computes the relative absolute error (condensed using standard deviation formula)"""
    numerator = np.mean(np.abs(label - pred), axis=None)
    denominator = np.mean(np.abs(label - np.mean(label, axis=None)), axis=None)
    return numerator / denominator

def corr(label, pred):
    """computes the empirical correlation coefficient"""
    numerator1 = label - np.mean(label, axis=0)
    numerator2 = pred - np.mean(pred, axis = 0)
    numerator = np.mean(numerator1 * numerator2, axis=0)
    denominator = np.std(label, axis=0) * np.std(pred, axis=0)
    return np.mean(numerator / denominator)

def get_custom_metrics():
    """
    :return: mxnet metric object
    """
    _rse = mx.metric.create(rse)
    _rae = mx.metric.create(rae)
    _corr = mx.metric.create(corr)
    return mx.metric.create([_rae, _rse, _corr])

def evaluate(pred, label):
    return {"RAE":rae(label, pred), "RSE":rse(label,pred),"CORR": corr(label,pred)}