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

import math
import os
import numpy as np
import mxnet as mx
import log_uniform
from mxnet import ndarray

class LogUniformSampler():
    def __init__(self, range_max, num_sampled):
        self.range_max = range_max
        self.num_sampled = num_sampled
        self.sampler = log_uniform.LogUniformSampler(range_max)

    def _prob_helper(self, num_tries, num_sampled, prob):
        if num_tries == num_sampled:
            return prob * num_sampled
        return (num_tries * (-prob).log1p()).expm1() * -1

    def draw(self, true_classes):
        """Draw samples from log uniform distribution and returns sampled candidates,
        expected count for true classes and sampled classes."""
        range_max = self.range_max
        num_sampled = self.num_sampled
        ctx = true_classes.context
        log_range = math.log(range_max + 1)
        num_tries = 0
        true_classes = true_classes.reshape((-1,))
        sampled_classes, num_tries = self.sampler.sample_unique(num_sampled)

        true_cls = true_classes.as_in_context(ctx).astype('float64')
        prob_true = ((true_cls + 2.0) / (true_cls + 1.0)).log() / log_range
        count_true = self._prob_helper(num_tries, num_sampled, prob_true)

        sampled_classes = ndarray.array(sampled_classes, ctx=ctx, dtype='int64')
        sampled_cls_fp64 = sampled_classes.astype('float64')
        prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)).log() / log_range
        count_sampled = self._prob_helper(num_tries, num_sampled, prob_sampled)
        return [sampled_classes, count_true, count_sampled]
