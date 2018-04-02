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
# pylint: disable=wildcard-import, unused-wildcard-import
"""Contrib Symbol API of MXNet."""
import math
from .random import uniform
from .symbol import Symbol
try:
    from .gen_contrib import *
except ImportError:
    pass

__all__ = ["rand_zipfian"]

def rand_zipfian(true_classes, num_sampled, range_max):
    """Draw random samples from an approximately log-uniform or Zipfian distribution.

    This operation randomly samples *num_sampled* candidates the range of integers [0, range_max).
    The elements of sampled_candidates are drawn with replacement from the base distribution.

    The base distribution for this operator is an approximately log-uniform or Zipfian distribution:

    P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)

    This sampler is useful when the true classes approximately follow such a distribution.
    For example, if the classes represent words in a lexicon sorted in decreasing order of \
    frequency. If your classes are not ordered by decreasing frequency, do not use this op.

    Additionaly, it also returns the number of times each of the \
    true classes and the sampled classes is expected to occur.

    Parameters
    ----------
    true_classes : Symbol
        The target classes in 1-D.
    num_sampled: int
        The number of classes to randomly sample.
    range_max: int
        The number of possible classes.

    Returns
    -------
    samples: Symbol
        The sampled candidate classes in 1-D `int64` dtype.
    expected_count_true: Symbol
        The expected count for true classes in 1-D `float64` dtype.
    expected_count_sample: Symbol
        The expected count for sampled candidates in 1-D `float64` dtype.

    Examples
    --------
    >>> true_cls = mx.nd.array([3])
    >>> samples, exp_count_true, exp_count_sample = mx.nd.contrib.rand_zipfian(true_cls, 4, 5)
    >>> samples
    [1 3 3 3]
    <NDArray 4 @cpu(0)>
    >>> exp_count_true
    [ 0.12453879]
    <NDArray 1 @cpu(0)>
    >>> exp_count_sample
    [ 0.22629439  0.12453879  0.12453879  0.12453879]
    <NDArray 4 @cpu(0)>
    """
    assert(isinstance(true_classes, Symbol)), "unexpected type %s" % type(true_classes)
    log_range = math.log(range_max + 1)
    rand = uniform(0, log_range, shape=(num_sampled,), dtype='float64')
    # make sure sampled_classes are in the range of [0, range_max)
    sampled_classes = (rand.exp() - 1).astype('int64') % range_max

    true_classes = true_classes.astype('float64')
    expected_prob_true = ((true_classes + 2.0) / (true_classes + 1.0)).log() / log_range
    expected_count_true = expected_prob_true * num_sampled
    # cast sampled classes to fp64 to avoid interget division
    sampled_cls_fp64 = sampled_classes.astype('float64')
    expected_prob_sampled = ((sampled_cls_fp64 + 2.0) / (sampled_cls_fp64 + 1.0)).log() / log_range
    expected_count_sampled = expected_prob_sampled * num_sampled
    return sampled_classes, expected_count_true, expected_count_sampled
