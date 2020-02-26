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

# pylint: skip-file
from __future__ import absolute_import
from __future__ import division
import itertools
import os
import sys
from os import path
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../python/common/'))
sys.path.append(os.path.join(curr_path, '../python/unittest/'))
sys.path.insert(0, os.path.join(curr_path, '../../../python'))
import unittest
import numpy as _np
import mxnet as mx
from mxnet import np, npx, autograd
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray, retry, use_np
from common import with_seed
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf, assert_exception, is_op_runnable, collapse_sum_like
from mxnet.ndarray.ndarray import py_slice
from mxnet.base import integer_types
import scipy.stats as ss


@retry(5)
@with_seed()
@use_np
def test_np_exponential():
    samples = 1000000
    # Generation test
    trials = 8
    num_buckets = 5
    for scale in [1.0, 5.0]:
        buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.expon.ppf(x, scale=scale), num_buckets)
        buckets = np.array(buckets, dtype="float32").tolist()
        probs = [(buckets[i][1] - buckets[i][0])/scale for i in range(num_buckets)]
        generator_mx_np = lambda x: mx.np.random.exponential(size=x).asnumpy()
        verify_generator(generator=generator_mx_np, buckets=buckets, probs=probs, nsamples=samples, nrepeat=trials)


@retry(5)
@with_seed()
@use_np
def test_np_uniform():
    types = [None, "float32", "float64"]
    ctx = mx.context.current_context()
    samples = 1000000
    # Generation test
    trials = 8
    num_buckets = 5
    for dtype in types:
        for low, high in [(-100.0, -98.0), (99.0, 101.0)]:
            scale = high - low
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.uniform.ppf(x, loc=low, scale=scale), num_buckets)
            buckets = np.array(buckets, dtype=dtype).tolist()
            probs = [(buckets[i][1] - buckets[i][0])/scale for i in range(num_buckets)]
            generator_mx_np = lambda x: mx.np.random.uniform(low, high, size=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx_np, buckets=buckets, probs=probs, nsamples=samples, nrepeat=trials)


@retry(5)
@with_seed()
@use_np
def test_np_logistic():
    samples = 1000000
    # Generation test
    trials = 8
    num_buckets = 20
    for loc, scale in [(0.0, 1.0), (1.0, 5.0)]:
        buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.logistic.ppf(x, loc=loc, scale=scale), num_buckets)
        buckets = np.array(buckets).tolist()
        probs = [(ss.logistic.cdf(buckets[i][1], loc, scale) -
                  ss.logistic.cdf(buckets[i][0], loc, scale)) for i in range(num_buckets)]
        generator_mx_np = lambda x: mx.np.random.logistic(loc, scale, size=x).asnumpy()
        verify_generator(generator=generator_mx_np, buckets=buckets, probs=probs, nsamples=samples, nrepeat=trials)


@retry(5)
@with_seed()
@use_np
def test_np_gumbel():
    samples = 1000000
    # Generation test
    trials = 8
    num_buckets = 5
    for loc, scale in [(0.0, 1.0), (1.0, 5.0)]:
        buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.gumbel_r.ppf(x, loc=loc, scale=scale), num_buckets)
        buckets = np.array(buckets).tolist()
        probs = [(buckets[i][1] - buckets[i][0])/scale for i in range(num_buckets)]
        generator_mx_np = lambda x: mx.np.random.gumbel(loc, scale, size=x).asnumpy()
        verify_generator(generator=generator_mx_np, buckets=buckets, probs=probs, nsamples=samples, nrepeat=trials)


@retry(5)
@with_seed()
@use_np
def test_np_normal():
    types = [None, "float32", "float64"]
    ctx = mx.context.current_context()
    samples = 1000000
    # Generation test
    trials = 8
    num_buckets = 5
    for dtype in types:
        for loc, scale in [(0.0, 1.0), (1.0, 5.0)]:
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.norm.ppf(x, loc=loc, scale=scale), num_buckets)
            buckets = np.array(buckets, dtype=dtype).tolist()
            probs = [(ss.norm.cdf(buckets[i][1], loc, scale) -
                      ss.norm.cdf(buckets[i][0], loc, scale)) for i in range(num_buckets)]
            generator_mx_np = lambda x: mx.np.random.normal(loc, scale, size=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx_np, buckets=buckets, probs=probs, nsamples=samples, nrepeat=trials)


@retry(5)
@with_seed()
@use_np
def test_np_gamma():
    types = [None, "float32", "float64"]
    ctx = mx.context.current_context()
    samples = 1000000
    # Generation test
    trials = 8
    num_buckets = 5
    for dtype in types:
        for alpha, beta in [(2.0, 3.0), (0.5, 1.0)]:
            buckets, probs = gen_buckets_probs_with_ppf(
                lambda x: ss.gamma.ppf(x, a=alpha, loc=0, scale=beta), num_buckets)
            buckets = np.array(buckets).tolist()
            def generator_mx(x): return np.random.gamma(
                alpha, beta, size=samples, ctx=ctx).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs,
                             nsamples=samples, nrepeat=trials)
            generator_mx_same_seed =\
                lambda x: _np.concatenate(
                    [np.random.gamma(alpha, beta, size=(x // 10), ctx=ctx).asnumpy()
                        for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs,
                             nsamples=samples, nrepeat=trials)


if __name__ == '__main__':
    import nose
    nose.runmodule()
