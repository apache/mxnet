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

import os
import tempfile

import mxnet as mx
from mxnet import gluon
import mxnet.gluon.probability as mgp
from mxnet.gluon import HybridBlock
from mxnet.test_utils import use_np, assert_almost_equal
from mxnet import np, npx, autograd
import numpy as _np
from common import (setup_module, with_seed, assertRaises, teardown,
                    assert_raises_cudnn_not_satisfied)
from numpy.testing import assert_array_equal
from nose.tools import raises, assert_raises
import scipy.stats as ss
import scipy.special as scipy_special
import warnings
import json
import unittest
import random
import itertools


@with_seed()
@use_np
def test_gluon_normal():
    class TestNormal(HybridBlock):
        def __init__(self, func):
            super(TestNormal, self).__init__()
            self._func = func

        def hybrid_forward(self, F, loc, scale, *args):
            normal = mgp.Normal(loc, scale, F)
            return getattr(normal, self._func)(*args)

    shapes = [(), (1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.normal(size=shape)
        net = TestNormal("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale, samples).asnumpy()
        np_out = _np.log(ss.norm(loc.asnumpy(),
                                 scale.asnumpy()).pdf(samples.asnumpy()))
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # TODO: test cdf
    # TODO: test icdf


@with_seed()
@use_np
def test_gluon_bernoulli():
    class TestBernoulli(HybridBlock):
        def __init__(self, func, is_logit=False):
            super(TestBernoulli, self).__init__()
            self._is_logit = is_logit
            self._func = func

        def hybrid_forward(self, F, params, *args):
            bernoulli = mgp.Bernoulli(logit=params, F=F) if self._is_logit else \
                        mgp.Bernoulli(prob=params, F=F)
            if (len(args) == 0):
                return getattr(bernoulli, self._func)
            return getattr(bernoulli, self._func)(*args)

    def prob_to_logit(prob):
        return np.log(prob) - np.log1p(-prob)

    # Test log_prob
    shapes = [(1,), (2, 3), 6]
    for shape, hybridize, use_logit in itertools.product(shapes, [True, False], [True, False]):
        prob = np.random.uniform(size=shape)
        sample = npx.random.bernoulli(prob=0.5, size=shape)
        param = prob
        if use_logit:
            param = prob_to_logit(param)
        net = TestBernoulli("log_prob", use_logit)
        if hybridize:
            net.hybridize()
        mx_out = net(param, sample).asnumpy()
        np_out = _np.log(ss.bernoulli.pmf(sample.asnumpy(), prob.asnumpy()))
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                        rtol=1e-3, use_broadcast=False)

    # Test variance
    for shape, hybridize, use_logit in itertools.product(shapes, [True, False], [True, False]):
        prob = np.random.uniform(size=shape)
        sample = npx.random.bernoulli(prob=0.5, size=shape)
        param = prob
        if use_logit:
            param = prob_to_logit(param)
        net = TestBernoulli("variance", use_logit)
        if hybridize:
            net.hybridize()
        mx_out = net(param).asnumpy()
        np_out = ss.bernoulli(prob.asnumpy()).var()
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                        rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_affine_transform():
    r"""
    Test the correctness of affine transformation by performing it
    on a standard normal, since N(\mu, \sigma^2) = \mu + \sigma * N(0, 1)
    """
    class TestAffineTransform(HybridBlock):
        def __init__(self, func):
            super(TestAffineTransform, self).__init__()
            self._func = func

        def hybrid_forward(self, F, loc, scale, *args):
            std_normal = mgp.Normal(F.np.zeros_like(loc),
                                    F.np.ones_like(scale), F)
            transforms = [mgp.AffineTransform(loc=0, scale=scale),
                          mgp.AffineTransform(loc=loc, scale=1)]
            transformed_normal = mgp.TransformedDistribution(std_normal, transforms)
            if (len(args) == 0):
                return getattr(transformed_normal, self._func)
            return getattr(transformed_normal, self._func)(*args)

    shapes = [(1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]): 
        loc = np.random.uniform(-1, 1, shape)
        loc.attach_grad()
        scale = np.random.uniform(0.5, 1.5, shape)
        scale.attach_grad()
        samples = np.random.normal(size=shape)
        net = TestAffineTransform('log_prob')
        if hybridize:
            net.hybridize()
        with autograd.record():
            mx_out = net(loc, scale, samples)
        np_out = _np.log(ss.norm(loc.asnumpy(),
                                 scale.asnumpy()).pdf(samples.asnumpy()))
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)
        mx_out.backward()
        loc_expected_grad = ((samples - loc) / scale ** 2).asnumpy()
        scale_expected_grad = (samples - loc) ** 2 * np.power(scale, -3) - (1 / scale)
        assert_almost_equal(loc.grad.asnumpy(), loc_expected_grad, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)
        assert_almost_equal(scale.grad.asnumpy(), scale_expected_grad, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test sampling
    for shape, hybridize in itertools.product(shapes, [True, False]): 
        loc = np.random.uniform(-1, 1, shape)
        loc.attach_grad()
        scale = np.random.uniform(0.5, 1.5, shape)
        scale.attach_grad()
        if not isinstance(shape, tuple):
            shape = (shape,)
        expected_shape = (4, 5) + shape
        net = TestAffineTransform('sample')
        mx_out = net(loc, scale, expected_shape).asnumpy()
        assert mx_out.shape == expected_shape


@with_seed()
@use_np
def test_compose_transform():
    class TestComposeTransform(HybridBlock):
        def __init__(self, func):
            super(TestComposeTransform, self).__init__()
            self._func = func

        def hybrid_forward(self, F, loc, scale, *args):
            # Generate a log_normal distribution.
            std_normal = mgp.Normal(F.np.zeros_like(loc),
                                    F.np.ones_like(scale), F)
            transforms = mgp.ComposeTransform([
                            mgp.AffineTransform(loc=0, scale=scale),
                            mgp.AffineTransform(loc=loc, scale=1),
                            mgp.ExpTransform()
                            ])
            transformed_normal = mgp.TransformedDistribution(std_normal, transforms)
            if (len(args) == 0):
                return getattr(transformed_normal, self._func)
            return getattr(transformed_normal, self._func)(*args)

    shapes = [(1,), (2, 3), 6]
    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [False]): 
        loc = np.random.uniform(-1, 1, shape)
        loc.attach_grad()
        scale = np.random.uniform(0.5, 1.5, shape)
        scale.attach_grad()
        samples = np.random.uniform(1, 2, size=shape)
        net = TestComposeTransform('log_prob')
        if hybridize:
            net.hybridize()
        with autograd.record():
            mx_out = net(loc, scale, samples)
        np_out = _np.log(
                    ss.lognorm(s=scale.asnumpy(), scale=np.exp(loc).asnumpy()).
                    pdf(samples.asnumpy())
                )
        assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)
            

if __name__ == '__main__':
    import nose
    nose.runmodule()
