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
from numbers import Number


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
        np_out = ss.norm(loc.asnumpy(),
                        scale.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test cdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.normal(size=shape)
        net = TestNormal("cdf")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale, samples).asnumpy()
        np_out = ss.norm(loc.asnumpy(),
                        scale.asnumpy()).cdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test icdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(size=shape)
        net = TestNormal("icdf")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale, samples).asnumpy()
        np_out = ss.norm(loc.asnumpy(),
                        scale.asnumpy()).ppf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test entropy
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        net = TestNormal("entropy")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale).asnumpy()
        np_out = ss.norm(loc.asnumpy(),
                        scale.asnumpy()).entropy()
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)


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
                out = getattr(bernoulli, self._func)
                if callable(out):
                    return out()
                else:
                    return out
            return getattr(bernoulli, self._func)(*args)

    def prob_to_logit(prob):
        return np.log(prob) - np.log1p(-prob)

    # Test log_prob
    shapes = [(), (1,), (2, 3), 6]
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

    # Test entropy
    for shape, hybridize, use_logit in itertools.product(shapes, [True, False], [True, False]):
        prob = np.random.uniform(size=shape)
        sample = npx.random.bernoulli(prob=0.5, size=shape)
        param = prob
        if use_logit:
            param = prob_to_logit(param)
        net = TestBernoulli("entropy", use_logit)
        if hybridize:
            net.hybridize()
        mx_out = net(param).asnumpy()
        np_out = ss.bernoulli(prob.asnumpy()).entropy()
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                        rtol=1e-3, use_broadcast=False)

    # Test constraint violation
    for shape, hybridize, use_logit in itertools.product(shapes, [True, False], [True, False]):
        prob = np.random.uniform(size=shape)
        sample = npx.random.bernoulli(prob=0.5, size=shape) + 0.1
        param = prob
        if use_logit:
            param = prob_to_logit(param)
        net = TestBernoulli("log_prob", use_logit)
        if hybridize:
            net.hybridize()
        try:
            mx_out = net(param, sample).asnumpy()
        except ValueError:
            pass
        else:
            assert False


@with_seed()
@use_np
def test_gluon_categorical():
    class TestCategorical(HybridBlock):
        def __init__(self, func, is_logit=False, batch_shape=None, num_events=None):
            super(TestCategorical, self).__init__()
            self._is_logit = is_logit
            self._func = func
            self._batch_shape = batch_shape
            self._num_events = num_events

        def hybrid_forward(self, F, params, *args):
            categorical = mgp.Categorical(self._num_events, logit=params, F=F) if self._is_logit else \
                        mgp.Categorical(self._num_events, prob=params, F=F)
            
            if self._func == "sample":
                return categorical.sample(self._batch_shape)

            if (len(args) == 0):
                out = getattr(categorical, self._func)
                if callable(out):
                    return out()
                else:
                    return out
            return getattr(categorical, self._func)(*args)

    event_shapes = [2, 5, 10]
    batch_shapes = [None, (2, 3), (4, 0, 5)]
    sample_shapes = [(), (2,), (3, 4)]

    # Test sampling
    for event_shape, batch_shape in itertools.product(event_shapes, batch_shapes):
        for use_logit, hybridize in itertools.product([True, False], [True, False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            param = prob
            if use_logit:
                param = np.log(param)
            net = TestCategorical("sample", use_logit, batch_shape, event_shape)
            if hybridize:
                net.hybridize()
            mx_out = net(param).asnumpy()
            desired_shape = batch_shape if batch_shape is not None else ()
            assert mx_out.shape == desired_shape

    # Test log_prob
    for event_shape, batch_shape, sample_shape in itertools.product(event_shapes, batch_shapes, sample_shapes):
        for use_logit, hybridize in itertools.product([True, False], [True, False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            param = prob
            desired_shape = sample_shape + (batch_shape if batch_shape is not None else ())
            samples = np.random.choice(event_shape, size=desired_shape)
            if use_logit:
                param = np.log(param)
            net = TestCategorical("log_prob", use_logit, batch_shape, event_shape)
            if hybridize:
                net.hybridize()
            mx_out = net(param, samples)
            # Check shape
            assert mx_out.shape == desired_shape
            # Check value
            log_pmf, indices = np.broadcast_arrays(np.log(prob), np.expand_dims(samples, -1))
            if indices.ndim >= 1:
                indices = indices[..., :1]
            expect_log_prob = _np.take_along_axis(log_pmf, indices.astype('int'), axis=-1).asnumpy()
            assert_almost_equal(mx_out.asnumpy(), expect_log_prob.squeeze(), atol=1e-4,
                        rtol=1e-3, use_broadcast=False)

    # Test enumerate_support
    for event_shape, batch_shape in itertools.product(event_shapes, batch_shapes):
        for use_logit, hybridize in itertools.product([True, False], [True, False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            param = prob
            if use_logit:
                param = np.log(param)
            net = TestCategorical("enumerate_support", use_logit, batch_shape, event_shape)
            if hybridize:
                net.hybridize()
            mx_out = net(param).asnumpy()
            desired_shape = (event_shape,) + (batch_shape if batch_shape is not None else ())
            assert mx_out.shape == desired_shape


@with_seed()
@use_np
def test_gluon_one_hot_categorical():
    def one_hot(a, num_classes):
        return np.identity(num_classes)[a]

    class TestOneHotCategorical(HybridBlock):
        def __init__(self, func, is_logit=False, batch_shape=None, num_events=None):
            super(TestOneHotCategorical, self).__init__()
            self._is_logit = is_logit
            self._func = func
            self._batch_shape = batch_shape
            self._num_events = num_events

        def hybrid_forward(self, F, params, *args):
            categorical = mgp.OneHotCategorical(num_events=self._num_events, logit=params, F=F) \
                          if self._is_logit else \
                          mgp.OneHotCategorical(num_events=self._num_events, prob=params, F=F)
            if self._func == "sample":
                return categorical.sample(self._batch_shape)
            if (len(args) == 0):
                out = getattr(categorical, self._func)
                if callable(out):
                    return out()
                else:
                    return out
            return getattr(categorical, self._func)(*args)

    event_shapes = [2, 5, 10]
    batch_shapes = [None, (2, 3), (4, 0, 5)]
    sample_shapes = [(), (2,), (3, 4)]

    # Test sampling
    for event_shape, batch_shape in itertools.product(event_shapes, batch_shapes):
        for use_logit, hybridize in itertools.product([True, False], [True, False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            param = prob
            if use_logit:
                param = np.log(param)
            net = TestOneHotCategorical("sample", use_logit, batch_shape, event_shape)
            if hybridize:
                net.hybridize()
            mx_out = net(param).asnumpy()
            desired_shape = batch_shape if batch_shape is not None else ()
            assert mx_out.shape == desired_shape + (event_shape,)

    # Test log_prob
    for event_shape, batch_shape, sample_shape in itertools.product(event_shapes, batch_shapes, sample_shapes):
        for use_logit, hybridize in itertools.product([True, False], [True, False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            param = prob
            desired_shape = sample_shape + (batch_shape if batch_shape is not None else ())
            samples = np.random.choice(event_shape, size=desired_shape)
            samples = one_hot(samples, event_shape)
            if use_logit:
                param = np.log(param)
            net = TestOneHotCategorical("log_prob", use_logit, batch_shape, event_shape)
            if hybridize:
                net.hybridize()
            mx_out = net(param, samples)
            # Check shape
            assert mx_out.shape == desired_shape

    # Test enumerate support
    for event_shape, batch_shape in itertools.product(event_shapes, batch_shapes):
        for use_logit, hybridize in itertools.product([True, False], [True, False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            param = prob
            if use_logit:
                param = np.log(param)
            net = TestOneHotCategorical("enumerate_support", use_logit, batch_shape, event_shape)
            if hybridize:
                net.hybridize()
            mx_out = net(param).asnumpy()
            desired_shape = batch_shape if batch_shape is not None else ()
            assert mx_out.shape == (event_shape,) + desired_shape + (event_shape,)

            

@with_seed()
@use_np
def test_gluon_half_normal():
    class TestHalfNormal(HybridBlock):
        def __init__(self, func):
            super(TestHalfNormal, self).__init__()
            self._func = func

        def hybrid_forward(self, F, scale, *args):
            half_normal = mgp.HalfNormal(scale, F)
            return getattr(half_normal, self._func)(*args)

    shapes = [(), (1,), (2, 3), 6]

    # Test sampling
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        net = TestHalfNormal("sample")
        if hybridize:
            net.hybridize()
        mx_out = net(scale).asnumpy()
        if isinstance(shape, Number):
            shape = (shape,)
        assert mx_out.shape == shape

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.abs(np.random.normal(size=shape))
        net = TestHalfNormal("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(scale, samples).asnumpy()
        np_out = ss.halfnorm(0, scale.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False) 
    
    # Test cdf

    # Test icdf


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


@use_np
def test_cached_property():
    x = np.random.normal()
    x.attach_grad()
    scale = 0.1

    class Dummy(object):
        def __init__(self, x):
            super(Dummy, self).__init__()
            self.x = x

        @mgp.cached_property
        def y(self):
            return scale * self.x + 1

    with autograd.record():
        obj = Dummy(x)
        obj.y.backward()
    assert_almost_equal(x.grad.asnumpy(), scale * np.ones((1,)))

    class DummyBlock(HybridBlock):
        def hybrid_forward(self, F, x):
            obj = Dummy(x)
            return obj.y

    x = np.random.normal()
    x.attach_grad()
    net = DummyBlock()
    with autograd.record():
        y = net(x)
        y.backward()
    assert_almost_equal(x.grad.asnumpy(), scale * np.ones((1,)))

    x = np.random.normal()
    x.attach_grad()
    net.hybridize()
    with autograd.record():
        y = net(x)
        y.backward()
    assert_almost_equal(x.grad.asnumpy(), scale * np.ones((1,)))


@use_np
def test_independent():
    class TestIndependent(HybridBlock):
        def __init__(self, event_dim, func):
            super(TestIndependent, self).__init__()
            self._event_dim = event_dim
            self._func = func

        def hybrid_forward(self, F, logit, *args):
            base_dist = mgp.Bernoulli(logit=logit)
            reshaped_dist = mgp.Independent(base_dist, self._event_dim)
            return getattr(reshaped_dist, self._func)(*args)

    event_shapes = [(1,), (4,), (2, 2)]
    batch_shapes = [(2, 3), (2,)]
    for (batch_shape, event_shape) in itertools.product(batch_shapes, event_shapes):
        for hybridize in [False, True]:
            for func in ['log_prob']:
                full_shape = batch_shape + event_shape
                logit = np.random.normal(0,2, size=full_shape)
                samples = np.round(np.random.uniform(size=full_shape))
                net = TestIndependent(len(event_shape), func)
                if hybridize:
                    net.hybridize()
                mx_out = net(logit, samples)
                assert mx_out.shape == batch_shape


if __name__ == '__main__':
    import nose
    nose.runmodule()
