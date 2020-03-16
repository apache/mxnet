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
from mxnet import np, npx, autograd
from mxnet import gluon
import mxnet.gluon.probability as mgp
from mxnet.gluon.probability import StochasticBlock, StochasticSequential
from mxnet.gluon import HybridBlock
from mxnet.test_utils import use_np, assert_almost_equal, set_default_context
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

# set_default_context(mx.gpu(0))

def _distribution_method_invoker(dist, func, *args):
    """Wrapper for invoking different types of class methods
    with one unified interface.
    
    Parameters
    ----------
    dist : Distribution
    func : method
    """
    if (len(args) == 0):
        out = getattr(dist, func)
        if callable(out):
            return out()
        else:
            return out
    return getattr(dist, func)(*args)


@with_seed()
@use_np
def test_gluon_uniform():
    class TestUniform(HybridBlock):
        def __init__(self, func):
            super(TestUniform, self).__init__()
            self._func = func

        def hybrid_forward(self, F, low, high, *args):
            uniform = mgp.Uniform(low, high)
            return _distribution_method_invoker(uniform, self._func, *args)

    shapes = [(), (1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        low = np.random.uniform(-1, 1, shape)
        high = low + np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(low, high)
        net = TestUniform("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(low, high, samples).asnumpy()
        np_out = ss.uniform(low.asnumpy(),
                            (high - low).asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test cdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        low = np.random.uniform(-1, 1, shape)
        high = low + np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(low, high)
        net = TestUniform("cdf")
        if hybridize:
            net.hybridize()
        mx_out = net(low, high, samples).asnumpy()
        np_out = ss.uniform(low.asnumpy(),
                            (high - low).asnumpy()).cdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test icdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        low = np.random.uniform(-1, 1, shape)
        high = low + np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(size=shape)
        net = TestUniform("icdf")
        if hybridize:
            net.hybridize()
        mx_out = net(low, high, samples).asnumpy()
        np_out = ss.uniform(low.asnumpy(),
                            (high - low).asnumpy()).ppf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test entropy
    for shape, hybridize in itertools.product(shapes, [True, False]):
        low = np.random.uniform(-1, 1, shape)
        high = low + np.random.uniform(0.5, 1.5, shape)
        net = TestUniform("entropy")
        if hybridize:
            net.hybridize()
        mx_out = net(low, high).asnumpy()
        np_out = ss.uniform(low.asnumpy(),
                            (high - low).asnumpy()).entropy()
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)


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
def test_gluon_cauchy():
    class TestCauchy(HybridBlock):
        def __init__(self, func):
            self._func = func
            super(TestCauchy, self).__init__()

        def hybrid_forward(self, F, loc, scale, *args):
            cauchy = mgp.Cauchy(loc, scale, F)
            return _distribution_method_invoker(cauchy, self._func, *args)

    shapes = [(), (1,), (2, 3), 6]

    # Test sampling
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.normal(size=shape)
        net = TestCauchy("sample")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale)
        desired_shape = (shape,) if isinstance(shape, Number) else shape
        assert mx_out.shape == desired_shape

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.normal(size=shape)
        net = TestCauchy("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale, samples).asnumpy()
        np_out = ss.cauchy(loc.asnumpy(),
                        scale.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test cdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.normal(size=shape)
        net = TestCauchy("cdf")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale, samples).asnumpy()
        np_out = ss.cauchy(loc.asnumpy(),
                        scale.asnumpy()).cdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test icdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(size=shape)
        net = TestCauchy("icdf")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale, samples).asnumpy()
        np_out = ss.cauchy(loc.asnumpy(),
                        scale.asnumpy()).ppf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test entropy
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        net = TestCauchy("entropy")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale).asnumpy()
        np_out = ss.cauchy(loc.asnumpy(),
                        scale.asnumpy()).entropy()
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_gluon_half_cauchy():
    class TestHalfCauchy(HybridBlock):
        def __init__(self, func):
            super(TestHalfCauchy, self).__init__()
            self._func = func

        def hybrid_forward(self, F, scale, *args):
            half_normal = mgp.HalfCauchy(scale, F)
            return getattr(half_normal, self._func)(*args)

    shapes = [(), (1,), (2, 3), 6]

    # Test sampling
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        net = TestHalfCauchy("sample")
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
        net = TestHalfCauchy("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(scale, samples).asnumpy()
        np_out = ss.halfcauchy(0, scale.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False) 
    
    # Test cdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.abs(np.random.normal(size=shape))
        net = TestHalfCauchy("cdf")
        if hybridize:
            net.hybridize()
        mx_out = net(scale, samples).asnumpy()
        np_out = ss.halfcauchy(0, scale.asnumpy()).cdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False) 

    # Test icdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(size=shape)
        net = TestHalfCauchy("icdf")
        if hybridize:
            net.hybridize()
        mx_out = net(scale, samples).asnumpy()
        np_out = ss.halfcauchy(0, scale.asnumpy()).ppf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False) 


@with_seed()
@use_np
def test_gluon_exponential():
    class TestExponential(HybridBlock):
        def __init__(self, func):
            self._func = func
            super(TestExponential, self).__init__()
        
        def hybrid_forward(self, F, scale, *args):
            exponential = mgp.Exponential(scale, F)
            return _distribution_method_invoker(exponential, self._func, *args)
    
    shapes = [(), (1,), (2, 3), 6]
    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(0.2, 1.2, size=shape)
        net = TestExponential("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(scale, samples).asnumpy()
        np_out = ss.expon(scale=scale.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test cdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(0.2, 1.2, size=shape)
        net = TestExponential("cdf")
        if hybridize:
            net.hybridize()
        mx_out = net(scale, samples).asnumpy()
        np_out = ss.expon(scale=scale.asnumpy()).cdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test icdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(0.0, 1.0, size=shape)
        net = TestExponential("icdf")
        if hybridize:
            net.hybridize()
        mx_out = net(scale, samples).asnumpy()
        np_out = ss.expon(scale=scale.asnumpy()).ppf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test entropy
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        net = TestExponential("entropy")
        if hybridize:
            net.hybridize()
        mx_out = net(scale).asnumpy()
        np_out = ss.expon(scale=scale.asnumpy()).entropy()
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_gluon_weibull():
    class TestWeibull(HybridBlock):
        def __init__(self, func):
            super(TestWeibull, self).__init__()
            self._func = func

        def hybrid_forward(self, F, concentration, scale, *args):
            weibull = mgp.Weibull(concentration, scale, F)
            return _distribution_method_invoker(weibull, self._func, *args)

    shapes = [(), (1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        concentration = np.random.uniform(size=shape)
        scale = np.random.uniform(size=shape)
        samples = np.random.uniform(size=shape)
        net = TestWeibull("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(concentration, scale, samples).asnumpy()
        np_out = ss.weibull_min(c=concentration.asnumpy(), scale=scale.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test cdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        concentration = np.random.uniform(size=shape)
        scale = np.random.uniform(size=shape)
        samples = np.random.uniform(size=shape)
        net = TestWeibull("cdf")
        if hybridize:
            net.hybridize()
        mx_out = net(concentration, scale, samples).asnumpy()
        np_out = ss.weibull_min(c=concentration.asnumpy(), scale=scale.asnumpy()).cdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test icdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        concentration = np.random.uniform(size=shape)
        scale = np.random.uniform(size=shape)
        samples = np.random.uniform(size=shape)
        net = TestWeibull("icdf")
        if hybridize:
            net.hybridize()
        mx_out = net(concentration, scale, samples).asnumpy()
        np_out = ss.weibull_min(c=concentration.asnumpy(), scale=scale.asnumpy()).ppf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test entropy
    for shape, hybridize in itertools.product(shapes, [True, False]):
        concentration = np.random.uniform(size=shape)
        scale = np.random.uniform(size=shape)
        net = TestWeibull("entropy")
        if hybridize:
            net.hybridize()
        mx_out = net(concentration, scale).asnumpy()
        np_out = ss.weibull_min(c=concentration.asnumpy(), scale=scale.asnumpy()).entropy()
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_gluon_pareto():
    class TestPareto(HybridBlock):
        def __init__(self, func):
            super(TestPareto, self).__init__()
            self._func = func

        def hybrid_forward(self, F, alpha, scale, *args):
            pareto = mgp.Pareto(alpha, scale, F)
            return _distribution_method_invoker(pareto, self._func, *args)

    shapes = [(), (1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        alpha = np.random.uniform(size=shape)
        scale = np.random.uniform(size=shape)
        samples = np.random.uniform(1, 2, size=shape)
        net = TestPareto("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(alpha, scale, samples).asnumpy()
        np_out = ss.pareto(b=alpha.asnumpy(), scale=scale.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test cdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        alpha = np.random.uniform(size=shape)
        scale = np.random.uniform(size=shape)
        samples = np.random.uniform(1.0, 2.0, size=shape)
        net = TestPareto("cdf")
        if hybridize:
            net.hybridize()
        mx_out = net(alpha, scale, samples).asnumpy()
        np_out = ss.pareto(b=alpha.asnumpy(), scale=scale.asnumpy()).cdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test icdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        alpha = np.random.uniform(size=shape)
        scale = np.random.uniform(size=shape)
        samples = np.random.uniform(size=shape)
        net = TestPareto("icdf")
        if hybridize:
            net.hybridize()
        mx_out = net(alpha, scale, samples).asnumpy()
        np_out = ss.pareto(b=alpha.asnumpy(), scale=scale.asnumpy()).ppf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test entropy
    for shape, hybridize in itertools.product(shapes, [True, False]):
        alpha = np.random.uniform(size=shape)
        scale = np.random.uniform(size=shape)
        net = TestPareto("entropy")
        if hybridize:
            net.hybridize()
        mx_out = net(alpha, scale).asnumpy()
        np_out = ss.pareto(b=alpha.asnumpy(), scale=scale.asnumpy()).entropy()
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_gluon_gamma():
    class TestGamma(HybridBlock):
        def __init__(self, func):
            super(TestGamma, self).__init__()
            self._func = func

        def hybrid_forward(self, F, shape, scale, *args):
            gamma = mgp.Gamma(shape, scale, F)
            return _distribution_method_invoker(gamma, self._func, *args)

    shapes = [(), (1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        alpha = np.random.uniform(0.5, 1.5, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(size=shape)
        net = TestGamma("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(alpha, scale, samples).asnumpy()
        np_out = ss.gamma(a=alpha.asnumpy(), loc=0, scale=scale.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test `mean` and `var`
    for shape, hybridize in itertools.product(shapes, [True, False]):
        for func in ['mean', 'variance']:
            alpha = np.random.uniform(0.5, 1.5, shape)
            scale = np.random.uniform(0.5, 1.5, shape)
            net = TestGamma(func)
            if hybridize:
                net.hybridize()
            mx_out = net(alpha, scale).asnumpy()
            ss_gamma = ss.gamma(a=alpha.asnumpy(), loc=0, scale=scale.asnumpy())
            if func == 'mean':
                np_out = ss_gamma.mean()
            else:
                np_out = ss_gamma.var()
            assert_almost_equal(mx_out, np_out, atol=1e-4,
                                rtol=1e-3, use_broadcast=False)

                        
@with_seed()
@use_np
def test_gluon_dirichlet():
    class TestDirichlet(HybridBlock):
        def __init__(self, func):
            super(TestDirichlet, self).__init__()
            self._func = func

        def hybrid_forward(self, F, alpha, *args):
            dirichlet = mgp.Dirichlet(alpha, F)
            return _distribution_method_invoker(dirichlet, self._func, *args)

    event_shapes = [2, 5, 10]
    batch_shapes = [None, (2, 3), (4, 0, 5)]

    # Test sampling
    for event_shape, batch_shape in itertools.product(event_shapes, batch_shapes):
        for hybridize in [True, False]:
            desired_shape = (batch_shape if batch_shape is not None else ()) + (event_shape,)
            alpha = np.random.uniform(size=desired_shape)
            net = TestDirichlet("sample")
            if hybridize:
                net.hybridize()
            mx_out = net(alpha).asnumpy()
            # Check shape
            assert mx_out.shape == desired_shape
            # Check simplex
            assert_almost_equal(mx_out.sum(-1), _np.ones_like(mx_out.sum(-1)), atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test log_prob
    for event_shape, batch_shape in itertools.product(event_shapes, batch_shapes[:1]):
        for hybridize in [True, False]:
            desired_shape = (batch_shape if batch_shape is not None else ()) + (event_shape,)
            alpha = np.random.uniform(size=desired_shape)
            np_samples = _np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape)
            net = TestDirichlet("log_prob")
            if hybridize:
                net.hybridize()
            mx_out = net(alpha, np.array(np_samples)).asnumpy()
            np_out = ss.dirichlet(alpha=alpha.asnumpy()).logpdf(np_samples)
            assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_gluon_beta():
    class TestBeta(HybridBlock):
        def __init__(self, func):
            super(TestBeta, self).__init__()
            self._func = func

        def hybrid_forward(self, F, alpha, beta, *args):
            beta_dist = mgp.Beta(alpha, beta, F)
            return _distribution_method_invoker(beta_dist, self._func, *args)

    shapes = [(), (1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        alpha = np.random.uniform(0.5, 1.5, shape)
        beta = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(size=shape)
        net = TestBeta("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(alpha, beta, samples).asnumpy()
        np_out = ss.beta(alpha.asnumpy(), beta.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test `mean` and `var`
    for shape, hybridize in itertools.product(shapes, [True, False]):
        for func in ['mean', 'variance']:
            alpha = np.random.uniform(0.5, 1.5, shape)
            beta = np.random.uniform(0.5, 1.5, shape)
            net = TestBeta(func)
            if hybridize:
                net.hybridize()
            mx_out = net(alpha, beta).asnumpy()
            ss_beta = ss.beta(alpha.asnumpy(), beta.asnumpy())
            if func == 'mean':
                np_out = ss_beta.mean()
            else:
                np_out = ss_beta.var()
            assert_almost_equal(mx_out, np_out, atol=1e-4,
                                rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_gluon_fisher_snedecor():
    class TestFisherSnedecor(HybridBlock):
        def __init__(self, func):
            super(TestFisherSnedecor, self).__init__()
            self._func = func

        def hybrid_forward(self, F, df1, df2, *args):
            beta_dist = mgp.FisherSnedecor(df1, df2, F)
            return _distribution_method_invoker(beta_dist, self._func, *args)

    shapes = [(), (1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        df1 = np.random.uniform(0.5, 1.5, shape)
        df2 = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(size=shape)
        net = TestFisherSnedecor("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(df1, df2, samples).asnumpy()
        np_out = ss.f(dfn=df1.asnumpy(), dfd=df2.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test `mean` and `var`
    for shape, hybridize in itertools.product(shapes, [True, False]):
        for func in ['mean', 'variance']:
            df1 = np.random.uniform(0.5, 1.5, shape)
            df2 = np.random.uniform(4.0, 6.0, shape)
            net = TestFisherSnedecor(func)
            if hybridize:
                net.hybridize()
            mx_out = net(df1, df2).asnumpy()
            ss_f = ss.f(dfn=df1.asnumpy(), dfd=df2.asnumpy())
            if func == 'mean':
                np_out = ss_f.mean()
            else:
                np_out = ss_f.var()
            assert_almost_equal(mx_out, np_out, atol=1e-4,
                                rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_gluon_student_t():
    class TestT(HybridBlock):
        def __init__(self, func):
            super(TestT, self).__init__()
            self._func = func

        def hybrid_forward(self, F, df, loc, scale, *args):
            t_dist = mgp.StudentT(df, loc, scale, F)
            return _distribution_method_invoker(t_dist, self._func, *args)

    shapes = [(), (1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.zeros(shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        df = np.random.uniform(2, 4, shape)
        samples = np.random.uniform(0, 4, size=shape)
        net = TestT("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(df, loc, scale, samples).asnumpy()
        np_out = ss.t(loc=0, scale=scale.asnumpy(), df=df.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test `mean` and `var`
    for shape, hybridize in itertools.product(shapes, [False, True]):
        for func in ['mean', 'variance']:
            loc = np.zeros(shape)
            scale = np.random.uniform(0.5, 1.5, shape)
            df = np.random.uniform(3, 4, shape)
            net = TestT(func)
            if hybridize:
                net.hybridize()
            mx_out = net(df, loc, scale).asnumpy()
            ss_f = ss.t(loc=0, scale=scale.asnumpy(), df=df.asnumpy())
            if func == 'mean':
                np_out = ss_f.mean()
            else:
                np_out = ss_f.var()
            assert_almost_equal(mx_out, np_out, atol=1e-4,
                                rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_gluon_gumbel():
    class TestGumbel(HybridBlock):
        def __init__(self, func):
            super(TestGumbel, self).__init__()
            self._func = func

        def hybrid_forward(self, F, loc, scale, *args):
            normal = mgp.Gumbel(loc, scale, F)
            return getattr(normal, self._func)(*args)

    shapes = [(), (1,), (2, 3), 6]

    # Test log_prob
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.normal(size=shape)
        net = TestGumbel("log_prob")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale, samples).asnumpy()
        import torch
        from torch.distributions import Gumbel
        np_out = ss.gumbel_r(loc=loc.asnumpy(),
                        scale=scale.asnumpy()).logpdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test cdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.normal(size=shape)
        net = TestGumbel("cdf")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale, samples).asnumpy()
        np_out = ss.gumbel_r(loc.asnumpy(),
                        scale.asnumpy()).cdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test icdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(size=shape)
        net = TestGumbel("icdf")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale, samples).asnumpy()
        np_out = ss.gumbel_r(loc.asnumpy(),
                        scale.asnumpy()).ppf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)

    # Test entropy
    for shape, hybridize in itertools.product(shapes, [True, False]):
        loc = np.random.uniform(-1, 1, shape)
        scale = np.random.uniform(0.5, 1.5, shape)
        net = TestGumbel("entropy")
        if hybridize:
            net.hybridize()
        mx_out = net(loc, scale).asnumpy()
        np_out = ss.gumbel_r(loc.asnumpy(),
                        scale.asnumpy()).entropy()
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False)


@with_seed()
@use_np
def test_gluon_multinomial():
    pass


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
            return _distribution_method_invoker(bernoulli, self._func, *args)

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
def test_relaxed_bernoulli():
    class TestRelaxedBernoulli(HybridBlock):
        def __init__(self, func, is_logit=False):
            super(TestRelaxedBernoulli, self).__init__()
            self._is_logit = is_logit
            self._func = func

        def hybrid_forward(self, F, params, *args):
            relaxed_bernoulli = mgp.RelaxedBernoulli(T=1.0, logit=params, F=F) if self._is_logit else \
                        mgp.RelaxedBernoulli(T=1.0, prob=params, F=F)
            if self._func == "sample":
                return relaxed_bernoulli.sample()
            return _distribution_method_invoker(relaxed_bernoulli, self._func, *args)

    def prob_to_logit(prob):
        return np.log(prob) - np.log1p(-prob)

    shapes = [(), (1,), (2, 3), 6]
    # Test sampling
    for shape, hybridize, use_logit in itertools.product(shapes, [True, False], [True, False]):
        prob = np.random.uniform(size=shape)
        param = prob
        if use_logit:
            param = prob_to_logit(param)
        param.attach_grad()
        net = TestRelaxedBernoulli("sample", use_logit)
        if hybridize:
            net.hybridize()
        with autograd.record():
            mx_out = net(param)
        mx_out.backward()
        desired_shape = (shape,) if isinstance(shape, int) else shape
        assert param.grad.shape == desired_shape

    for shape, hybridize, use_logit in itertools.product(shapes, [True, False], [True, False]):
        prob = np.random.uniform(size=shape)
        sample = np.random.uniform(0.1, 0.9, size=shape)
        param = prob
        if use_logit:
            param = prob_to_logit(param)
        net = TestRelaxedBernoulli("log_prob", use_logit)
        if hybridize:
            net.hybridize()
        mx_out = net(param, sample).asnumpy()
        desired_shape = (shape,) if isinstance(shape, int) else shape
        assert mx_out.shape == desired_shape


@with_seed()
@use_np
def test_gluon_categorical():
    class TestCategorical(HybridBlock):
        def __init__(self, func, is_logit=False, batch_shape=None, num_events=None, sample_shape=None):
            super(TestCategorical, self).__init__()
            self._is_logit = is_logit
            self._func = func
            self._batch_shape = batch_shape
            self._num_events = num_events
            self._sample_shape = sample_shape

        def hybrid_forward(self, F, params, *args):
            categorical = mgp.Categorical(self._num_events, logit=params, F=F) if self._is_logit else \
                        mgp.Categorical(self._num_events, prob=params, F=F)
            if self._func == "sample":
                return categorical.sample(self._batch_shape)
            if self._func == "sample_n":
                return categorical.sample_n(self._sample_shape)
            return _distribution_method_invoker(categorical, self._func, *args)

    event_shapes = [2, 5, 10]
    batch_shapes = [None, (2, 3)] #, (4, 0, 5)]
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
    
    # Test sample_n
    for event_shape, batch_shape, sample_shape in itertools.product(event_shapes, batch_shapes, sample_shapes):
        for use_logit, hybridize in itertools.product([True, False], [True, False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            param = prob
            if use_logit:
                param = np.log(param)
            net = TestCategorical("sample_n",
                                is_logit=False, batch_shape=batch_shape,
                                num_events=event_shape, sample_shape=sample_shape
                    )
            if hybridize:
                net.hybridize()
            mx_out = net(param).asnumpy()
            desired_shape = sample_shape + (batch_shape if batch_shape is not None else ())
            assert mx_out.shape == desired_shape

    # Test log_prob
    for event_shape, batch_shape, sample_shape in itertools.product(event_shapes, batch_shapes, sample_shapes):
        for use_logit, hybridize in itertools.product([True, False], [True, False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            eps = _np.finfo('float32').eps
            prob = np.clip(prob, eps, 1 - eps)
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
            return _distribution_method_invoker(categorical, self._func, *args)

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
            eps = _np.finfo('float32').eps
            prob = np.clip(prob, eps, 1 - eps)
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
def test_relaxed_one_hot_categorical():
    class TestRelaxedOneHotCategorical(HybridBlock):
        def __init__(self, func, is_logit=False, batch_shape=None, num_events=None):
                super(TestRelaxedOneHotCategorical, self).__init__()
                self._is_logit = is_logit
                self._func = func
                self._batch_shape = batch_shape
                self._num_events = num_events

        def hybrid_forward(self, F, params, *args):
            categorical = mgp.RelaxedOneHotCategorical(T=1.0, num_events=self._num_events, logit=params, F=F) \
                            if self._is_logit else \
                            mgp.RelaxedOneHotCategorical(T=1.0, num_events=self._num_events, prob=params, F=F)
            if self._func == "sample":
                return categorical.sample(self._batch_shape)
            return _distribution_method_invoker(categorical, self._func, *args)

    event_shapes = [2, 5, 10]
    batch_shapes = [None, (2, 3)] #, (4, 0, 5)]
    sample_shapes = [(), (2,), (3, 4)]

    # Test sampling
    for event_shape, batch_shape in itertools.product(event_shapes, batch_shapes):
        for use_logit, hybridize in itertools.product([True, False], [True, False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            param = prob
            if use_logit:
                param = np.log(param)
            param.attach_grad()
            net = TestRelaxedOneHotCategorical("sample", use_logit, batch_shape, event_shape)
            if hybridize:
                net.hybridize()
            with autograd.record():
                mx_out = net(param)
            mx_out.backward()
            desired_shape = batch_shape if batch_shape is not None else ()
            assert mx_out.shape == desired_shape + (event_shape,)
            assert param.grad.shape == param.shape

    # Test log_prob
    for event_shape, batch_shape, sample_shape in itertools.product(event_shapes, batch_shapes, sample_shapes):
        for use_logit, hybridize in itertools.product([True, False], [False]):
            prob = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=batch_shape))
            eps = _np.finfo('float32').eps
            prob = np.clip(prob, eps, 1 - eps)
            param = prob
            desired_shape = sample_shape + (batch_shape if batch_shape is not None else ())
            # Samples from a Relaxed One-hot Categorical lie on a simplex.
            samples = np.array(_np.random.dirichlet([1 / event_shape] * event_shape, size=desired_shape))
            if use_logit:
                param = np.log(param)
            net = TestRelaxedOneHotCategorical("log_prob", use_logit, batch_shape, event_shape)
            if hybridize:
                net.hybridize()
            mx_out = net(param, samples)
            # Check shape
            assert mx_out.shape == desired_shape    


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
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.abs(np.random.normal(size=shape))
        net = TestHalfNormal("cdf")
        if hybridize:
            net.hybridize()
        mx_out = net(scale, samples).asnumpy()
        np_out = ss.halfnorm(0, scale.asnumpy()).cdf(samples.asnumpy())
        assert_almost_equal(mx_out, np_out, atol=1e-4,
                            rtol=1e-3, use_broadcast=False) 

    # Test icdf
    for shape, hybridize in itertools.product(shapes, [True, False]):
        scale = np.random.uniform(0.5, 1.5, shape)
        samples = np.random.uniform(size=shape)
        net = TestHalfNormal("icdf")
        if hybridize:
            net.hybridize()
        mx_out = net(scale, samples).asnumpy()
        np_out = ss.halfnorm(0, scale.asnumpy()).ppf(samples.asnumpy())
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
    for shape, hybridize in itertools.product(shapes, [True, False]): 
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
        np_out = ss.lognorm(s=scale.asnumpy(), scale=np.exp(loc).asnumpy()).logpdf(samples.asnumpy())
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

@with_seed()
@use_np
def test_gluon_stochastic_block():
    class dummyBlock(StochasticBlock):
        """In this test case, we generate samples from a Gaussian
        parameterized by `loc` and `scale` and accumulate the KL-divergence
        between it and its prior into the block's loss storage
        """
        @StochasticBlock.collectLoss
        def hybrid_forward(self, F, loc, scale):
            qz = mgp.Normal(loc, scale)
            # prior
            pz = mgp.Normal(F.np.zeros_like(loc), F.np.ones_like(scale))
            self.add_loss(mgp.kl_divergence(qz, pz))
            return qz.sample()

    shape = (4, 4)
    for hybridize in [True, False]:
        net = dummyBlock()
        if hybridize:
            net.hybridize()
        loc = np.random.randn(*shape)
        scale = np.random.rand(*shape)
        mx_out = net(loc, scale).asnumpy()
        kl = net.losses[0].asnumpy()
        assert mx_out.shape == loc.shape
        assert kl.shape == loc.shape

@with_seed()
@use_np
def test_gluon_stochastic_sequential():
    class normalBlock(HybridBlock):
        def hybrid_forward(self, F, x):
            return (x + 1)

    class stochasticBlock(StochasticBlock):
        @StochasticBlock.collectLoss
        def hybrid_forward(self, F, x):
            self.add_loss(x ** 2)
            self.add_loss(x - 1)
            return (x + 1)

    shape = (4, 4)
    for hybridize in [True, False]:
        initial_value = np.ones(shape)
        net = StochasticSequential()
        net.add(stochasticBlock())
        net.add(normalBlock())
        net.add(stochasticBlock())
        net.add(normalBlock())
        if hybridize:
            net.hybridize()
        mx_out = net(initial_value).asnumpy()
        assert_almost_equal(mx_out, _np.ones(shape) * 5)
        accumulated_loss = net.losses
        assert len(accumulated_loss) == 2
        assert_almost_equal(accumulated_loss[0][0].asnumpy(), _np.ones(shape))
        assert_almost_equal(accumulated_loss[0][1].asnumpy(), _np.ones(shape) - 1)
        assert_almost_equal(accumulated_loss[1][0].asnumpy(), _np.ones(shape) * 9)
        assert_almost_equal(accumulated_loss[1][1].asnumpy(), _np.ones(shape) + 1)


if __name__ == '__main__':
    import nose
    nose.runmodule()
