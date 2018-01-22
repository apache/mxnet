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
import mxnet as mx
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf
import numpy as np
import scipy.stats as ss

def same(a, b):
    return np.sum(a != b) == 0

def check_with_device(device, dtype):
    # The thresholds chosen for the tests are too loose. We will rely on the other tests to test the samples from the
    #  generators.
    tol = 0.1
    symbols = [
        {
            'name': 'normal',
            'symbol': mx.sym.random.normal,
            'ndop': mx.nd.random.normal,
            'params': { 'loc': 10.0, 'scale': 0.5 },
            'inputs': [ ('loc',[ [ 0.0, 2.5 ], [ -9.75, -7.0 ] ]) , ('scale',[ [ 1.0, 3.7 ], [ 4.2, 1.5 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64) - params['loc']),  tol),
                ('std',  lambda x, params: np.std(x.astype(np.float64)) - params['scale'], tol)
            ]
        },
        {
            'name': 'uniform',
            'symbol': mx.sym.random.uniform,
            'ndop': mx.nd.random.uniform,
            'params': { 'low': -1.5, 'high': 3.0 },
            'inputs': [ ('low', [ [ 0.0, 2.5 ], [ -9.75, -1.0 ] ]) , ('high', [ [ 1.0, 3.7 ], [ 4.2, 10.5 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - (params['low'] + params['high']) / 2.0, tol),
                ('std', lambda x,  params: np.std(x.astype(np.float64)) - np.sqrt(1.0 / 12.0) * (params['high'] - params['low']), tol)
            ]
        },
        {
                'name': 'gamma',
                'symbol': mx.sym.random.gamma,
                'ndop': mx.nd.random.gamma,
                'params': { 'alpha': 9.0, 'beta': 0.5 },
                'inputs': [ ('alpha', [ [ 0.0, 2.5 ], [ 9.75, 11.0 ] ]) , ('beta', [ [ 1.0, 0.7 ], [ 0.5, 0.3 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['alpha'] * params['beta'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['alpha'] * params['beta'] ** 2), tol)
                ]
            },
            {
                'name': 'exponential',
                'symbol': mx.sym.random.exponential,
                'ndop': mx.nd.random.exponential,
                'params': { 'scale': 1.0/4.0 },
                'inputs': [ ('scale', [ [ 1.0/1.0, 1.0/8.5 ], [ 1.0/2.7 , 1.0/0.5 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['scale'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - params['scale'], tol)
                ]
            },
            {
                'name': 'poisson',
                'symbol': mx.sym.random.poisson,
                'ndop': mx.nd.random.poisson,
                'params': { 'lam': 4.0 },
                'inputs': [ ('lam', [ [ 25.0, 8.5 ], [ 2.7 , 0.5 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['lam'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['lam']), tol)
                ]
            },
            {
                'name': 'neg-binomial',
                'symbol': mx.sym.random.negative_binomial,
                'ndop': mx.nd.random.negative_binomial,
                'params': { 'k': 3, 'p': 0.4 },
                'inputs': [ ('k', [ [ 3, 4 ], [ 5 , 6 ] ]) , ('p', [ [ 0.4 , 0.77 ], [ 0.5, 0.84 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['k'] * (1.0 - params['p']) /  params['p'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['k'] * (1.0 - params['p']))/params['p'], tol)
                ]
            },
            {
                'name': 'gen-neg-binomial',
                'symbol': mx.sym.random.generalized_negative_binomial,
                'ndop': mx.nd.random.generalized_negative_binomial,
                'params': { 'mu': 2.0, 'alpha': 0.3 },
                'inputs': [ ('mu', [ [ 2.0, 2.5 ], [ 1.3, 1.9 ] ]) , ('alpha', [ [ 1.0, 0.1 ], [ 0.2, 0.5 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['mu'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['mu'] + params['alpha'] * params['mu'] ** 2 ), tol)
                ]
            }

        ]

    # Create enough samples such that we get a meaningful distribution.
    shape = (500, 500)
    for symbdic in symbols:
        name = symbdic['name']
        ndop = symbdic['ndop']

        # check directly
        params = symbdic['params'].copy()
        params.update(shape=shape, dtype=dtype, ctx=device)
        mx.random.seed(128)
        ret1 = ndop(**params).asnumpy()
        mx.random.seed(128)
        ret2 = ndop(**params).asnumpy()
        assert same(ret1, ret2), \
                "ndarray test: `%s` should give the same result with the same seed" % name

        for check_name, check_func, tol in symbdic['checks']:
            assert np.abs(check_func(ret1, params)) < tol, "ndarray test: %s check for `%s` did not pass" % (check_name, name)

        # check multi-distribution sampling
        params = {'shape': shape, 'dtype': dtype, 'ctx': device}
        params.update({k : mx.nd.array(v, ctx=device, dtype=dtype) for k, v in symbdic['inputs']})
        mx.random.seed(128)
        ret1 = ndop(**params).asnumpy()
        mx.random.seed(128)
        ret2 = ndop(**params).asnumpy()
        assert same(ret1, ret2), \
                "ndarray test: `%s` should give the same result with the same seed" % name
        for i in range(2):
            for j in range(2):
                stats = {k : v[i][j] for k, v in symbdic['inputs']}
                for check_name, check_func, tol in symbdic['checks']:
                    err = np.abs(check_func(ret2[i,j], stats))
                    assert err < tol, "%f vs %f: symbolic test: %s check for `%s` did not pass" % (err, tol, check_name, name)

        # check symbolic
        symbol = symbdic['symbol']
        X = mx.sym.Variable("X")
        params = symbdic['params'].copy()
        params.update(shape=shape, dtype=dtype)
        Y = symbol(**params) + X
        x = mx.nd.zeros(shape, dtype=dtype, ctx=device)
        xgrad = mx.nd.zeros(shape, dtype=dtype, ctx=device)
        yexec = Y.bind(device, {'X' : x}, {'X': xgrad})
        mx.random.seed(128)
        yexec.forward(is_train=True)
        yexec.backward(yexec.outputs[0])
        un1 = (yexec.outputs[0] - x).copyto(device)
        assert same(xgrad.asnumpy(), un1.asnumpy())
        mx.random.seed(128)
        yexec.forward()
        un2 = (yexec.outputs[0] - x).copyto(device)
        assert same(un1.asnumpy(), un2.asnumpy()), \
                "symbolic test: `%s` should give the same result with the same seed" % name

        ret1 = un1.asnumpy()
        for check_name, check_func, tol in symbdic['checks']:
            assert np.abs(check_func(ret1, params)) < tol, "symbolic test: %s check for `%s` did not pass" % (check_name, name)

        # check multi-distribution sampling
        symbol = symbdic['symbol']
        params = { 'shape' : shape, 'dtype' : dtype }
        single_param = len(symbdic['inputs']) == 1;
        v1 = mx.sym.Variable('v1')
        v2 = mx.sym.Variable('v2')
        Y = symbol(v1,**params) if single_param else symbol(v1,v2,**params)
        bindings = { 'v1' : mx.nd.array(symbdic['inputs'][0][1]) }
        if not single_param :
            bindings.update({ 'v2' : mx.nd.array(symbdic['inputs'][1][1]) })
        yexec = Y.bind(ctx=device, args=bindings)
        yexec.forward()
        un1 = yexec.outputs[0].copyto(device).asnumpy()
        params = {}
        for i, r in enumerate(symbdic['inputs'][0][1]):
            for j, p1 in enumerate(r):
                params.update({ symbdic['inputs'][0][0] : p1 })
                if not single_param:
                   params.update({ symbdic['inputs'][1][0] : symbdic['inputs'][1][1][i][j] })
                samples = un1[i,j]
                for check_name, check_func, tol in symbdic['checks']:
                    assert np.abs(check_func(samples, params)) < tol, "symbolic test: %s check for `%s` did not pass" % (check_name, name)

def test_random():
    check_with_device(mx.context.current_context(), 'float16')
    check_with_device(mx.context.current_context(), 'float32')
    check_with_device(mx.context.current_context(), 'float64')


# Set seed variously based on `start_seed` and `num_init_seeds`, then set seed finally to `final_seed`
def set_seed_variously(init_seed, num_init_seeds, final_seed):
    end_seed = init_seed + num_init_seeds
    for seed in range(init_seed, end_seed):
        mx.random.seed(seed)
    mx.random.seed(final_seed)
    return end_seed

# Tests that seed setting of std (non-parallel) rng is synchronous w.r.t. rng use before and after.
def test_random_seed_setting():
    ctx = mx.context.current_context()
    seed_to_test = 1234
    num_temp_seeds = 25
    probs = [0.125, 0.25, 0.25, 0.0625, 0.125, 0.1875]
    num_samples = 100000
    for dtype in ['float16', 'float32', 'float64']:
        seed = set_seed_variously(1, num_temp_seeds, seed_to_test)
        samples1 = mx.nd.random.multinomial(data=mx.nd.array(probs, ctx=ctx, dtype=dtype),
                                            shape=num_samples)
        seed = set_seed_variously(seed, num_temp_seeds, seed_to_test)
        samples2 = mx.nd.random.multinomial(data=mx.nd.array(probs, ctx=ctx, dtype=dtype),
                                            shape=num_samples)
        samples1np = samples1.asnumpy()
        set_seed_variously(seed, num_temp_seeds, seed_to_test+1)
        samples2np = samples2.asnumpy()
        assert same(samples1np, samples2np), \
            "seed-setting test: `multinomial` should give the same result with the same seed"


# Tests that seed setting of parallel rng is synchronous w.r.t. rng use before and after.
def test_parallel_random_seed_setting():
    ctx = mx.context.current_context()
    seed_to_test = 1234
    for dtype in ['float16', 'float32', 'float64']:
        # Avoid excessive test cpu runtimes
        num_temp_seeds = 25 if ctx.device_type == 'gpu' else 1
        # To flush out a possible race condition, run multiple times
        for _ in range(20):
            # Create enough samples such that we get a meaningful distribution.
            shape = (200, 200)
            params = { 'low': -1.5, 'high': 3.0 }
            params.update(shape=shape, dtype=dtype, ctx=ctx)

            # check directly
            seed = set_seed_variously(1, num_temp_seeds, seed_to_test)
            ret1 = mx.nd.random.uniform(**params)
            seed = set_seed_variously(seed, num_temp_seeds, seed_to_test)
            ret2 = mx.nd.random.uniform(**params)
            seed = set_seed_variously(seed, num_temp_seeds, seed_to_test)
            assert same(ret1.asnumpy(), ret2.asnumpy()), \
                "ndarray seed-setting test: `uniform` should give the same result with the same seed"

            # check symbolic
            X = mx.sym.Variable("X")
            Y = mx.sym.random.uniform(**params) + X
            x = mx.nd.zeros(shape, dtype=dtype, ctx=ctx)
            xgrad = mx.nd.zeros(shape, dtype=dtype, ctx=ctx)
            yexec = Y.bind(ctx, {'X' : x}, {'X': xgrad})
            seed = set_seed_variously(seed, num_temp_seeds, seed_to_test)
            yexec.forward(is_train=True)
            yexec.backward(yexec.outputs[0])
            un1 = (yexec.outputs[0] - x).copyto(ctx)
            seed = set_seed_variously(seed, num_temp_seeds, seed_to_test)
            yexec.forward()
            set_seed_variously(seed, num_temp_seeds, seed_to_test)
            un2 = (yexec.outputs[0] - x).copyto(ctx)
            assert same(un1.asnumpy(), un2.asnumpy()), \
                "symbolic seed-setting test: `uniform` should give the same result with the same seed"


def test_sample_multinomial():
    x = mx.nd.array([[0,1,2,3,4],[4,3,2,1,0]])/10.0
    dx = mx.nd.ones_like(x)
    mx.contrib.autograd.mark_variables([x], [dx])
    with mx.autograd.record():
        y, prob = mx.nd.random.multinomial(x, shape=1000, get_prob=True)
        r = prob * 5
        r.backward()

    y = y.asnumpy()
    x = x.asnumpy()
    for i in range(x.shape[0]):

        freq = np.bincount(y[i], minlength=5)/1000.0*x[i].sum()
        mx.test_utils.assert_almost_equal(freq, x[i], rtol=0.25)
        rprob = x[i][y[i]]/x[i].sum()
        mx.test_utils.assert_almost_equal(np.log(rprob), prob.asnumpy()[i])

        real_dx = np.zeros((5,))
        for j in range(1000):
            real_dx[y[i][j]] += 5.0 / rprob[j]
        mx.test_utils.assert_almost_equal(real_dx, dx.asnumpy()[i])

# Test the generators with the chi-square testing
def test_normal_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        for mu, sigma in [(0.0, 1.0), (1.0, 5.0)]:
            print("ctx=%s, dtype=%s, Mu=%g, Sigma=%g:" % (ctx, dtype, mu, sigma))
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.norm.ppf(x, mu, sigma), 5)
            generator_mx = lambda x: mx.nd.random.normal(mu, sigma, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs)
            generator_mx_same_seed =\
                lambda x: np.concatenate(
                    [mx.nd.random.normal(mu, sigma, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                     for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)

def test_uniform_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        for low, high in [(-1.0, 1.0), (1.0, 3.0)]:
            print("ctx=%s, dtype=%s, Low=%g, High=%g:" % (ctx, dtype, low, high))
            scale = high - low
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.uniform.ppf(x, loc=low, scale=scale), 5)
            # Quantize bucket boundaries to reflect the actual dtype and adjust probs accordingly
            buckets = np.array(buckets, dtype=dtype).tolist()
            probs = [(buckets[i][1] - buckets[i][0])/scale for i in range(5)]
            generator_mx = lambda x: mx.nd.random.uniform(low, high, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs)
            generator_mx_same_seed = \
                lambda x: np.concatenate(
                    [mx.nd.random.uniform(low, high, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                     for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)

def test_gamma_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        for kappa, theta in [(0.5, 1.0), (1.0, 5.0)]:
            print("ctx=%s, dtype=%s, Shape=%g, Scale=%g:" % (ctx, dtype, kappa, theta))
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.gamma.ppf(x, a=kappa, loc=0, scale=theta), 5)
            generator_mx = lambda x: mx.nd.random.gamma(kappa, theta, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs)
            generator_mx_same_seed = \
                lambda x: np.concatenate(
                    [mx.nd.random.gamma(kappa, theta, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                     for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)

def test_exponential_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        for scale in [0.1, 1.0]:
            print("ctx=%s, dtype=%s, Scale=%g:" % (ctx, dtype, scale))
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.expon.ppf(x, loc=0, scale=scale), 5)
            generator_mx = lambda x: mx.nd.random.exponential(scale, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs)
            generator_mx_same_seed = \
                lambda x: np.concatenate(
                    [mx.nd.random.exponential(scale, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                     for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)

def test_poisson_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        for lam in [1, 10]:
            print("ctx=%s, dtype=%s, Lambda=%d:" % (ctx, dtype, lam))
            buckets = [(-1.0, lam - 0.5), (lam - 0.5, 2 * lam + 0.5), (2 * lam + 0.5, np.inf)]
            probs = [ss.poisson.cdf(bucket[1], lam) - ss.poisson.cdf(bucket[0], lam) for bucket in buckets]
            generator_mx = lambda x: mx.nd.random.poisson(lam, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs)
            generator_mx_same_seed = \
                lambda x: np.concatenate(
                    [mx.nd.random.poisson(lam, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                     for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)

def test_negative_binomial_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        success_num = 2
        success_prob = 0.2
        print("ctx=%s, dtype=%s, Success Num=%d:, Success Prob=%g" % (ctx, dtype, success_num, success_prob))
        buckets = [(-1.0, 2.5), (2.5, 5.5), (5.5, 8.5), (8.5, np.inf)]
        probs = [ss.nbinom.cdf(bucket[1], success_num, success_prob) -
                 ss.nbinom.cdf(bucket[0], success_num, success_prob) for bucket in buckets]
        generator_mx = lambda x: mx.nd.random.negative_binomial(success_num, success_prob,
                                                                shape=x, ctx=ctx, dtype=dtype).asnumpy()
        verify_generator(generator=generator_mx, buckets=buckets, probs=probs)
        generator_mx_same_seed = \
            lambda x: np.concatenate(
                [mx.nd.random.negative_binomial(success_num, success_prob, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                 for _ in range(10)])
        verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)
        # Also test the Gamm-Poisson Mixture
        print('Gamm-Poisson Mixture Test:')
        alpha = 1.0 / success_num
        mu = (1.0 - success_prob) / success_prob / alpha
        generator_mx = lambda x: mx.nd.random.generalized_negative_binomial(mu, alpha,
                                                                            shape=x, ctx=ctx, dtype=dtype).asnumpy()
        verify_generator(generator=generator_mx, buckets=buckets, probs=probs)
        generator_mx_same_seed = \
            lambda x: np.concatenate(
                [mx.nd.random.generalized_negative_binomial(mu, alpha, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                 for _ in range(10)])
        verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)

def test_multinomial_generator():
    ctx = mx.context.current_context()
    probs = [0.1, 0.2, 0.3, 0.05, 0.15, 0.2]
    buckets = list(range(6))
    for dtype in ['float16', 'float32', 'float64']:
        print("ctx=%s, dtype=%s" %(ctx, dtype))
        generator_mx = lambda x: mx.nd.random.multinomial(data=mx.nd.array(np.array(probs), ctx=ctx, dtype=dtype),
                                                          shape=x).asnumpy()
        verify_generator(generator_mx, buckets, probs)
        generator_mx_same_seed = \
            lambda x: np.concatenate(
                [mx.nd.random.multinomial(data=mx.nd.array(np.array(probs), ctx=ctx, dtype=dtype),
                                                          shape=x // 10).asnumpy()
                 for _ in range(10)])
        verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)


if __name__ == '__main__':
    import nose
    nose.runmodule()
