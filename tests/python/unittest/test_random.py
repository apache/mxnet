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
import math
import itertools
import mxnet as mx
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf, assert_almost_equal
import numpy as np
import random as rnd
from common import retry, random_seed
import scipy.stats as ss
import unittest
import pytest
from mxnet.test_utils import *
from mxnet.base import MXNetError
from common import assertRaises

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
            'pdfsymbol': mx.sym.random_pdf_normal,
            'pdffunc': ss.norm.pdf,
            'discrete': False,
            'params': { 'loc': 10.0, 'scale': 0.5 },
            'inputs': [ ('loc',[ [ 0.0, 2.5 ], [ -9.75, -7.0 ] ]) , ('scale',[ [ 1.0, 3.7 ], [ 4.2, 1.5 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64) - params['loc']),  tol),
                ('std',  lambda x, params: np.std(x.astype(np.float64)) - params['scale'], tol)
            ]
        },
        {
            'name': 'normal_like',
            'symbol': mx.sym.random.normal_like,
            'ndop': mx.nd.random.normal_like,
            'params': { 'loc': 10.0, 'scale': 0.5 },
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64) - params['loc']),  tol),
                ('std',  lambda x, params: np.std(x.astype(np.float64)) - params['scale'], tol)
            ]
        },
        {
            'name': 'randn',
            'symbol': mx.sym.random.randn,
            'ndop': mx.nd.random.randn,
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
            'pdfsymbol': mx.sym.random_pdf_uniform,
            'pdffunc': lambda x, low, high: ss.uniform.pdf(x, low, high-low),
            'discrete': False,
            'params': { 'low': -1.5, 'high': 3.0 },
            'inputs': [ ('low', [ [ 0.0, 2.5 ], [ -9.75, -1.0 ] ]) , ('high', [ [ 1.0, 3.7 ], [ 4.2, 10.5 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - (params['low'] + params['high']) / 2.0, tol),
                ('std', lambda x,  params: np.std(x.astype(np.float64)) - np.sqrt(1.0 / 12.0) * (params['high'] - params['low']), tol)
            ]
        },
        {
            'name': 'uniform_like',
            'symbol': mx.sym.random.uniform_like,
            'ndop': mx.nd.random.uniform_like,
            'params': { 'low': -1.5, 'high': 3.0 },
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - (params['low'] + params['high']) / 2.0, tol),
                ('std', lambda x,  params: np.std(x.astype(np.float64)) - np.sqrt(1.0 / 12.0) * (params['high'] - params['low']), tol)
            ]
        },
        {
            'name': 'gamma',
            'symbol': mx.sym.random.gamma,
            'ndop': mx.nd.random.gamma,
            'pdfsymbol': mx.sym.random_pdf_gamma,
            'pdffunc': lambda x, alpha, beta: ss.gamma.pdf(x, alpha, 0, 1/beta),
            'discrete': False,
            'params': { 'alpha': 9.0, 'beta': 0.5 },
            'inputs': [ ('alpha', [ [ 0.1, 2.5 ], [ 9.75, 11.0 ] ]) , ('beta', [ [ 1.0, 0.7 ], [ 0.5, 0.3 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['alpha'] * params['beta'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['alpha'] * params['beta'] ** 2), tol)
            ]
        },
        {
            'name': 'gamma_like',
            'symbol': mx.sym.random.gamma_like,
            'ndop': mx.nd.random.gamma_like,
            'params': { 'alpha': 9.0, 'beta': 0.5 },
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['alpha'] * params['beta'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['alpha'] * params['beta'] ** 2), tol)
            ]
        },
        {
            'name': 'exponential',
            'symbol': mx.sym.random.exponential,
            'ndop': mx.nd.random.exponential,
            'pdfsymbol': mx.sym.random_pdf_exponential,
            'pdffunc': lambda x, lam: ss.expon.pdf(x, 0, 1/lam),
            'discrete': False,
            'params': { 'scale': 1.0/4.0 },
            'inputs': [ ('scale', [ [ 1.0/1.0, 1.0/8.5 ], [ 1.0/2.7 , 1.0/0.5 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['scale'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - params['scale'], tol)
            ]
        },
        {
            'name': 'exponential_like',
            'symbol': mx.sym.random.exponential_like,
            'ndop': mx.nd.random.exponential_like,
            'params': { 'lam': 4.0 },
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - 1.0/params['lam'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - 1.0/params['lam'], tol)
            ]
        },
        {
            'name': 'poisson',
            'symbol': mx.sym.random.poisson,
            'ndop': mx.nd.random.poisson,
            'pdfsymbol': mx.sym.random_pdf_poisson,
            'pdffunc': ss.poisson.pmf,
            'discrete': True,
            'params': { 'lam': 4.0 },
            'inputs': [ ('lam', [ [ 25.0, 8.5 ], [ 2.7 , 0.5 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['lam'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['lam']), tol)
            ]
        },
        {
            'name': 'poisson_like',
            'symbol': mx.sym.random.poisson_like,
            'ndop': mx.nd.random.poisson_like,
            'params': { 'lam': 4.0 },
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['lam'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['lam']), tol)
            ]
        },
        {
            'name': 'neg_binomial',
            'symbol': mx.sym.random.negative_binomial,
            'ndop': mx.nd.random.negative_binomial,
            'pdfsymbol': mx.sym.random_pdf_negative_binomial,
            'pdffunc': ss.nbinom.pmf,
            'discrete': True,
            'params': { 'k': 3, 'p': 0.4 },
            'inputs': [ ('k', [ [ 3, 4 ], [ 5 , 6 ] ]) , ('p', [ [ 0.4 , 0.77 ], [ 0.5, 0.84 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['k'] * (1.0 - params['p']) /  params['p'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['k'] * (1.0 - params['p']))/params['p'], tol)
            ]
        },
        {
            'name': 'neg_binomial_like',
            'symbol': mx.sym.random.negative_binomial_like,
            'ndop': mx.nd.random.negative_binomial_like,
            'params': { 'k': 3, 'p': 0.4 },
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['k'] * (1.0 - params['p']) /  params['p'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['k'] * (1.0 - params['p']))/params['p'], tol)
            ]
        },
        {
            'name': 'gen_neg_binomial',
            'symbol': mx.sym.random.generalized_negative_binomial,
            'ndop': mx.nd.random.generalized_negative_binomial,
            'pdfsymbol': mx.sym.random_pdf_generalized_negative_binomial,
            'pdffunc': lambda x, mu, alpha: ss.nbinom.pmf(x, 1.0/alpha, 1.0/(mu*alpha+1.0)),
            'discrete': True,
            'params': { 'mu': 2.0, 'alpha': 0.3 },
            'inputs': [ ('mu', [ [ 2.0, 2.5 ], [ 1.3, 1.9 ] ]) , ('alpha', [ [ 1.0, 0.1 ], [ 0.2, 0.5 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['mu'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['mu'] + params['alpha'] * params['mu'] ** 2 ), tol)
            ]
        },
        {
            'name': 'gen_neg_binomial_like',
            'symbol': mx.sym.random.generalized_negative_binomial_like,
            'ndop': mx.nd.random.generalized_negative_binomial_like,
            'params': { 'mu': 2.0, 'alpha': 0.3 },
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['mu'], tol),
                ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['mu'] + params['alpha'] * params['mu'] ** 2 ), tol)
            ]
        },

    ]

    # Create enough samples such that we get a meaningful distribution.
    shape = (500, 500)
    # Test pdf on smaller shapes as backward checks will take too long otherwise.
    # This must be a subshape of the former one.
    pdfshape = (30, 30)
    for symbdic in symbols:
        name = symbdic['name']
        ndop = symbdic['ndop']

        # check directly
        params = symbdic['params'].copy()
        params.update(shape=shape, dtype=dtype, ctx=device)
        args = ()
        if name == 'randn':
            params.pop('shape')  # randn does not accept shape param
            args = shape
        if name.endswith('_like'):
            params['data'] = mx.nd.ones(params.pop('shape'),
                                        dtype=params.pop('dtype'),
                                        ctx=params.pop('ctx'))
        mx.random.seed(128)
        ret1 = ndop(*args, **params).asnumpy()
        mx.random.seed(128)
        ret2 = ndop(*args, **params).asnumpy()
        assert same(ret1, ret2), \
                f"ndarray test: `{name}` should give the same result with the same seed"

        for check_name, check_func, tol in symbdic['checks']:
            assert np.abs(check_func(ret1, params)) < tol, f"ndarray test: {check_name} check for `{name}` did not pass"

        # check multi-distribution sampling
        if 'inputs' not in symbdic: continue  # randn does not support multi-distribution sampling

        params = {'shape': shape, 'dtype': dtype, 'ctx': device}
        params.update({k : mx.nd.array(v, ctx=device, dtype=dtype) for k, v in symbdic['inputs']})
        if name == 'randn':
            params.pop('shape')  # randn does not accept shape param
            args = shape
        mx.random.seed(128)
        ret1 = ndop(*args, **params).asnumpy()
        mx.random.seed(128)
        ret2 = ndop(*args, **params).asnumpy()
        assert same(ret1, ret2), \
                f"ndarray test: `{name}` should give the same result with the same seed"
        for i in range(2):
            for j in range(2):
                stats = {k : v[i][j] for k, v in symbdic['inputs']}
                for check_name, check_func, tol in symbdic['checks']:
                    err = np.abs(check_func(ret2[i,j], stats))
                    assert err < tol, f"{err} vs {tol}: symbolic test: {check_name} check for `{name}` did not pass"

        # check symbolic
        symbol = symbdic['symbol']
        X = mx.sym.Variable("X")
        params = symbdic['params'].copy()
        params.update(shape=shape, dtype=dtype)
        if name.endswith('_like') or name == 'randn':
            params['data'] = mx.sym.ones(params.pop('shape'))
        Y = symbol(**params) + X
        x = mx.nd.zeros(shape, dtype=dtype, ctx=device)
        xgrad = mx.nd.zeros(shape, dtype=dtype, ctx=device)
        yexec = Y._bind(device, {'X' : x}, {'X': xgrad})
        mx.random.seed(128)
        yexec.forward(is_train=True)
        yexec.backward(yexec.outputs[0])
        un1 = (yexec.outputs[0] - x).copyto(device)
        assert same(xgrad.asnumpy(), un1.asnumpy())
        mx.random.seed(128)
        yexec.forward()
        un2 = (yexec.outputs[0] - x).copyto(device)
        assert same(un1.asnumpy(), un2.asnumpy()), \
                f"symbolic test: `{name}` should give the same result with the same seed"

        ret1 = un1.asnumpy()
        for check_name, check_func, tol in symbdic['checks']:
            assert np.abs(check_func(ret1, params)) < tol, f"symbolic test: {check_name} check for `{name}` did not pass"
        if name.endswith('_like'): continue

        # check multi-distribution sampling
        symbol = symbdic['symbol']
        params = { 'shape' : shape, 'dtype' : dtype }
        single_param = len(symbdic['inputs']) == 1
        v1 = mx.sym.Variable('v1')
        v2 = mx.sym.Variable('v2')
        if name == 'randn':
            params.pop('shape')  # randn does not accept shape param
            args=shape
            Y = symbol(v1, **params) if single_param else symbol(*args, loc=v1, scale=v2,**params)
        else:
            Y = symbol(v1,**params) if single_param else symbol(v1,v2,**params)
        bindings = { 'v1' : mx.nd.array(symbdic['inputs'][0][1]) }
        if not single_param :
            bindings.update({ 'v2' : mx.nd.array(symbdic['inputs'][1][1]) })
        yexec = Y._bind(ctx=device, args=bindings)
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
                    assert np.abs(check_func(samples, params)) < tol, f"symbolic test: {check_name} check for `{name}` did not pass"

        if 'pdfsymbol' not in symbdic: continue  # randn not tested for pdf

        # check pdfs with only a subset of the generated samples
        un1 = np.resize(un1, (un1.shape[0], un1.shape[1], pdfshape[0], pdfshape[1]))
        symbol  = symbdic['pdfsymbol']
        pdffunc = symbdic['pdffunc']
        v0 = mx.sym.Variable('v0')
        v1 = mx.sym.Variable('v1')
        v2 = mx.sym.Variable('v2')
        p1 = np.array(symbdic['inputs'][0][1])
        p2 = None if single_param else np.array(symbdic['inputs'][1][1])
        # Move samples away from boundaries of support
        if name == 'gamma' or name == 'exponential':
           un1 = np.maximum(un1, 1e-1)
        if name == 'uniform':
           un1 = np.minimum(np.maximum(un1.reshape((un1.shape[0],un1.shape[1],-1)), p1.reshape((p1.shape[0],p1.shape[1],-1))+1e-4),
                            p2.reshape((p2.shape[0],p2.shape[1],-1))-1e-4).reshape(un1.shape)
        for use_log in [False, True]:
            test_pdf = symbol(v0, v1, is_log=use_log) if single_param else symbol(v0, v1, v2, is_log=use_log)
            forw_atol  = 1e-7 if dtype != np.float16 else 1e-3
            forw_rtol  = 1e-4 if dtype != np.float16 else 5e-2
            backw_atol = 1e-3
            backw_rtol = 5e-2
            if single_param:
                res = pdffunc(un1.reshape((un1.shape[0],un1.shape[1],-1)),
                    p1.reshape((p1.shape[0],p1.shape[1],-1))).reshape(un1.shape)
                if use_log:
                    res = np.log(res)
                check_symbolic_forward(test_pdf, [un1, p1], [res], atol=forw_atol, rtol=forw_rtol, dtype=dtype)
                if dtype == np.float64:
                  grad_nodes = ['v1'] if symbdic['discrete'] else ['v0', 'v1']
                  check_numeric_gradient(test_pdf, [un1, p1], grad_nodes=grad_nodes, atol=backw_atol, rtol=backw_rtol, dtype=dtype)
            else:
                res = pdffunc(un1.reshape((un1.shape[0],un1.shape[1],-1)),
                    p1.reshape((p1.shape[0],p1.shape[1],-1)),
                    p2.reshape((p2.shape[0],p2.shape[1],-1))).reshape(un1.shape)
                if use_log:
                    res = np.log(res)
                check_symbolic_forward(test_pdf, [un1, p1, p2], [res], atol=forw_atol, rtol=forw_rtol, dtype=dtype)
                if dtype == np.float64:
                  grad_nodes = ['v1', 'v2'] if symbdic['discrete'] else ['v0', 'v1', 'v2']
                  check_numeric_gradient(test_pdf, [un1, p1, p2], grad_nodes=grad_nodes, atol=backw_atol, rtol=backw_rtol, dtype=dtype)

@pytest.mark.seed(1000)
@pytest.mark.serial
def test_dirichlet():
    num_classes = 2
    num = 100
    alpha = np.random.uniform(low=0.5, high=2, size=(4, num_classes))

    samples = []
    results = []
    for a in alpha:
        v = ss.dirichlet.rvs(a, size=num)
        samples.append(v)
        results.append(ss.dirichlet.logpdf(v.transpose(), a))
    samples = np.concatenate(samples, axis=0).reshape((2, 2, num, num_classes))
    results = np.concatenate(results, axis=0).reshape((2, 2, num))

    alpha = alpha.reshape((2, 2, num_classes))

    for dtype in [np.float32, np.float64]:
        forw_atol  = 1e-5
        forw_rtol  = 1e-4
        for use_log in [False, True]:
            v0 = mx.sym.Variable('v0')
            v1 = mx.sym.Variable('v1')
            test_pdf = mx.sym.random_pdf_dirichlet(v0, v1, is_log=use_log)
            res = results if use_log else np.exp(results)
            check_symbolic_forward(test_pdf, [samples, alpha], [res], atol=forw_atol, rtol=forw_rtol, dtype=dtype)
            if dtype == np.float64:
                backw_atol = 1e-2
                backw_rtol = 1e-2
                eps = 1e-5
                check_numeric_gradient(test_pdf, [samples, alpha], numeric_eps=eps, atol=backw_atol, rtol=backw_rtol, dtype=dtype)

@pytest.mark.serial
def test_random():
    for dtype in [np.float16, np.float32, np.float64]:
        check_with_device(mx.context.current_context(), dtype)

# Set seed variously based on `start_seed` and `num_init_seeds`, then set seed finally to `final_seed`
def set_seed_variously(init_seed, num_init_seeds, final_seed):
    end_seed = init_seed + num_init_seeds
    for seed in range(init_seed, end_seed):
        mx.random.seed(seed)
    mx.random.seed(final_seed)
    return end_seed

# Tests that seed setting of std (non-parallel) rng is synchronous w.r.t. rng use before and after.
@pytest.mark.serial
def test_random_seed_setting():
    ctx = mx.context.current_context()
    seed_to_test = 1234
    num_temp_seeds = 25
    probs = [0.125, 0.25, 0.25, 0.0625, 0.125, 0.1875]
    num_samples = 100000
    for dtype in ['float16', 'float32', 'float64']:
        seed = set_seed_variously(1, num_temp_seeds, seed_to_test)
        samples1 = mx.nd.random.categorical(data=mx.nd.array(probs, ctx=ctx, dtype=dtype),
                                            shape=num_samples)
        seed = set_seed_variously(seed, num_temp_seeds, seed_to_test)
        samples2 = mx.nd.random.categorical(data=mx.nd.array(probs, ctx=ctx, dtype=dtype),
                                            shape=num_samples)
        samples1np = samples1.asnumpy()
        set_seed_variously(seed, num_temp_seeds, seed_to_test+1)
        samples2np = samples2.asnumpy()
        assert same(samples1np, samples2np), \
            "seed-setting test: `categorical` should give the same result with the same seed"


# Tests that seed setting of parallel rng is synchronous w.r.t. rng use before and after.
@pytest.mark.serial
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
            yexec = Y._bind(ctx, {'X' : x}, {'X': xgrad})
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

# Set seed for the context variously based on `start_seed` and `num_init_seeds`, then set seed finally to `final_seed`
def set_seed_variously_for_context(ctx, init_seed, num_init_seeds, final_seed):
    end_seed = init_seed + num_init_seeds
    for seed in range(init_seed, end_seed):
        mx.random.seed(seed, ctx=ctx)
    mx.random.seed(final_seed, ctx=ctx)
    return end_seed

# Tests that seed setting of std (non-parallel) rng for specific context is synchronous w.r.t. rng use before and after.
@pytest.mark.serial
def test_random_seed_setting_for_context():
    seed_to_test = 1234
    num_temp_seeds = 25
    probs = [0.125, 0.25, 0.25, 0.0625, 0.125, 0.1875]
    num_samples = 100000
    dev_type = mx.context.current_context().device_type
    for dtype in ['float16', 'float32', 'float64']:
        samples_imp = []
        samples_sym = []
        # Collect random number samples from the generators of all devices, each seeded with the same number.
        for dev_id in range(0, mx.device.num_gpus() if dev_type == 'gpu' else 1):
            with mx.Context(dev_type, dev_id):
                ctx = mx.context.current_context()
                seed = set_seed_variously_for_context(ctx, 1, num_temp_seeds, seed_to_test)

                # Check imperative. `categorical` uses non-parallel rng.
                rnds = mx.nd.random.categorical(data=mx.nd.array(probs, dtype=dtype), shape=num_samples)
                samples_imp.append(rnds.asnumpy())

                # Check symbolic. `categorical` uses non-parallel rng.
                P = mx.sym.Variable("P")
                X = mx.sym.random.categorical(data=P, shape=num_samples, get_prob=False)
                exe = X._bind(ctx, {"P": mx.nd.array(probs, dtype=dtype)})
                set_seed_variously_for_context(ctx, seed, num_temp_seeds, seed_to_test)
                exe.forward()
                samples_sym.append(exe.outputs[0].asnumpy())
        # The samples should be identical across different gpu devices.
        for i in range(1, len(samples_imp)):
            assert same(samples_imp[i - 1], samples_imp[i])
        for i in range(1, len(samples_sym)):
            assert same(samples_sym[i - 1], samples_sym[i])

# Tests that seed setting of parallel rng for specific context is synchronous w.r.t. rng use before and after.
@pytest.mark.serial
def test_parallel_random_seed_setting_for_context():
    seed_to_test = 1234
    dev_type = mx.context.current_context().device_type
    for dtype in ['float16', 'float32', 'float64']:
        samples_imp = []
        samples_sym = []
        # Collect random number samples from the generators of all devices, each seeded with the same number.
        for dev_id in range(0, mx.device.num_gpus() if dev_type == 'gpu' else 1):
            with mx.Context(dev_type, dev_id):
                ctx = mx.context.current_context()
                # Avoid excessive test cpu runtimes.
                num_temp_seeds = 25 if dev_type == 'gpu' else 1
                # To flush out a possible race condition, run multiple times.
                for _ in range(20):
                    # Create enough samples such that we get a meaningful distribution.
                    shape = (200, 200)
                    params = { 'low': -1.5, 'high': 3.0 }
                    params.update(shape=shape, dtype=dtype)

                    # Check imperative. `uniform` uses parallel rng.
                    seed = set_seed_variously_for_context(ctx, 1, num_temp_seeds, seed_to_test)
                    rnds = mx.nd.random.uniform(**params)
                    samples_imp.append(rnds.asnumpy())

                    # Check symbolic. `uniform` uses parallel rng.
                    X = mx.sym.Variable("X")
                    Y = mx.sym.random.uniform(**params) + X
                    x = mx.nd.zeros(shape, dtype=dtype)
                    xgrad = mx.nd.zeros(shape, dtype=dtype)
                    yexec = Y._bind(ctx, {'X' : x}, {'X': xgrad})
                    set_seed_variously_for_context(ctx, seed, num_temp_seeds, seed_to_test)
                    yexec.forward(is_train=True)
                    yexec.backward(yexec.outputs[0])
                    samples_sym.append(yexec.outputs[0].asnumpy())
        # The samples should be identical across different gpu devices.
        for i in range(1, len(samples_imp)):
            assert same(samples_imp[i - 1], samples_imp[i])
        for i in range(1, len(samples_sym)):
            assert same(samples_sym[i - 1], samples_sym[i])

@pytest.mark.parametrize('dtype', ['uint8', 'int32', 'float16', 'float32', 'float64'])
@pytest.mark.parametrize('x', [[[0,1,2,3,4],[4,3,2,1,0]], [0,1,2,3,4]])
@pytest.mark.serial
def test_sample_categorical(dtype, x):
    x = mx.nd.array(x) / 10.0
    dx = mx.nd.ones_like(x)
    mx.autograd.mark_variables([x], [dx])
    # Adding rtol and increasing samples needed to pass with seed 2951820647
    samples = 10000
    with mx.autograd.record():
        y, prob = mx.nd.random.categorical(x, shape=samples, get_prob=True, dtype=dtype)
        r = prob * 5
        r.backward()

    assert(np.dtype(dtype) == y.dtype)
    y = y.asnumpy()
    x = x.asnumpy()
    dx = dx.asnumpy()
    if len(x.shape) is 1:
        x = x.reshape((1, x.shape[0]))
        dx = dx.reshape(1, dx.shape[0])
        y = y.reshape((1, y.shape[0]))
        prob = prob.reshape((1, prob.shape[0]))
    for i in range(x.shape[0]):
        freq = np.bincount(y[i,:].astype('int32'), minlength=5)/np.float32(samples)*x[i,:].sum()
        assert_almost_equal(freq, x[i], rtol=0.20, atol=1e-1)
        rprob = x[i][y[i].astype('int32')]/x[i].sum()
        assert_almost_equal(np.log(rprob), prob.asnumpy()[i], atol=1e-5)

        real_dx = np.zeros((5,))
        for j in range(samples):
            real_dx[int(y[i][j])] += 5.0 / rprob[j]
        assert_almost_equal(real_dx, dx[i, :], rtol=1e-4, atol=1e-5)

# Test the generators with the chi-square testing
@pytest.mark.serial
def test_normal_generator():
    ctx = mx.context.current_context()
    samples = 1000000
    # Default success rate is 0.25, so 2 successes of 8 trials will pass.
    trials = 8
    num_buckets = 5
    for dtype in ['float16', 'float32', 'float64']:
        for mu, sigma in [(0.0, 1.0), (1.0, 5.0)]:
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.norm.ppf(x, mu, sigma), num_buckets)
            # Quantize bucket boundaries to reflect the actual dtype and adjust probs accordingly
            buckets = np.array(buckets, dtype=dtype).tolist()
            probs = [(ss.norm.cdf(buckets[i][1], mu, sigma) -
                      ss.norm.cdf(buckets[i][0], mu, sigma)) for i in range(num_buckets)]
            generator_mx = lambda x: mx.nd.random.normal(mu, sigma, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs,
                             nsamples=samples, nrepeat=trials)
            generator_mx_same_seed =\
                lambda x: np.concatenate(
                    [mx.nd.random.normal(mu, sigma, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                     for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs,
                             nsamples=samples, nrepeat=trials)

@pytest.mark.serial
def test_uniform_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        for low, high in [(-1.0, 1.0), (1.0, 3.0)]:
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

@pytest.mark.serial
def test_gamma_generator():
    success_rate = 0.05
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        for kappa, theta in [(0.5, 1.0), (1.0, 5.0)]:
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.gamma.ppf(x, a=kappa, loc=0, scale=theta), 5)
            generator_mx = lambda x: mx.nd.random.gamma(kappa, theta, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs, success_rate=success_rate)
            generator_mx_same_seed = \
                lambda x: np.concatenate(
                    [mx.nd.random.gamma(kappa, theta, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                     for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs, success_rate=success_rate)

@pytest.mark.serial
def test_exponential_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        for scale in [0.1, 1.0]:
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.expon.ppf(x, loc=0, scale=scale), 5)
            generator_mx = lambda x: mx.nd.random.exponential(scale, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs, success_rate=0.20)
            generator_mx_same_seed = \
                lambda x: np.concatenate(
                    [mx.nd.random.exponential(scale, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                     for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs, success_rate=0.20)

@pytest.mark.serial
def test_poisson_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        for lam in [1, 10]:
            buckets = [(-1.0, lam - 0.5), (lam - 0.5, 2 * lam + 0.5), (2 * lam + 0.5, np.inf)]
            probs = [ss.poisson.cdf(bucket[1], lam) - ss.poisson.cdf(bucket[0], lam) for bucket in buckets]
            generator_mx = lambda x: mx.nd.random.poisson(lam, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs)
            generator_mx_same_seed = \
                lambda x: np.concatenate(
                    [mx.nd.random.poisson(lam, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                     for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)

@pytest.mark.serial
def test_binomial_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        trials_num = 10000
        success_prob = 0.25

        buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.binom.ppf(x, trials_num, success_prob), 10)
        generator_mx = lambda x: mx.nd.random.binomial(trials_num, success_prob,
                                                                shape=x, ctx=ctx, dtype=dtype).asnumpy()
        nsamples = 1000
        verify_generator(generator=generator_mx, buckets=buckets, probs=probs, nsamples=nsamples)
        generator_mx_same_seed = \
            lambda x: np.concatenate(
                [mx.nd.random.binomial(trials_num, success_prob, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                 for _ in range(10)])
        verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs, nsamples=nsamples)

@pytest.mark.serial
def test_negative_binomial_generator():
    ctx = mx.context.current_context()
    for dtype in ['float16', 'float32', 'float64']:
        success_num = 2
        success_prob = 0.2
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

@pytest.mark.serial
def test_categorical_generator():
    # This test fails with dtype float16 if the probabilities themselves cannot be
    # well-represented in float16.  When the float16 random picks are assigned to buckets,
    # only certain bucket-probabilities are possible.  Here we map the desired probabilites
    # (e.g. 0.1) to nearby float16 probabilities (e.g. 0.10009766) that are achievable.
    def quantize_probs(probs, dtype):
        if dtype == 'float16':
            # float16 has a 10-bit fraction plus an implicit leading 1, so all probabilities
            # of the form N/2^11 (where N is an integer) are representable.
            num_quanta = 2048.0
            quantized_probs = np.rint(np.array(probs) * num_quanta) / num_quanta
            # Ensure probabilities add to 1
            quantized_probs[0] += 1.0 - quantized_probs.sum()
        else:
            # no need to quantize probs with this data precision
            quantized_probs = np.array(probs)
        return quantized_probs

    ctx = mx.context.current_context()
    probs = [0.1, 0.2, 0.3, 0.05, 0.15, 0.2]
    samples = 1000000
    trials = 5
    buckets = list(range(6))
    for dtype in ['float16', 'float32', 'float64']:
        quantized_probs = quantize_probs(probs, dtype)
        generator_mx = lambda x: mx.nd.random.categorical(data=mx.nd.array(quantized_probs, ctx=ctx, dtype=dtype),
                                                          shape=x).asnumpy()
        # success_rate was set to 0.15 since PR #13498 and became flaky
        # both of previous issues(#14457, #14158) failed with success_rate 0.25
        # In func verify_generator inside test_utilis.py
        # it raise the error when success_num(1) < nrepeat(5) * success_rate(0.25)
        # by changing the 0.25 -> 0.2 solve these edge case but still have strictness
        verify_generator(generator=generator_mx, buckets=buckets, probs=quantized_probs,
                         nsamples=samples, nrepeat=trials, success_rate=0.20)
        generator_mx_same_seed = \
            lambda x: np.concatenate(
                [mx.nd.random.categorical(data=mx.nd.array(quantized_probs, ctx=ctx, dtype=dtype),
                                                          shape=x // 10).asnumpy()
                 for _ in range(10)])
        verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=quantized_probs,
                         nsamples=samples, nrepeat=trials, success_rate=0.20)


@pytest.mark.serial
def test_multinomial_generator():
    def repeat_i(arr):
        """
        Return an array containing ordered values from 0 to arr.size()-1,
        where each value i is repeated arr[i] times.

        Example:
        >>> repeat_i([3, 1, 2, 1])
        [0, 0, 0, 1, 2, 2, 3]
        """
        ind = mx.nd.expand_dims(mx.nd.cumsum(mx.nd.concat(mx.nd.array([0]), arr[:arr.size-1], dim=0)), axis=0)
        data = mx.nd.ones((arr.size,))
        shape = (int(mx.nd.sum(arr).asscalar()),)
        return mx.nd.cumsum(mx.nd.scatter_nd(data, ind, shape)) - 1

    ctx = mx.context.current_context()
    probs = np.array([0.1, 0.2, 0.3, 0.05, 0.15, 0.2])

    buckets = list(range(6))
    for dtype in ['float16', 'float32', 'float64']:
        generator_mx = lambda x: repeat_i(mx.nd.random.multinomial(n=mx.nd.array([x]), p=mx.nd.array([probs]), ctx=ctx)[0]).asnumpy()
        verify_generator(generator=generator_mx, buckets=buckets, probs=probs)

        generator_mx_same_seed = \
            lambda x: np.concatenate([generator_mx(x // 10) for _ in range(10)])
        verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs)


@pytest.mark.serial
def test_with_random_seed():
    ctx = mx.context.current_context()
    size = 100
    shape = (size,)

    def check_same(x, y, name):
        assert same(x, y), \
            f"{name} rng should give the same result with the same seed"

    def check_diff(x, y, name):
        assert not same(x, y), \
            f"{name} rng should give different results with different seeds"

    # generate python, numpy and mxnet datasets with the given seed
    def gen_data(seed=None):
        with random_seed(seed):
            python_data = [rnd.random() for _ in range(size)]
            np_data = np.random.rand(size)
            mx_data = mx.random.uniform(shape=shape, ctx=ctx).asnumpy()
        return (seed, python_data, np_data, mx_data)

    # check data, expecting them to be the same or different based on the seeds
    def check_data(a, b):
        seed_a = a[0]
        seed_b = b[0]
        if seed_a == seed_b and seed_a is not None:
            check_same(a[1], b[1], 'python')
            check_same(a[2], b[2], 'numpy')
            check_same(a[3], b[3], 'mxnet')
        else:
            check_diff(a[1], b[1], 'python')
            check_diff(a[2], b[2], 'numpy')
            check_diff(a[3], b[3], 'mxnet')

    # 5 tests that include a duplicated seed 1 and randomizing seed None
    seeds = [1, 2, 1, None, None]
    data = [gen_data(seed) for seed in seeds]

    # Add more complicated test case scenarios
    with random_seed(1):
        seeds.append(None)
        data.append(gen_data(None))
    with random_seed(2):
        seeds.append(None)
        data.append(gen_data(None))
    with random_seed():
        seeds.append(1)
        data.append(gen_data(1))
    with random_seed():
        seeds.append(2)
        data.append(gen_data(2))
    with random_seed(1):
        seeds.append(2)
        data.append(gen_data(2))

    num_seeds = len(seeds)
    for i in range(0, num_seeds-1):
        for j in range(i+1, num_seeds):
            check_data(data[i],data[j])

@pytest.mark.serial
def test_random_seed():
    shape = (5, 5)
    seed = rnd.randint(-(1 << 31), (1 << 31))

    def _assert_same_mx_arrays(a, b):
        assert len(a) == len(b)
        for a_i, b_i in zip(a, b):
            assert (a_i.asnumpy() == b_i.asnumpy()).all()

    N = 100
    mx.random.seed(seed)
    v1 = [mx.random.uniform(shape=shape) for _ in range(N)]

    mx.random.seed(seed)
    v2 = [mx.random.uniform(shape=shape) for _ in range(N)]
    _assert_same_mx_arrays(v1, v2)

    try:
        long
        mx.random.seed(long(seed))
        v3 = [mx.random.uniform(shape=shape) for _ in range(N)]
        _assert_same_mx_arrays(v1, v3)
    except NameError:
        pass

@pytest.mark.serial
def test_unique_zipfian_generator():
    ctx = mx.context.current_context()
    if ctx.device_type == 'cpu':
        num_sampled = 8192
        range_max = 793472
        batch_size = 4
        op = mx.nd._internal._sample_unique_zipfian
        classes, num_trials = op(range_max, shape=(batch_size, num_sampled))
        for i in range(batch_size):
            num_trial = num_trials[i].asscalar()
            # test uniqueness
            assert np.unique(classes[i].asnumpy()).size == num_sampled
            # test num trials. reference count obtained from pytorch implementation
            assert num_trial > 14500
            assert num_trial < 17000

@pytest.mark.serial
def test_zipfian_generator():
    # dummy true classes
    num_true = 5
    num_sampled = 1000
    range_max = 20

    def compute_expected_prob():
        # P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)
        classes = mx.nd.arange(0, range_max)
        expected_counts = ((classes + 2).log() - (classes + 1).log()) / np.log(range_max + 1)
        return expected_counts

    exp_cnt = compute_expected_prob() * num_sampled

    # test ndarray
    true_classes = mx.nd.random.uniform(0, range_max, shape=(num_true,)).astype('int32')
    sampled_classes, exp_cnt_true, exp_cnt_sampled = mx.nd.contrib.rand_zipfian(true_classes, num_sampled, range_max)
    assert_almost_equal(exp_cnt_sampled, exp_cnt[sampled_classes], rtol=1e-1, atol=1e-2)
    assert_almost_equal(exp_cnt_true, exp_cnt[true_classes], rtol=1e-1, atol=1e-2)

    # test symbol
    true_classes_var = mx.sym.var('true_classes')
    outputs = mx.sym.contrib.rand_zipfian(true_classes_var, num_sampled, range_max)
    outputs = mx.sym.Group(outputs)
    executor = outputs._bind(mx.context.current_context(), {'true_classes' : true_classes})
    executor.forward()
    sampled_classes, exp_cnt_true, exp_cnt_sampled = executor.outputs
    assert_almost_equal(exp_cnt_sampled, exp_cnt[sampled_classes], rtol=1e-1, atol=1e-2)
    assert_almost_equal(exp_cnt_true, exp_cnt[true_classes], rtol=1e-1, atol=1e-2)

# Issue #10277 (https://github.com/apache/mxnet/issues/10277) discusses this test.
@pytest.mark.serial
def test_shuffle():
    def check_first_axis_shuffle(arr):
        stride = int(arr.size / arr.shape[0])
        column0 = arr.reshape((arr.size,))[::stride]
        seq = mx.nd.arange(0, arr.size - stride + 1, stride, ctx=arr.context)
        assert (column0.sort() == seq).prod() == 1
        # Check for ascending flattened-row sequences for 2D or greater inputs.
        if stride > 1:
            ascending_seq = mx.nd.arange(0, stride, ctx=arr.context)
            equalized_columns = arr.reshape((arr.shape[0], stride)) - ascending_seq
            column0_2d = column0.reshape((arr.shape[0],1))
            assert (column0_2d == equalized_columns).prod() == 1

    # This tests that the shuffling is along the first axis with `repeat1` number of shufflings
    # and the outcomes are uniformly distributed with `repeat2` number of shufflings.
    # Note that the enough number of samples (`repeat2`) to verify the uniformity of the distribution
    # of the outcomes grows factorially with the length of the first axis of the array `data`.
    # So we have to settle down with small arrays in practice.
    # `data` must be a consecutive sequence of integers starting from 0 if it is flattened.
    def testSmall(data, repeat1, repeat2):
        # Check that the shuffling is along the first axis.
        # The order of the elements in each subarray must not change.
        # This takes long time so `repeat1` need to be small.
        for _ in range(repeat1):
            ret = mx.nd.random.shuffle(data)
            check_first_axis_shuffle(ret)
        # Count the number of each different outcome.
        # The sequence composed of the first elements of the subarrays is enough to discriminate
        # the outcomes as long as the order of the elements in each subarray does not change.
        count = {}
        stride = int(data.size / data.shape[0])
        for _ in range(repeat2):
            ret = mx.nd.random.shuffle(data)
            h = str(ret.reshape((ret.size,))[::stride])
            c = count.get(h, 0)
            count[h] = c + 1
        # Check the total number of possible outcomes.
        # If `repeat2` is not large enough, this could fail with high probability.
        assert len(count) == math.factorial(data.shape[0])
        # The outcomes must be uniformly distributed.
        # If `repeat2` is not large enough, this could fail with high probability.
        for p in itertools.permutations(range(0, data.size - stride + 1, stride)):
            err = abs(1. * count[str(mx.nd.array(p))] / repeat2 - 1. / math.factorial(data.shape[0]))
            assert err < 0.01, "The absolute error {} is larger than the tolerance.".format(err)
        # Check symbol interface
        a = mx.sym.Variable('a')
        b = mx.sym.random.shuffle(a)
        c = mx.sym.random.shuffle(data=b, name='c')
        d = mx.sym.sort(c, axis=0)
        assert (d.eval(a=data, ctx=mx.current_context())[0] == data).prod() == 1

    # This test is weaker than `testSmall` and to test larger arrays.
    # `repeat` should be much smaller than the factorial of `len(x.shape[0])`.
    # `data` must be a consecutive sequence of integers starting from 0 if it is flattened.
    def testLarge(data, repeat):
        # Check that the shuffling is along the first axis
        # and count the number of different outcomes.
        stride = int(data.size / data.shape[0])
        count = {}
        for _ in range(repeat):
            ret = mx.nd.random.shuffle(data)
            check_first_axis_shuffle(ret)
            h = str(ret.reshape((ret.size,))[::stride])
            c = count.get(h, 0)
            count[h] = c + 1
        # The probability of duplicated outcomes is very low for large arrays.
        assert len(count) == repeat

    # Test small arrays with different shapes
    testSmall(mx.nd.arange(0, 3), 100, 40000)
    testSmall(mx.nd.arange(0, 9).reshape((3, 3)), 100, 40000)
    testSmall(mx.nd.arange(0, 18).reshape((3, 2, 3)), 100, 40000)
    # Test larger arrays
    testLarge(mx.nd.arange(0, 100000).reshape((10, 10000)), 10)
    testLarge(mx.nd.arange(0, 100000).reshape((10000, 10)), 10)
    testLarge(mx.nd.arange(0, 100000), 10)


@pytest.mark.serial
def test_randint():
    dtypes = ['int32', 'int64']
    for dtype in dtypes:
        params = {
            'low': -1,
            'high': 3,
            'shape' : (500, 500),
            'dtype' : dtype,
            'ctx' : mx.context.current_context()
            }
        mx.random.seed(128)
        ret1 = mx.nd.random.randint(**params).asnumpy()
        mx.random.seed(128)
        ret2 = mx.nd.random.randint(**params).asnumpy()
        assert same(ret1, ret2), \
                "ndarray test: `%s` should give the same result with the same seed"

@pytest.mark.serial
def test_randint_extremes():
    a = mx.nd.random.randint(dtype='int64', low=50000000, high=50000010, ctx=mx.context.current_context())
    assert a>=50000000 and a<=50000010

@pytest.mark.serial
def test_randint_generator():
    ctx = mx.context.current_context()
    for dtype in ['int32', 'int64']:
        for low, high in [(50000000, 50001000),(-50000100,-50000000),(-500,199)]:
            scale = high - low
            buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.uniform.ppf(x, loc=low, scale=scale), 5)
            # Quantize bucket boundaries to reflect the actual dtype and adjust probs accordingly
            buckets = np.array(buckets, dtype=dtype).tolist()
            probs = [(buckets[i][1] - buckets[i][0]) / float(scale) for i in range(5)]
            generator_mx = lambda x: mx.nd.random.randint(low, high, shape=x, ctx=ctx, dtype=dtype).asnumpy()
            verify_generator(generator=generator_mx, buckets=buckets, probs=probs, nrepeat=100)
            # Scipy uses alpha = 0.01 for testing discrete distribution generator but we are using default alpha=0.05 (higher threshold ensures robustness)
            # Refer - https://github.com/scipy/scipy/blob/9f12af697763fb5f9767d5cb1280ce62456a3974/scipy/stats/tests/test_discrete_basic.py#L45
            generator_mx_same_seed = \
                lambda x: np.concatenate(
                    [mx.nd.random.randint(low, high, shape=x // 10, ctx=ctx, dtype=dtype).asnumpy()
                        for _ in range(10)])
            verify_generator(generator=generator_mx_same_seed, buckets=buckets, probs=probs, nrepeat=100)

@pytest.mark.serial
def test_randint_without_dtype():
    a = mx.nd.random.randint(low=50000000, high=50000010, ctx=mx.context.current_context())
    assert a.dtype == np.int32


@pytest.mark.serial
def test_sample_categorical_num_outputs():
    ctx = mx.context.current_context()
    probs = [[0.125, 0.25, 0.25], [0.0625, 0.125, 0.1875]]
    out = mx.nd.random.categorical(data=mx.nd.array(probs, ctx=ctx), shape=10000, get_prob=False)
    assert isinstance(out, mx.nd.NDArray)
    out = mx.nd.random.categorical(data=mx.nd.array(probs, ctx=ctx), shape=10000, get_prob=True)
    assert isinstance(out, list)
    assert len(out) == 2


@use_np
def test_dirichlet_zero_size_dim():
    """ Tests for no error when dealing with zero-size array in calculating PDF of Poisson distribution
    Issue: https://github.com/apache/mxnet/issues/18936
    """

    def test_valid_zero_dim():
        alpha = mx.nd.array(np.random.rand(0))
        sample = mx.nd.array(np.random.rand(4, 0))
        res = mx.nd.op.random_pdf_dirichlet(sample=sample, alpha=alpha)
        assert res.shape == sample.shape[:-1]

    def test_valid_zero_multi_dim():
        alpha = mx.nd.array(np.random.rand(4, 0))
        sample = mx.nd.array(np.random.rand(4, 3, 0))
        res = mx.nd.op.random_pdf_dirichlet(sample=sample, alpha=alpha)
        assert res.shape == sample.shape[:-1]

    def test_invalid_zero_dim():
        """The shape of *alpha* must match the left-most part of the *sample* shape"""
        alpha = mx.nd.array(np.random.rand(1))
        sample = mx.nd.array(np.random.rand(4, 0))
        assertRaises(MXNetError, mx.nd.op.random_pdf_dirichlet, sample, alpha)
        
    test_valid_zero_dim()
    test_valid_zero_multi_dim()
    test_invalid_zero_dim()

@use_np
def test_poisson_zero_size_dim():
    """ Tests for no error when dealing with zero-size array in calculating PDF of Poisson distribution
    Issue: https://github.com/apache/mxnet/issues/18937
    """

    def test_valid_zero_dim():
        lam = mx.nd.array(np.random.rand(0))
        sample = mx.nd.array(np.random.rand(0, 2))
        res = mx.nd.op.random_pdf_poisson(sample=sample, lam=lam)
        assert res.shape == sample.shape

    def test_invalid_zero_dim():
        """The shape of *lam* must match the leftmost part of the *sample* shape"""
        lam = mx.nd.array(np.random.rand(0))
        sample = mx.nd.array(np.random.rand(1, 2))
        assertRaises(MXNetError, mx.nd.op.random_pdf_poisson, sample, lam)

    test_valid_zero_dim()
    test_invalid_zero_dim()
