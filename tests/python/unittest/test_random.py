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
import numpy as np

def same(a, b):
    return np.sum(a != b) == 0

def check_with_device(device, dtype):
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
        assert device.device_type == 'gpu' or same(ret1, ret2), \
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
        assert device.device_type == 'gpu' or same(ret1, ret2), \
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
        assert device.device_type == 'gpu' or same(un1.asnumpy(), un2.asnumpy()), \
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


if __name__ == '__main__':
    import nose
    nose.runmodule()
