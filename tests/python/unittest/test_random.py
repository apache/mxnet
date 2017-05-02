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
            'symbol': mx.sym.random_normal,
            'multisymbol': mx.sym.sample_normal,
            'ndop': mx.random.normal,
            'params': { 'loc': 10.0, 'scale': 0.5 },
            'inputs': [ ('loc',[ [ 0.0, 2.5 ], [ -9.75, -7.0 ] ]) , ('scale',[ [ 1.0, 3.7 ], [ 4.2, 1.5 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64) - params['loc']),  tol),
                ('std',  lambda x, params: np.std(x.astype(np.float64)) - params['scale'], tol)
            ]
        },
        {
            'name': 'uniform',
            'symbol': mx.sym.random_uniform,
            'multisymbol': mx.sym.sample_uniform,
            'ndop': mx.random.uniform,
            'params': { 'low': -1.5, 'high': 3.0 },
            'inputs': [ ('low', [ [ 0.0, 2.5 ], [ -9.75, -1.0 ] ]) , ('high', [ [ 1.0, 3.7 ], [ 4.2, 10.5 ] ]) ],
            'checks': [
                ('mean', lambda x, params: np.mean(x.astype(np.float64)) - (params['low'] + params['high']) / 2.0, tol),
                ('std', lambda x,  params: np.std(x.astype(np.float64)) - np.sqrt(1.0 / 12.0) * (params['high'] - params['low']), tol)
            ]
        }
    ]
    if device.device_type == 'cpu':
        symbols.extend([
            {
                'name': 'gamma',
                'symbol': mx.sym.random_gamma,
                'multisymbol': mx.sym.sample_gamma,
                'ndop': mx.random.gamma,
                'params': { 'alpha': 9.0, 'beta': 0.5 },
                'inputs': [ ('alpha', [ [ 0.0, 2.5 ], [ 9.75, 11.0 ] ]) , ('beta', [ [ 1.0, 0.7 ], [ 0.5, 0.3 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['alpha'] * params['beta'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['alpha'] * params['beta'] ** 2), tol)
                ]
            },
            {
                'name': 'exponential',
                'symbol': mx.sym.random_exponential,
                'multisymbol': mx.sym.sample_exponential,
                'ndop': mx.random.exponential,
                'params': { 'lam': 4.0 },
                'inputs': [ ('lam', [ [ 1.0, 8.5 ], [ 2.7 , 0.5 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - 1.0 / params['lam'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - 1.0 / params['lam'], tol)
                ]
            },
            {
                'name': 'poisson',
                'symbol': mx.sym.random_poisson,
                'ndop': mx.random.poisson,
                'multisymbol': mx.sym.sample_poisson,
                'params': { 'lam': 4.0 },
                'inputs': [ ('lam', [ [ 1.0, 8.5 ], [ 2.7 , 0.5 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['lam'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['lam']), tol)
                ]
            },
            {
                'name': 'neg-binomial',
                'symbol': mx.sym.random_negative_binomial,
                'multisymbol': mx.sym.sample_negative_binomial,
                'ndop': mx.random.negative_binomial,
                'params': { 'k': 3, 'p': 0.4 },
                'inputs': [ ('k', [ [ 20, 49 ], [ 15 , 16 ] ]) , ('p', [ [ 0.4 , 0.77 ], [ 0.5, 0.84 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['k'] * (1.0 - params['p']) /  params['p'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['k'] * (1.0 - params['p']))/params['p'], tol)
                ]
            },
            {
                'name': 'gen-neg-binomial',
                'symbol': mx.sym.random_generalized_negative_binomial,
                'multisymbol': mx.sym.sample_generalized_negative_binomial,
                'ndop': mx.random.generalized_negative_binomial,
                'params': { 'mu': 2.0, 'alpha': 0.3 },
                'inputs': [ ('mu', [ [ 2.0, 2.5 ], [ 1.3, 1.9 ] ]) , ('alpha', [ [ 1.0, 0.1 ], [ 0.2, 0.5 ] ]) ],
                'checks': [
                    ('mean', lambda x, params: np.mean(x.astype(np.float64)) - params['mu'], tol),
                    ('std', lambda x, params: np.std(x.astype(np.float64)) - np.sqrt(params['mu'] + params['alpha'] * params['mu'] ** 2 ), tol)
                ]
            }

        ])

    shape = (100, 100)
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

        # check multi-distribution sampling, only supports cpu for now
        if device.device_type == 'cpu':
            symbol = symbdic['multisymbol']
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



if __name__ == '__main__':
    test_random()
