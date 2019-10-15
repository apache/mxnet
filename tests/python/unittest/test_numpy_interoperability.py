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
import numpy as _np
from mxnet import np
from mxnet.test_utils import assert_almost_equal
from mxnet.test_utils import use_np
from common import assertRaises, with_seed
from mxnet.numpy_dispatch_protocol import with_array_function_protocol, with_array_ufunc_protocol
from mxnet.numpy_dispatch_protocol import _NUMPY_ARRAY_FUNCTION_LIST, _NUMPY_ARRAY_UFUNC_LIST

import itertools

class OpArgMngr(object):
    """Operator argument manager for storing operator workloads."""
    _args = {}

    @staticmethod
    def add_workload(name, *args, **kwargs):
        if name not in OpArgMngr._args:
            OpArgMngr._args[name] = []
        OpArgMngr._args[name].append({'args': args, 'kwargs': kwargs})

    @staticmethod
    def get_workloads(name):
        return OpArgMngr._args.get(name, None)


@use_np
def _prepare_workloads():
    array_pool = {
        '4x1': np.random.uniform(size=(4, 1)) + 2,
        '1x2': np.random.uniform(size=(1, 2)) + 2,
        '1x1x0': np.array([[[]]])
    }

    dt_int = [np.int8, np.int32, np.int64, np.uint8]
    dt_float = [np.float16, np.float32, np.float64]
    dt = dt_int + dt_float

    # workloads for array function protocol
    OpArgMngr.add_workload('argmax', array_pool['4x1'])
    OpArgMngr.add_workload('broadcast_arrays', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('broadcast_to', array_pool['4x1'], (4, 2))
    OpArgMngr.add_workload('clip', array_pool['4x1'], 0.2, 0.8)
    OpArgMngr.add_workload('concatenate', [array_pool['4x1'], array_pool['4x1']])
    OpArgMngr.add_workload('concatenate', [array_pool['4x1'], array_pool['4x1']], axis=1)
    OpArgMngr.add_workload('copy', array_pool['4x1'])

    for ctype in dt:
        OpArgMngr.add_workload('cumsum', np.array([1, 2, 10, 11, 6, 5, 4], dtype=ctype))
        OpArgMngr.add_workload('cumsum', np.array([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], dtype=ctype), axis=0)
        OpArgMngr.add_workload('cumsum', np.array([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], dtype=ctype), axis=1)

    OpArgMngr.add_workload('ravel', np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))

    OpArgMngr.add_workload('dot', array_pool['4x1'], array_pool['4x1'].T)
    OpArgMngr.add_workload('expand_dims', array_pool['4x1'], -1)
    OpArgMngr.add_workload('fix', array_pool['4x1'])
    OpArgMngr.add_workload('max', array_pool['4x1'])
    OpArgMngr.add_workload('min', array_pool['4x1'])
    OpArgMngr.add_workload('mean', array_pool['4x1'])
    OpArgMngr.add_workload('mean', array_pool['4x1'], axis=0, keepdims=True)
    OpArgMngr.add_workload('ones_like', array_pool['4x1'])
    OpArgMngr.add_workload('prod', array_pool['4x1'])

    OpArgMngr.add_workload('repeat', array_pool['4x1'], 3)
    OpArgMngr.add_workload('repeat', np.array(_np.arange(12).reshape(4, 3)[:, 2]), 3)

    m = _np.array([1, 2, 3, 4, 5, 6])
    m_rect = m.reshape((2, 3))

    # OpArgMngr.add_workload('repeat', np.array(m), [1, 3, 2, 1, 1, 2]) # Argument "repeats" only supports int
    OpArgMngr.add_workload('repeat', np.array(m), 2)
    B = np.array(m_rect)
    # OpArgMngr.add_workload('repeat', B, [2, 1], axis=0)  # Argument "repeats" only supports int
    # OpArgMngr.add_workload('repeat', B, [1, 3, 2], axis=1)  # Argument "repeats" only supports int
    OpArgMngr.add_workload('repeat', B, 2, axis=0)
    OpArgMngr.add_workload('repeat', B, 2, axis=1)

    # test_repeat_broadcasting
    a = _np.arange(60).reshape(3, 4, 5)
    for axis in itertools.chain(range(-a.ndim, a.ndim), [None]):
        OpArgMngr.add_workload('repeat', np.array(a), 2, axis=axis)
    #    OpArgMngr.add_workload('repeat', np.array(a), [2], axis=axis)   # Argument "repeats" only supports int

    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    OpArgMngr.add_workload('reshape', arr, (2, 6))
    OpArgMngr.add_workload('reshape', arr, (3, 4))
    # OpArgMngr.add_workload('reshape', arr, (3, 4), order='F') # Items are not equal with order='F'
    OpArgMngr.add_workload('reshape', arr, (3, 4), order='C')
    OpArgMngr.add_workload('reshape', np.array(_np.ones(100)), (100, 1, 1))
    
    # test_reshape_order
    a = np.array(_np.arange(6))
    # OpArgMngr.add_workload('reshape', a, (2, 3), order='F')  # Items are not equal with order='F'
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    b = a[:, 1]
    # OpArgMngr.add_workload('reshape', b, (2, 2), order='F')  # Items are not equal with order='F'

    a = np.array(_np.ones((0, 2)))
    OpArgMngr.add_workload('reshape', a, -1, 2)

    OpArgMngr.add_workload('rint', np.array(4607998452777363968))
    OpArgMngr.add_workload('rint', array_pool['4x1'])

    # test_roll1d(self)
    OpArgMngr.add_workload('roll', np.array(_np.arange(10)), 2)

    # test_roll2d(self)
    x2 = np.array(_np.reshape(_np.arange(10), (2, 5)))
    OpArgMngr.add_workload('roll', x2, 1)
    OpArgMngr.add_workload('roll', x2, 1, axis=0)
    OpArgMngr.add_workload('roll', x2, 1, axis=1)
    # # Roll multiple axes at once.
    OpArgMngr.add_workload('roll', x2, 1, axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (1, 0), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (-1, 0), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (0, 1), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (0, -1), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (1, 1), axis=(0, 1))
    OpArgMngr.add_workload('roll', x2, (-1, -1), axis=(0, 1))
    # # Roll the same axis multiple times.
    # OpArgMngr.add_workload('roll', x2, 1, axis=(0, 0)) # Check failed: axes[i - 1] < axes[i] (0 vs. 0) : axes have duplicates [0,0]
    # OpArgMngr.add_workload('roll', x2, 1, axis=(1, 1)) # Check failed: axes[i - 1] < axes[i] (1 vs. 1) : axes have duplicates [1,1]
    # # Roll more than one turn in either direction.
    OpArgMngr.add_workload('roll', x2, 6, axis=1)
    OpArgMngr.add_workload('roll', x2, -4, axis=1)
    # # test_roll_empty
    OpArgMngr.add_workload('roll', np.array([]), 1)
    
    OpArgMngr.add_workload('split', array_pool['4x1'], 2)
    OpArgMngr.add_workload('squeeze', array_pool['4x1'])
    OpArgMngr.add_workload('stack', [array_pool['4x1']] * 2)
    OpArgMngr.add_workload('std', array_pool['4x1'])
    OpArgMngr.add_workload('sum', array_pool['4x1'])
    OpArgMngr.add_workload('swapaxes', array_pool['4x1'], 0, 1)
    OpArgMngr.add_workload('tensordot', array_pool['4x1'], array_pool['4x1'])
    OpArgMngr.add_workload('tile', array_pool['4x1'], 2)
    OpArgMngr.add_workload('tile', np.array([[[]]]), (3, 2, 5))
    OpArgMngr.add_workload('transpose', array_pool['4x1'])
    OpArgMngr.add_workload('var', array_pool['4x1'])
    OpArgMngr.add_workload('zeros_like', array_pool['4x1'])

    # workloads for array ufunc protocol
    OpArgMngr.add_workload('add', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('add', array_pool['4x1'], 2)
    OpArgMngr.add_workload('add', 2, array_pool['4x1'])
    OpArgMngr.add_workload('add', array_pool['4x1'], array_pool['1x1x0'])
    OpArgMngr.add_workload('subtract', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('subtract', array_pool['4x1'], 2)
    OpArgMngr.add_workload('subtract', 2, array_pool['4x1'])
    OpArgMngr.add_workload('subtract', array_pool['4x1'], array_pool['1x1x0'])
    OpArgMngr.add_workload('multiply', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('multiply', array_pool['4x1'], 2)
    OpArgMngr.add_workload('multiply', 2, array_pool['4x1'])
    OpArgMngr.add_workload('multiply', array_pool['4x1'], array_pool['1x1x0'])
    OpArgMngr.add_workload('power', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('power', array_pool['4x1'], 2)
    OpArgMngr.add_workload('power', 2, array_pool['4x1'])
    OpArgMngr.add_workload('power', array_pool['4x1'], array_pool['1x1x0'])
    OpArgMngr.add_workload('mod', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('mod', array_pool['4x1'], 2)
    OpArgMngr.add_workload('mod', 2, array_pool['4x1'])
    OpArgMngr.add_workload('mod', array_pool['4x1'], array_pool['1x1x0'])

    # test remainder basic
    OpArgMngr.add_workload('remainder', np.array([0, 1, 2, 4, 2], dtype=np.float16),
                            np.array([-2, 5, 1, 4, 3], dtype=np.float16))

    def _signs(dt):
        if dt in [np.uint8]:
            return (+1,)
        else:
            return (+1, -1)

    for ct in dt:
        for sg1, sg2 in itertools.product(_signs(ct), _signs(ct)):
            a = np.array(sg1*71, dtype=ct)
            b = np.array(sg2*19, dtype=ct)
            OpArgMngr.add_workload('remainder', a, b)

    # test remainder exact
    nlst = list(range(-127, 0))
    plst = list(range(1, 128))
    dividend = nlst + [0] + plst
    divisor = nlst + plst
    arg = list(itertools.product(dividend, divisor))
    tgt = list(divmod(*t) for t in arg)
    a, b = np.array(arg, dtype=int).T
    # convert exact integer results from Python to float so that
    # signed zero can be used, it is checked.
    for dt in [np.float16, np.float32, np.float64]:
        fa = a.astype(dt)
        fb = b.astype(dt)
        OpArgMngr.add_workload('remainder', fa, fb)
    
    # test_float_remainder_roundoff
    for ct in dt_float:
        for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
            a = np.array(sg1*78*6e-8, dtype=ct)
            b = np.array(sg2*6e-8, dtype=ct)
            OpArgMngr.add_workload('remainder', a, b)

    # test_float_remainder_corner_cases
    # Check remainder magnitude.
    for ct in dt_float:
        b = _np.array(1.0)
        a = np.array(_np.nextafter(_np.array(0.0), -b), dtype=ct)
        b = np.array(b, dtype=ct)
        OpArgMngr.add_workload('remainder', a, b)
        OpArgMngr.add_workload('remainder', -a, -b)

        # Check nans, inf
        for ct in [np.float16, np.float32, np.float64]:
            fone = np.array(1.0, dtype=ct)
            fzer = np.array(0.0, dtype=ct)
            finf = np.array(np.inf, dtype=ct)
            fnan = np.array(np.nan, dtype=ct)
            # OpArgMngr.add_workload('remainder', fone, fzer) # failed
            OpArgMngr.add_workload('remainder', fone, fnan)
            OpArgMngr.add_workload('remainder', finf, fone)

    OpArgMngr.add_workload('maximum', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('maximum', array_pool['4x1'], 2)
    OpArgMngr.add_workload('maximum', 2, array_pool['4x1'])
    OpArgMngr.add_workload('maximum', array_pool['4x1'], array_pool['1x1x0'])
    OpArgMngr.add_workload('minimum', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('minimum', array_pool['4x1'], 2)
    OpArgMngr.add_workload('minimum', 2, array_pool['4x1'])
    OpArgMngr.add_workload('minimum', array_pool['4x1'], array_pool['1x1x0'])
    OpArgMngr.add_workload('negative', array_pool['4x1'])
    OpArgMngr.add_workload('absolute', array_pool['4x1'])
    
    OpArgMngr.add_workload('sign', array_pool['4x1'])
    OpArgMngr.add_workload('sign', np.array([-2, 5, 1, 4, 3], dtype=np.float16))
    OpArgMngr.add_workload('sign', np.array([-.1, 0, .1]))
    # OpArgMngr.add_workload('sign', np.array(_np.array([_np.nan]))) # failed

    OpArgMngr.add_workload('exp', array_pool['4x1'])
    OpArgMngr.add_workload('log', array_pool['4x1'])
    OpArgMngr.add_workload('log2', array_pool['4x1'])
    OpArgMngr.add_workload('log10', array_pool['4x1'])
    OpArgMngr.add_workload('expm1', array_pool['4x1'])
    OpArgMngr.add_workload('sqrt', array_pool['4x1'])
    OpArgMngr.add_workload('square', array_pool['4x1'])
    OpArgMngr.add_workload('cbrt', array_pool['4x1'])

    for ctype in [np.float16, np.float32, np.float64]:
        OpArgMngr.add_workload('reciprocal', np.array([-2, 5, 1, 4, 3], dtype=ctype))
        OpArgMngr.add_workload('reciprocal', np.array([-2, 0, 1, 0, 3], dtype=ctype))
        OpArgMngr.add_workload('reciprocal', np.array([0], dtype=ctype))

    OpArgMngr.add_workload('sin', array_pool['4x1'])
    OpArgMngr.add_workload('cos', array_pool['4x1'])
    OpArgMngr.add_workload('tan', array_pool['4x1'])
    OpArgMngr.add_workload('sinh', array_pool['4x1'])
    OpArgMngr.add_workload('cosh', array_pool['4x1'])
    OpArgMngr.add_workload('tanh', array_pool['4x1'])
    OpArgMngr.add_workload('arcsin', array_pool['4x1'] - 2)
    OpArgMngr.add_workload('arccos', array_pool['4x1'] - 2)
    OpArgMngr.add_workload('arctan', array_pool['4x1'])
    OpArgMngr.add_workload('arcsinh', array_pool['4x1'])
    OpArgMngr.add_workload('arccosh', array_pool['4x1'])
    OpArgMngr.add_workload('arctanh', array_pool['4x1'] - 2)
    OpArgMngr.add_workload('ceil', array_pool['4x1'])
    OpArgMngr.add_workload('trunc', array_pool['4x1'])
    OpArgMngr.add_workload('floor', array_pool['4x1'])


_prepare_workloads()


def _get_numpy_op_output(onp_op, *args, **kwargs):
    onp_args = [arg.asnumpy() if isinstance(arg, np.ndarray) else arg for arg in args]
    onp_kwargs = {k: v.asnumpy() if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    for i, v in enumerate(onp_args):
        if isinstance(v, (list, tuple)):
            new_arrs = [a.asnumpy() if isinstance(a, np.ndarray) else a for a in v]
            onp_args[i] = new_arrs

    return onp_op(*onp_args, **onp_kwargs)


def _check_interoperability_helper(op_name, *args, **kwargs):
    strs = op_name.split('.')
    if len(strs) == 1:
        onp_op = getattr(_np, op_name)
    elif len(strs) == 2:
        onp_op = getattr(getattr(_np, strs[0]), strs[1])
    else:
        assert False
    out = onp_op(*args, **kwargs)
    expected_out = _get_numpy_op_output(onp_op, *args, **kwargs)
    if isinstance(out, (tuple, list)):
        assert type(out) == type(expected_out)
        for arr in out:
            assert isinstance(arr, np.ndarray)
        for arr, expected_arr in zip(out, expected_out):
            assert isinstance(arr, np.ndarray)
            assert_almost_equal(arr.asnumpy(), expected_arr, rtol=1e-3, atol=1e-4, use_broadcast=False, equal_nan=True)
    else:
        assert isinstance(out, np.ndarray)
        assert_almost_equal(out.asnumpy(), expected_out, rtol=1e-3, atol=1e-4, use_broadcast=False, equal_nan=True)


def check_interoperability(op_list):
    for name in op_list:
        workloads = OpArgMngr.get_workloads(name)
        assert workloads is not None, 'Workloads for operator `{}` has not been ' \
                                      'added for checking interoperability with ' \
                                      'the official NumPy.'.format(name)
        for workload in workloads:
            _check_interoperability_helper(name, *workload['args'], **workload['kwargs'])


@with_seed()
@use_np
@with_array_function_protocol
def test_np_array_function_protocol():
    check_interoperability(_NUMPY_ARRAY_FUNCTION_LIST)


@with_seed()
@use_np
@with_array_ufunc_protocol
def test_np_array_ufunc_protocol():
    check_interoperability(_NUMPY_ARRAY_UFUNC_LIST)


if __name__ == '__main__':
    import nose
    nose.runmodule()
