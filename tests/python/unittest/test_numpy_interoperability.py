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

    # workloads for array function protocol
    OpArgMngr.add_workload('argmax', array_pool['4x1'])
    OpArgMngr.add_workload('broadcast_arrays', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('broadcast_to', array_pool['4x1'], (4, 2))
    OpArgMngr.add_workload('clip', array_pool['4x1'], 0.2, 0.8)
    OpArgMngr.add_workload('concatenate', [array_pool['4x1'], array_pool['4x1']])
    OpArgMngr.add_workload('concatenate', [array_pool['4x1'], array_pool['4x1']], axis=1)
    OpArgMngr.add_workload('copy', array_pool['4x1'])
    OpArgMngr.add_workload('cumsum', array_pool['4x1'])
    OpArgMngr.add_workload('cumsum', array_pool['4x1'], axis=1)
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
    OpArgMngr.add_workload('reshape', array_pool['4x1'], -1)
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
    OpArgMngr.add_workload('remainder', array_pool['4x1'], array_pool['1x2'])
    OpArgMngr.add_workload('remainder', array_pool['4x1'], 2)
    OpArgMngr.add_workload('remainder', 2, array_pool['4x1'])
    OpArgMngr.add_workload('remainder', array_pool['4x1'], array_pool['1x1x0'])
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
    OpArgMngr.add_workload('rint', array_pool['4x1'])
    OpArgMngr.add_workload('sign', array_pool['4x1'])
    OpArgMngr.add_workload('exp', array_pool['4x1'])
    OpArgMngr.add_workload('log', array_pool['4x1'])
    OpArgMngr.add_workload('log2', array_pool['4x1'])
    OpArgMngr.add_workload('log10', array_pool['4x1'])
    OpArgMngr.add_workload('expm1', array_pool['4x1'])
    OpArgMngr.add_workload('sqrt', array_pool['4x1'])
    OpArgMngr.add_workload('square', array_pool['4x1'])
    OpArgMngr.add_workload('cbrt', array_pool['4x1'])
    OpArgMngr.add_workload('reciprocal', array_pool['4x1'])
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
            assert_almost_equal(arr.asnumpy(), expected_arr, rtol=1e-3, atol=1e-4, use_broadcast=False)
    else:
        assert isinstance(out, np.ndarray)
        assert_almost_equal(out.asnumpy(), expected_out, rtol=1e-3, atol=1e-4, use_broadcast=False)


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
