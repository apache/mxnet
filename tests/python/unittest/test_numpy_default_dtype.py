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

from __future__ import absolute_import
import numpy as _np
import mxnet
from mxnet import npx
from mxnet import numpy as np
from mxnet.test_utils import use_np, use_np_default_dtype


class DtypeOpArgMngr(object):
    """Operator argument manager for storing operator workloads."""
    _args = {}

    @staticmethod
    def add_workload(name, *args, **kwargs):
        if name not in DtypeOpArgMngr._args:
            DtypeOpArgMngr._args[name] = []
        DtypeOpArgMngr._args[name].append({'args': args, 'kwargs': kwargs})

    @staticmethod
    def get_workloads(name):
        return DtypeOpArgMngr._args.get(name, None)


_NUMPY_DTYPE_DEFAULT_FUNC_LIST = [
    'array',
    'ones',
    'zeros',
    'eye',
    'full',
    'identity',
    'linspace',
    'logspace',
    'mean',
    'hanning',
    'hamming',
    'blackman',
    'random.gamma',
    'random.uniform',
    'random.normal',
    'random.chisquare',
    'true_divide'
]


def _add_dtype_workload_array():
    DtypeOpArgMngr.add_workload('array', [1, 2, 3])


def _add_dtype_workload_ones():
    DtypeOpArgMngr.add_workload('ones', 5)
    DtypeOpArgMngr.add_workload('ones', (5,))


def _add_dtype_workload_zeros():
    DtypeOpArgMngr.add_workload('zeros', 5)
    DtypeOpArgMngr.add_workload('zeros', (5,))


def _add_dtype_workload_eye():
    DtypeOpArgMngr.add_workload('eye', 3)
    DtypeOpArgMngr.add_workload('eye', 3, k=1)


def _add_dtype_workload_full():
    DtypeOpArgMngr.add_workload('full', (2, 2), 10)


def _add_dtype_workload_identity():
    DtypeOpArgMngr.add_workload('identity', 3)


def _add_dtype_workload_linspace():
    DtypeOpArgMngr.add_workload('linspace', 2.0, 3.0, num=5)
    DtypeOpArgMngr.add_workload('linspace', 2.0, 3.0, num=5, endpoint=False)


def _add_dtype_workload_logspace():
    DtypeOpArgMngr.add_workload('logspace', 2.0, 3.0, num=4)
    DtypeOpArgMngr.add_workload('logspace', 2.0, 3.0, num=4, endpoint=False)
    DtypeOpArgMngr.add_workload('logspace', 2.0, 3.0, num=4, base=2.0)


def _add_dtype_workload_mean():
    DtypeOpArgMngr.add_workload('mean', np.random.randint(0, 3,size=2))


def _add_dtype_workload_hanning():
    DtypeOpArgMngr.add_workload('hanning', 3)


def _add_dtype_workload_hamming():
    DtypeOpArgMngr.add_workload('hamming', 3)


def _add_dtype_workload_blackman():
    DtypeOpArgMngr.add_workload('blackman', 3)


def _add_dtype_workload_random_uniform():
    DtypeOpArgMngr.add_workload('random.uniform', -1, 1, size=3)


def _add_dtype_workload_random_normal():
    DtypeOpArgMngr.add_workload('random.normal', 0, 0.1, 3)


def _add_dtype_workload_random_gamma():
    DtypeOpArgMngr.add_workload('random.gamma', 3)


def _add_dtype_workload_random_chisquare():
    DtypeOpArgMngr.add_workload('random.chisquare', 2, 4)


def _add_dtype_workload_true_divide():
    DtypeOpArgMngr.add_workload('true_divide', np.array([1,2], dtype=int), 4)
    DtypeOpArgMngr.add_workload('true_divide', np.array([1,2], dtype=int), 2.0)
    DtypeOpArgMngr.add_workload('true_divide', 4.0, np.array([1,2], dtype=int))


def _prepare_workloads():
    _add_dtype_workload_array()
    _add_dtype_workload_ones()
    _add_dtype_workload_zeros()
    _add_dtype_workload_eye()
    _add_dtype_workload_full()
    _add_dtype_workload_identity()
    _add_dtype_workload_linspace()
    _add_dtype_workload_logspace()
    _add_dtype_workload_mean()
    _add_dtype_workload_hanning()
    _add_dtype_workload_hamming()
    _add_dtype_workload_blackman()
    _add_dtype_workload_random_gamma()
    _add_dtype_workload_random_uniform()
    _add_dtype_workload_random_normal()
    _add_dtype_workload_true_divide()
    _add_dtype_workload_random_chisquare()

_prepare_workloads()


@use_np
@use_np_default_dtype
def check_np_default_dtype(op, *args, **kwargs):
    assert op(*args, **kwargs).dtype == 'float64'


@use_np
def check_deepnp_default_dtype(op, *args, **kwargs):
    assert op(*args, **kwargs).dtype == 'float32'


def check_default_dtype(op_list):
    for op_name in op_list:
        print('Default dtype test:', op_name)
        workloads = DtypeOpArgMngr.get_workloads(op_name)
        strs = op_name.split('.')
        if len(strs) == 1:
            op = getattr(np, op_name)
        elif len(strs) == 2:
            op = getattr(getattr(np, strs[0]), strs[1])
        else:
            assert False
        assert workloads is not None, 'Workloads for operator `{}` has not been ' \
                                      'added for checking default dtype with the ' \
                                      'official NumPy and the deep NumPy.'.format(name)
        for workload in workloads:
            check_np_default_dtype(op, *workload['args'], **workload['kwargs'])
            check_deepnp_default_dtype(op, *workload['args'], **workload['kwargs'])


def test_default_float_dtype():
    import platform
    if 'Windows' not in platform.system():
        check_default_dtype(_NUMPY_DTYPE_DEFAULT_FUNC_LIST)


@use_np
def test_np_indices_default_dtype():
    import platform
    if 'Windows' not in platform.system():
        @use_np_default_dtype
        def check_np_indices_default_dtype():
            assert np.indices((3,)).dtype == 'int64'

        def check_deepnp_indices_default_dtype():
            assert np.indices((3,)).dtype == 'int64'
        
        check_deepnp_indices_default_dtype()
        check_np_indices_default_dtype()


@use_np
def test_np_arange_default_dtype():
    import platform
    if 'Windows' not in platform.system():
        @use_np_default_dtype
        def check_np_indices_default_dtype():
            assert np.arange(3, 7, 2).dtype == 'int64'

        def check_deepnp_indices_default_dtype():
            assert np.arange(3, 7, 2).dtype == 'float32'
        
        check_deepnp_indices_default_dtype()
        check_np_indices_default_dtype()
