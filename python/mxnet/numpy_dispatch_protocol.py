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

"""Utils for registering NumPy array function protocol for mxnet.numpy ops."""

import functools
import numpy as _np
from . import numpy as mx_np  # pylint: disable=reimported
from .numpy.multiarray import _NUMPY_ARRAY_FUNCTION_DICT, _NUMPY_ARRAY_UFUNC_DICT


def _find_duplicate(strs):
    str_set = set()
    for s in strs:
        if s in str_set:
            return s
        else:
            str_set.add(s)
    return None


def _implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        _NUMPY_ARRAY_FUNCTION_DICT[numpy_function] = func
        return func
    return decorator


def with_array_function_protocol(func):
    """A decorator for functions that expect array function protocol.
    The decorated function only runs when NumPy version >= 1.17."""
    from distutils.version import LooseVersion
    cur_np_ver = LooseVersion(_np.__version__)
    np_1_17_ver = LooseVersion('1.17')

    @functools.wraps(func)
    def _run_with_array_func_proto(*args, **kwargs):
        if cur_np_ver >= np_1_17_ver:
            try:
                func(*args, **kwargs)
            except Exception as e:
                raise RuntimeError('Running function {} with NumPy array function protocol failed'
                                   ' with exception {}'
                                   .format(func.__name__, str(e)))

    return _run_with_array_func_proto


def with_array_ufunc_protocol(func):
    """A decorator for functions that expect array ufunc protocol.
    The decorated function only runs when NumPy version >= 1.15."""
    from distutils.version import LooseVersion
    cur_np_ver = LooseVersion(_np.__version__)
    np_1_15_ver = LooseVersion('1.15')

    @functools.wraps(func)
    def _run_with_array_ufunc_proto(*args, **kwargs):
        if cur_np_ver >= np_1_15_ver:
            try:
                func(*args, **kwargs)
            except Exception as e:
                raise RuntimeError('Running function {} with NumPy array ufunc protocol failed'
                                   ' with exception {}'
                                   .format(func.__name__, str(e)))

    return _run_with_array_ufunc_proto


_NUMPY_ARRAY_FUNCTION_LIST = [
    'all',
    'any',
    'sometrue',
    'argmin',
    'argmax',
    'around',
    'round',
    'round_',
    'argsort',
    'sort',
    'append',
    'broadcast_arrays',
    'broadcast_to',
    'clip',
    'concatenate',
    'copy',
    'cumsum',
    'diag',
    'diagonal',
    'diagflat',
    'dot',
    'expand_dims',
    'fix',
    'flip',
    'flipud',
    'fliplr',
    'inner',
    'insert',
    'max',
    'amax',
    'mean',
    'min',
    'amin',
    'nonzero',
    'ones_like',
    'atleast_1d',
    'atleast_2d',
    'atleast_3d',
    'prod',
    'product',
    'ravel',
    'repeat',
    'reshape',
    'roll',
    'split',
    'array_split',
    'hsplit',
    'vsplit',
    'dsplit',
    'squeeze',
    'stack',
    'std',
    'sum',
    'swapaxes',
    'take',
    'tensordot',
    'tile',
    'transpose',
    'unique',
    'unravel_index',
    'diag_indices_from',
    'delete',
    'var',
    'vdot',
    'vstack',
    'column_stack',
    'hstack',
    'dstack',
    'zeros_like',
    'linalg.norm',
    'linalg.cholesky',
    'linalg.inv',
    'linalg.solve',
    'linalg.tensorinv',
    'linalg.tensorsolve',
    'linalg.pinv',
    'linalg.eigvals',
    'linalg.eig',
    'linalg.eigvalsh',
    'linalg.eigh',
    'shape',
    'trace',
    'tril',
    'meshgrid',
    'outer',
    'einsum',
    'polyval',
    'shares_memory',
    'may_share_memory',
    'quantile',
    'percentile',
    'diff',
    'ediff1d',
    'resize',
    'where',
    'full_like',
    'bincount',
    'empty_like',
    'nan_to_num',
    'isnan',
    'isfinite',
    'isposinf',
    'isneginf',
    'isinf',
    'pad',
]


@with_array_function_protocol
def _register_array_function():
    """Register __array_function__ protocol for mxnet.numpy operators so that
    ``mxnet.numpy.ndarray`` can be fed into the official NumPy operators and
    dispatched to MXNet implementation.

    Notes
    -----
    According the __array_function__ protocol (see the following reference),
    there are three kinds of operators that cannot be dispatched using this
    protocol:
    1. Universal functions, which already have their own protocol in the official
    NumPy package.
    2. Array creation functions.
    3. Dispatch for methods of any kind, e.g., methods on np.random.RandomState objects.

    References
    ----------
    https://numpy.org/neps/nep-0018-array-function-protocol.html
    """
    dup = _find_duplicate(_NUMPY_ARRAY_FUNCTION_LIST)
    if dup is not None:
        raise ValueError('Duplicate operator name {} in _NUMPY_ARRAY_FUNCTION_LIST'.format(dup))
    for op_name in _NUMPY_ARRAY_FUNCTION_LIST:
        strs = op_name.split('.')
        if len(strs) == 1:
            mx_np_op = getattr(mx_np, op_name)
            onp_op = getattr(_np, op_name)
            setattr(mx_np, op_name, _implements(onp_op)(mx_np_op))
        elif len(strs) == 2:
            mx_np_submodule = getattr(mx_np, strs[0])
            mx_np_op = getattr(mx_np_submodule, strs[1])
            onp_submodule = getattr(_np, strs[0])
            onp_op = getattr(onp_submodule, strs[1])
            setattr(mx_np_submodule, strs[1], _implements(onp_op)(mx_np_op))
        else:
            raise ValueError('Does not support registering __array_function__ protocol '
                             'for operator {}'.format(op_name))


# https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs
_NUMPY_ARRAY_UFUNC_LIST = [
    'abs',
    'fabs',
    'add',
    'arctan2',
    'copysign',
    'degrees',
    'hypot',
    'lcm',
    # 'ldexp',
    'subtract',
    'multiply',
    'true_divide',
    'negative',
    'power',
    'mod',
    'matmul',
    'absolute',
    'rint',
    'sign',
    'exp',
    'log',
    'log2',
    'log10',
    'expm1',
    'sqrt',
    'square',
    'cbrt',
    'reciprocal',
    'invert',
    'bitwise_not',
    'remainder',
    'sin',
    'cos',
    'tan',
    'sinh',
    'cosh',
    'tanh',
    'arcsin',
    'arccos',
    'arctan',
    'arcsinh',
    'arccosh',
    'arctanh',
    'maximum',
    'minimum',
    'ceil',
    'trunc',
    'floor',
    'bitwise_and',
    'bitwise_xor',
    'bitwise_or',
    'logical_not',
    'equal',
    'not_equal',
    'less',
    'less_equal',
    'greater',
    'greater_equal',
]


@with_array_ufunc_protocol
def _register_array_ufunc():
    """Register NumPy array ufunc protocol.

    References
    ----------
    https://numpy.org/neps/nep-0013-ufunc-overrides.html
    """
    dup = _find_duplicate(_NUMPY_ARRAY_UFUNC_LIST)
    if dup is not None:
        raise ValueError('Duplicate operator name {} in _NUMPY_ARRAY_UFUNC_LIST'.format(dup))
    for op_name in _NUMPY_ARRAY_UFUNC_LIST:
        try:
            mx_np_op = getattr(mx_np, op_name)
            _NUMPY_ARRAY_UFUNC_DICT[op_name] = mx_np_op
        except AttributeError:
            raise AttributeError('mxnet.numpy does not have operator named {}'.format(op_name))


_register_array_function()
_register_array_ufunc()
