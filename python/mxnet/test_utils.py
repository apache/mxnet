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

"""Tools for testing."""
# pylint: disable=too-many-lines
from __future__ import absolute_import, print_function, division
import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import errno
import logging
import bz2
import zipfile
from contextlib import contextmanager
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
try:
    import scipy.stats as ss
except ImportError:
    ss = None
try:
    import requests
except ImportError:
    # in rare cases requests may be not installed
    pass
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol


def default_context():
    """Get default context for regression test."""
    # _TODO: get context from environment variable to support
    # testing with GPUs
    return current_context()


def set_default_context(ctx):
    """Set default context."""
    Context._default_ctx.value = ctx


def default_dtype():
    """Get default data type for regression test."""
    # _TODO: get default dtype from environment variable
    return np.float32


def get_atol(atol=None):
    """Get default numerical threshold for regression test."""
    # _TODO: get from env variable, different threshold might
    # be needed for different device and dtype
    return 1e-20 if atol is None else atol


def get_rtol(rtol=None):
    """Get default numerical threshold for regression test."""
    # _TODO: get from env variable, different threshold might
    # be needed for different device and dtype
    return 1e-5 if rtol is None else rtol

def get_etol(etol=None):
    """Get default numerical threshold for regression test."""
    # _TODO: get from env variable, different threshold might
    # be needed for different device and dtype
    return 0 if etol is None else etol

def random_arrays(*shapes):
    """Generate some random numpy arrays."""
    arrays = [np.random.randn(*s).astype(default_dtype())
              for s in shapes]
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def random_sample(population, k):
    """Return a k length list of the elements chosen from the population sequence."""
    assert 0 <= k <= len(population)
    population_copy = population[:]
    np.random.shuffle(population_copy)
    return population_copy[0:k]


def _validate_csr_generation_inputs(num_rows, num_cols, density,
                                    distribution="uniform"):
    """Validates inputs for csr generation helper functions
    """
    total_nnz = int(num_rows * num_cols * density)
    if density < 0 or density > 1:
        raise ValueError("density has to be between 0 and 1")

    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("num_rows or num_cols should be greater than 0")

    if distribution == "powerlaw":
        if total_nnz < 2 * num_rows:
            raise ValueError("not supported for this density: %s"
                             " for this shape (%s, %s)"
                             " Please keep :"
                             " num_rows * num_cols * density >= 2 * num_rows"
                             % (density, num_rows, num_cols))


def shuffle_csr_column_indices(csr):
    """Shuffle CSR column indices per row
    This allows validation of unordered column indices, which is not a requirement
    for a valid CSR matrix
    """
    row_count = len(csr.indptr) - 1
    for i in range(row_count):
        start_index = csr.indptr[i]
        end_index = csr.indptr[i + 1]
        sublist = np.array(csr.indices[start_index : end_index])
        np.random.shuffle(sublist)
        csr.indices[start_index : end_index] = sublist


def _get_uniform_dataset_csr(num_rows, num_cols, density=0.1, dtype=None,
                             data_init=None, shuffle_csr_indices=False):
    """Returns CSRNDArray with uniform distribution
    This generates a csr matrix with totalnnz unique randomly chosen numbers
    from num_rows*num_cols and arranges them in the 2d array in the
    following way:
    row_index = (random_number_generated / num_rows)
    col_index = random_number_generated - row_index * num_cols
    """
    _validate_csr_generation_inputs(num_rows, num_cols, density,
                                    distribution="uniform")
    try:
        from scipy import sparse as spsp
        csr = spsp.rand(num_rows, num_cols, density, dtype=dtype, format="csr")
        if data_init is not None:
            csr.data.fill(data_init)
        if shuffle_csr_indices is True:
            shuffle_csr_column_indices(csr)
        result = mx.nd.sparse.csr_matrix((csr.data, csr.indices, csr.indptr),
                                         shape=(num_rows, num_cols), dtype=dtype)
    except ImportError:
        assert(data_init is None), \
               "data_init option is not supported when scipy is absent"
        assert(not shuffle_csr_indices), \
               "shuffle_csr_indices option is not supported when scipy is absent"
        # scipy not available. try to generate one from a dense array
        dns = mx.nd.random.uniform(shape=(num_rows, num_cols), dtype=dtype)
        masked_dns = dns * (dns < density)
        result = masked_dns.tostype('csr')
    return result

def _get_powerlaw_dataset_csr(num_rows, num_cols, density=0.1, dtype=None):
    """Returns CSRNDArray with powerlaw distribution
    with exponentially increasing number of non zeros in each row.
    Not supported for cases where total_nnz < 2*num_rows. This is because
    the algorithm first tries to ensure that there are rows with no zeros by
    putting non zeros at beginning of each row.
    """

    _validate_csr_generation_inputs(num_rows, num_cols, density,
                                    distribution="powerlaw")

    total_nnz = int(num_rows * num_cols * density)

    unused_nnz = total_nnz
    output_arr = np.zeros((num_rows, num_cols), dtype=dtype)
    # Start with ones on each row so that no row is empty
    for row in range(num_rows):
        output_arr[row][0] = 1 + rnd.uniform(0.001, 2)
        unused_nnz = unused_nnz - 1
        if unused_nnz <= 0:
            return mx.nd.array(output_arr).tostype("csr")

    # Populate rest of matrix with 2^i items in ith row.
    # if we have used all total nnz return the sparse matrix
    # else if we reached max column size then fill up full columns until we use all nnz
    col_max = 2
    for row in range(num_rows):
        col_limit = min(num_cols, col_max)
        # In case col_limit reached assign same value to all elements, which is much faster
        if col_limit == num_cols and unused_nnz > col_limit:
            output_arr[row] = 1 + rnd.uniform(0.001, 2)
            unused_nnz = unused_nnz - col_limit + 1
            if unused_nnz <= 0:
                return mx.nd.array(output_arr).tostype("csr")
            else:
                continue
        for col_index in range(1, col_limit):
            output_arr[row][col_index] = 1 + rnd.uniform(0.001, 2)
            unused_nnz = unused_nnz - 1
            if unused_nnz <= 0:
                return mx.nd.array(output_arr).tostype("csr")
        col_max = col_max * 2

    if unused_nnz > 0: # pylint: disable=no-else-raise
        raise ValueError("not supported for this density: %s"
                         " for this shape (%s,%s)" % (density, num_rows, num_cols))
    else:
        return mx.nd.array(output_arr).tostype("csr")


def assign_each(the_input, function):
    """Return ndarray composed of passing each array value through some function"""
    if function is None:
        output = np.array(the_input)
    else:
        it_input = np.nditer(the_input, flags=['f_index'])

        output = np.zeros(the_input.shape)
        it_out = np.nditer(output, flags=['f_index'], op_flags=['writeonly'])

        while not it_input.finished:
            val_input = it_input[0]
            it_out[0] = function(val_input)
            it_input.iternext()
            it_out.iternext()

    return output

def assign_each2(input1, input2, function):
    """Return ndarray composed of passing two array values through some function"""
    if function is None:
        output = np.array(input1)
    else:
        assert input1.shape == input2.shape
        it_input1 = np.nditer(input1, flags=['f_index'])
        it_input2 = np.nditer(input2, flags=['f_index'])

        output = np.zeros(input1.shape)
        it_out = np.nditer(output, flags=['f_index'], op_flags=['writeonly'])

        while not it_input1.finished:
            val_input1 = it_input1[0]
            val_input2 = it_input2[0]
            it_out[0] = function(val_input1, val_input2)
            it_input1.iternext()
            it_input2.iternext()
            it_out.iternext()

    return output

def rand_sparse_ndarray(shape, stype, density=None, dtype=None, distribution=None,
                        data_init=None, rsp_indices=None, modifier_func=None,
                        shuffle_csr_indices=False, ctx=None):
    """Generate a random sparse ndarray. Returns the ndarray, value(np) and indices(np)

    Parameters
    ----------
    shape: list or tuple
    stype: str
        valid values: "csr" or "row_sparse"
    density: float, optional
        should be between 0 and 1
    distribution: str, optional
        valid values: "uniform" or "powerlaw"
    dtype: numpy.dtype, optional
        default value is None

    Returns
    -------
    Result of type CSRNDArray or RowSparseNDArray

    Examples
    --------
    Below is an example of the powerlaw distribution with csr as the stype.
    It calculates the nnz using the shape and density.
    It fills up the ndarray with exponentially increasing number of elements.
    If there are enough unused_nnzs, n+1th row will have twice more nnzs compared to nth row.
    else, remaining unused_nnzs will be used in n+1th row
    If number of cols is too small and we have already reached column size it will fill up
    all following columns in all followings rows until we reach the required density.

    >>> csr_arr, _ = rand_sparse_ndarray(shape=(5, 16), stype="csr",
                                         density=0.50, distribution="powerlaw")
    >>> indptr = csr_arr.indptr.asnumpy()
    >>> indices = csr_arr.indices.asnumpy()
    >>> data = csr_arr.data.asnumpy()
    >>> row2nnz = len(data[indptr[1]:indptr[2]])
    >>> row3nnz = len(data[indptr[2]:indptr[3]])
    >>> assert(row3nnz == 2*row2nnz)
    >>> row4nnz = len(data[indptr[3]:indptr[4]])
    >>> assert(row4nnz == 2*row3nnz)

    """
    ctx = ctx if ctx else default_context()
    density = rnd.rand() if density is None else density
    dtype = default_dtype() if dtype is None else dtype
    distribution = "uniform" if distribution is None else distribution
    if stype == 'row_sparse':
        assert (distribution == "uniform"), \
               "Distribution %s not supported for row_sparse" % (distribution)
        # sample index
        if rsp_indices is not None:
            indices = rsp_indices
            assert(len(indices) <= shape[0])
        else:
            idx_sample = rnd.rand(shape[0])
            indices = np.argwhere(idx_sample < density).flatten()
        if indices.shape[0] == 0:
            result = mx.nd.zeros(shape, stype='row_sparse', dtype=dtype, ctx=ctx)
            return result, (np.array([], dtype=dtype), np.array([]))
        # generate random values
        val = rnd.rand(indices.shape[0], *shape[1:]).astype(dtype)

        # Allow caller to override or adjust random values
        if data_init is not None:
            val.fill(data_init)
        if modifier_func is not None:
            val = assign_each(val, modifier_func)

        arr = mx.nd.sparse.row_sparse_array((val, indices), shape=shape, dtype=dtype, ctx=ctx)
        return arr, (val, indices)
    elif stype == 'csr':
        assert len(shape) == 2
        if distribution == "uniform":
            csr = _get_uniform_dataset_csr(shape[0], shape[1], density,
                                           data_init=data_init,
                                           shuffle_csr_indices=shuffle_csr_indices, dtype=dtype).as_in_context(ctx)
            return csr, (csr.indptr, csr.indices, csr.data)
        elif distribution == "powerlaw":
            csr = _get_powerlaw_dataset_csr(shape[0], shape[1], density=density, dtype=dtype).as_in_context(ctx)
            return csr, (csr.indptr, csr.indices, csr.data)
        else:
            assert(False), "Distribution not supported: %s" % (distribution)
            return False
    else:
        assert(False), "unknown storage type"
        return False

def rand_ndarray(shape, stype='default', density=None, dtype=None, modifier_func=None,
                 shuffle_csr_indices=False, distribution=None, ctx=None):
    """Generate a random sparse ndarray. Returns the generated ndarray."""
    ctx = ctx if ctx else default_context()
    if stype == 'default':
        arr = mx.nd.array(random_arrays(shape), dtype=dtype, ctx=ctx)
    else:
        arr, _ = rand_sparse_ndarray(shape, stype, density=density,
                                     modifier_func=modifier_func, dtype=dtype,
                                     shuffle_csr_indices=shuffle_csr_indices,
                                     distribution=distribution, ctx=ctx)
    return arr


def create_sparse_array(shape, stype, data_init=None, rsp_indices=None,
                        dtype=None, modifier_func=None, density=.5,
                        shuffle_csr_indices=False):
    """Create a sparse array, For Rsp, assure indices are in a canonical format"""
    if stype == 'row_sparse':
        if rsp_indices is not None:
            arr_indices = np.asarray(rsp_indices)
            arr_indices.sort()
        else:
            arr_indices = None
        arr_data, (_, _) = rand_sparse_ndarray(shape, stype,
                                               density=density,
                                               data_init=data_init,
                                               rsp_indices=arr_indices,
                                               dtype=dtype,
                                               modifier_func=modifier_func)
    elif stype == 'csr':
        arr_data, (_, _, _) = rand_sparse_ndarray(shape,
                                                  stype,
                                                  density=density,
                                                  data_init=data_init,
                                                  dtype=dtype,
                                                  modifier_func=modifier_func,
                                                  shuffle_csr_indices=shuffle_csr_indices)
    else:
        msg = "Unknown storage type: " + stype
        raise AssertionError(msg)

    return arr_data


def create_sparse_array_zd(shape, stype, density, data_init=None,
                           rsp_indices=None, dtype=None, modifier_func=None,
                           shuffle_csr_indices=False):
    """Create sparse array, using only rsp_indices to determine density"""
    if stype == 'row_sparse':
        density = 0.0
        if rsp_indices is not None:
            assert len(rsp_indices) <= shape[0]
    return create_sparse_array(shape, stype,
                               data_init=data_init,
                               rsp_indices=rsp_indices,
                               dtype=dtype,
                               modifier_func=modifier_func,
                               density=density,
                               shuffle_csr_indices=shuffle_csr_indices)

def rand_shape_2d(dim0=10, dim1=10):
    return rnd.randint(1, dim0 + 1), rnd.randint(1, dim1 + 1)


def rand_shape_3d(dim0=10, dim1=10, dim2=10):
    return rnd.randint(1, dim0 + 1), rnd.randint(1, dim1 + 1), rnd.randint(1, dim2 + 1)


def rand_shape_nd(num_dim, dim=10):
    return tuple(rnd.randint(1, dim+1, size=num_dim))


def rand_coord_2d(x_low, x_high, y_low, y_high):
    x = np.random.randint(x_low, x_high, dtype=np.int64)
    y = np.random.randint(y_low, y_high, dtype=np.int64)
    return x, y


def np_reduce(dat, axis, keepdims, numpy_reduce_func):
    """Compatible reduce for old version of NumPy.

    Parameters
    ----------
    dat : np.ndarray
        Same as NumPy.

    axis : None or int or list-like
        Same as NumPy.

    keepdims : bool
        Same as NumPy.

    numpy_reduce_func : function
        A NumPy reducing function like ``np.sum`` or ``np.max``.
    """
    if isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis) if axis is not None else range(len(dat.shape))
    ret = dat
    for i in reversed(sorted(axis)):
        ret = numpy_reduce_func(ret, axis=i)
    if keepdims:
        keepdims_shape = list(dat.shape)
        for i in axis:
            keepdims_shape[i] = 1
        ret = ret.reshape(tuple(keepdims_shape))
    return ret


def find_max_violation(a, b, rtol=None, atol=None):
    """Finds and returns the location of maximum violation."""
    rtol = get_rtol(rtol)
    atol = get_atol(atol)
    diff = np.abs(a-b)
    tol = atol + rtol*np.abs(b)
    violation = diff/(tol+1e-20)
    loc = np.argmax(violation)
    idx = np.unravel_index(loc, violation.shape)
    return idx, np.max(violation)


def same(a, b):
    """Test if two NumPy arrays are the same.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    """
    return np.array_equal(a, b)

def almost_equal(a, b, rtol=None, atol=None, equal_nan=False):
    """Test if two numpy arrays are almost equal."""
    # pylint: disable=unexpected-keyword-arg
    return np.allclose(a, b, rtol=get_rtol(rtol), atol=get_atol(atol), equal_nan=equal_nan)
    # pylint: enable=unexpected-keyword-arg

def assert_almost_equal(a, b, rtol=None, atol=None, names=('a', 'b'), equal_nan=False):
    """Test that two numpy arrays are almost equal. Raise exception message if not.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    threshold : None or float
        The checking threshold. Default threshold will be used if set to ``None``.
    """
    rtol = get_rtol(rtol)
    atol = get_atol(atol)
    if almost_equal(a, b, rtol, atol, equal_nan=equal_nan):
        return
    index, rel = find_max_violation(a, b, rtol, atol)
    np.set_printoptions(threshold=4, suppress=True)
    msg = npt.build_err_msg([a, b],
                            err_msg="Error %f exceeds tolerance rtol=%f, atol=%f. "
                                    " Location of maximum error:%s, a=%f, b=%f"
                            % (rel, rtol, atol, str(index), a[index], b[index]),
                            names=names)
    raise AssertionError(msg)

def assert_almost_equal_with_err(a, b, rtol=None, atol=None, etol=None, names=('a', 'b'), equal_nan=False):
    """Test that two numpy arrays are almost equal within given error rate. Raise exception message if not.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    threshold : None or float
        The checking threshold. Default threshold will be used if set to ``None``.
    etol : None or float
        The error rate threshold. If etol is float, return true if error_rate < etol even if
        any error is found.
    """
    rtol = get_rtol(rtol)
    atol = get_atol(atol)
    etol = get_etol(etol)
    if etol:
        equals = np.isclose(a, b, rtol=rtol, atol=atol)
        err = 1 - np.count_nonzero(equals) / equals.size
        if err > etol:
            #if True:
            index, rel = find_max_violation(a, b, rtol, atol)
            np.set_printoptions(threshold=4, suppress=True)
            msg = npt.build_err_msg([a, b],
                                    err_msg="Error %f exceeds tolerance rtol=%f, atol=%f, etol=%f."
                                            " Error_rate=%f. Location of maximum error:%s, a=%f, b=%f"
                                    % (rel, rtol, atol, etol, err, str(index), a[index], b[index]),
                                    names=names)
            raise AssertionError(msg)

        if almost_equal(a, b, rtol, atol, equal_nan=equal_nan):
            return
    else:
        if almost_equal(a, b, rtol, atol, equal_nan=equal_nan):
            return
        index, rel = find_max_violation(a, b, rtol, atol)
        np.set_printoptions(threshold=4, suppress=True)
        msg = npt.build_err_msg([a, b],
                                err_msg="Error %f exceeds tolerance rtol=%f, atol=%f. "
                                        " Location of maximum error:%s, a=%f, b=%f"
                                % (rel, rtol, atol, str(index), a[index], b[index]),
                                names=names)
        raise AssertionError(msg)


def almost_equal_ignore_nan(a, b, rtol=None, atol=None):
    """Test that two NumPy arrays are almost equal (ignoring NaN in either array).
    Combines a relative and absolute measure of approximate eqality.
    If either the relative or absolute check passes, the arrays are considered equal.
    Including an absolute check resolves issues with the relative check where all
    array values are close to zero.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    rtol : None or float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float
        The absolute threshold. Default threshold will be used if set to ``None``.
    """
    a = np.copy(a)
    b = np.copy(b)
    nan_mask = np.logical_or(np.isnan(a), np.isnan(b))
    a[nan_mask] = 0
    b[nan_mask] = 0

    return almost_equal(a, b, rtol, atol)


def assert_almost_equal_ignore_nan(a, b, rtol=None, atol=None, names=('a', 'b')):
    """Test that two NumPy arrays are almost equal (ignoring NaN in either array).
    Combines a relative and absolute measure of approximate eqality.
    If either the relative or absolute check passes, the arrays are considered equal.
    Including an absolute check resolves issues with the relative check where all
    array values are close to zero.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    rtol : None or float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float
        The absolute threshold. Default threshold will be used if set to ``None``.
    """
    a = np.copy(a)
    b = np.copy(b)
    nan_mask = np.logical_or(np.isnan(a), np.isnan(b))
    a[nan_mask] = 0
    b[nan_mask] = 0

    assert_almost_equal(a, b, rtol, atol, names)

def assert_exception(f, exception_type, *args, **kwargs):
    """Test that function f will throw an exception of type given by `exception_type`"""
    try:
        f(*args, **kwargs)
        assert(False)
    except exception_type:
        return

def retry(n):
    """Retry n times before failing for stochastic test cases."""
    assert n > 0
    def decorate(f):
        """Decorate a test case."""
        def wrapper(*args, **kwargs):
            """Wrapper for tests function."""
            for _ in range(n):
                try:
                    f(*args, **kwargs)
                except AssertionError as e:
                    err = e
                    continue
                return
            raise err
        return wrapper
    return decorate


def simple_forward(sym, ctx=None, is_train=False, **inputs):
    """A simple forward function for a symbol.

    Primarily used in doctest to test the functionality of a symbol.
    Takes NumPy arrays as inputs and outputs are also converted to NumPy arrays.

    Parameters
    ----------
    ctx : Context
        If ``None``, will take the default context.
    inputs : keyword arguments
        Mapping each input name to a NumPy array.

    Returns
    -------
    The result as a numpy array. Multiple results will
    be returned as a list of NumPy arrays.
    """
    ctx = ctx or default_context()
    inputs = {k: array(v) for k, v in inputs.items()}
    exe = sym.bind(ctx, args=inputs)
    exe.forward(is_train=is_train)
    outputs = [x.asnumpy() for x in exe.outputs]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def _parse_location(sym, location, ctx, dtype=default_dtype()):
    """Parses the given location to a dictionary.

    Arguments of the provided op `sym` are used as dictionary keys
    and elements of `location` are used as values.

    Parameters
    ----------
    sym : Symbol
        Symbol containing op
    location : list or tuple or dict
        Argument values location

        - if type is list or tuple of `np.ndarray`
            inner elements are arrays correspoding to
            ``sym.list_arguments()``.
        - if type is dict of str -> `np.ndarray`
            maps the name of arguments to the corresponding `np.ndarray`.
        *In either case, value of all the arguments must be provided.*
    ctx : Context
        Device context.
    dtype: "asnumpy" or np.float16 or np.float32 or np.float64
        If dtype is "asnumpy" then the mx.nd.array created will have the same
        type as th numpy array from which it is copied.
        Otherwise, dtype is the explicit datatype for all mx.nd.array objects
        created in this function.

    Returns
    -------
    dict
        Dictionary with `sym` arguments as keys and `location` elements as
        values.

    Examples
    -------
    >>> a = mx.symbol.Variable('a')
    >>> b = mx.symbol.Variable('b')
    >>> l1 = np.ndarray([2,3])
    >>> l2 = np.ndarray([3,4])
    >>> _parse_location(a * b, [l1, l2], None)
    {'a': <NDArray 2x3 @cpu(0)>, 'b': <NDArray 3x4 @cpu(0)>}
    >>> _parse_location(a * b, {'a': l1, 'b': l2}, None)
    {'a': <NDArray 2x3 @cpu(0)>, 'b': <NDArray 3x4 @cpu(0)>}
    >>> _parse_location(a * b, {'a': l1}, None)
    ValueError: Symbol arguments and keys of the given location do not match.
    """
    assert isinstance(location, (dict, list, tuple))
    assert dtype == "asnumpy" or dtype in (np.float16, np.float32, np.float64)
    if isinstance(location, dict):
        if set(location.keys()) != set(sym.list_arguments()):
            raise ValueError("Symbol arguments and keys of the given location do not match."
                             "symbol args:%s, location.keys():%s"
                             % (str(set(sym.list_arguments())), str(set(location.keys()))))
    else:
        location = {k: v for k, v in zip(sym.list_arguments(), location)}
    location = {k: mx.nd.array(v, ctx=ctx, dtype=v.dtype if dtype == "asnumpy" else dtype) \
               if isinstance(v, np.ndarray) else v for k, v in location.items()}
    return location


def _parse_aux_states(sym, aux_states, ctx, dtype=default_dtype()):
    """Parses the given auxiliary states to a dictionary.

    Auxiliary states of the provided op `sym` are used as dictionary
    keys and elements of `aux_states` are used as values.

    Parameters
    ----------
    sym : Symbol
        Symbol containing op
    aux_states : None or list or dict
        Aux states

        - if type is list or tuple of `np.ndarray`
            inner elements are arrays correspoding to
            ``sym.list_auxiliary_states()``.
        - if type is dict of str -> `np.ndarray`
            maps the name of arguments to the corresponding `np.ndarray`.
        *In either case, all aux states of `sym` must be provided.*
    ctx : Context
        Device context.
    dtype: "asnumpy" or np.float16 or np.float32 or np.float64
        If dtype is "asnumpy" then the mx.nd.array created will have the same
        type as th numpy array from which it is copied.
        Otherwise, dtype is the explicit datatype for all mx.nd.array objects
        created in this function.

    Returns
    -------
    dict
        Dictionary with `sym` aux states as keys and `aux_states` elements
        as values.

    Examples
    -------
    >>> data = mx.symbol.Variable('data')
    >>> weight = mx.sym.Variable(name='fc1_weight')
    >>> fc1 = mx.symbol.FullyConnected(data = data, weight=weight, name='fc1', num_hidden=128)
    >>> fc2 = mx.symbol.BatchNorm(fc1, name='batchnorm0')
    >>> mean_states = np.ones(3)
    >>> var_states = np.ones(3)
    >>> _parse_aux_states(fc2, [mean_states, var_states], None)
    {'batchnorm0_moving_var': <NDArray 3 @cpu(0)>, 'batchnorm0_moving_mean': <NDArray 3 @cpu(0)>}
    >>> _parse_aux_states(fc2, {'batchnorm0_moving_var': mean_states,
    ...                         'batchnorm0_moving_mean': var_states}, None)
    {'batchnorm0_moving_var': <NDArray 3 @cpu(0)>, 'batchnorm0_moving_mean': <NDArray 3 @cpu(0)>}
    >>> _parse_aux_states(fc2, {'batchnorm0_moving_var': mean_states}, None)
    ValueError: Symbol aux_states names and given aux_states do not match.
    """
    assert dtype == "asnumpy" or dtype in (np.float16, np.float32, np.float64)
    if aux_states is not None:
        if isinstance(aux_states, dict):
            if set(aux_states.keys()) != set(sym.list_auxiliary_states()):
                raise ValueError("Symbol aux_states names and given aux_states do not match."
                                 "symbol aux_names:%s, aux_states.keys:%s"
                                 % (str(set(sym.list_auxiliary_states())),
                                    str(set(aux_states.keys()))))
        elif isinstance(aux_states, (list, tuple)):
            aux_names = sym.list_auxiliary_states()
            aux_states = {k:v for k, v in zip(aux_names, aux_states)}
        aux_states = {k: mx.nd.array(v, ctx=ctx, dtype=v.dtype if dtype == "asnumpy" else dtype) \
                      for k, v in aux_states.items()}
    return aux_states


def numeric_grad(executor, location, aux_states=None, eps=1e-4,
                 use_forward_train=True, dtype=default_dtype()):
    """Calculates a numeric gradient via finite difference method.

    Class based on Theano's `theano.gradient.numeric_grad` [1]

    Parameters
    ----------
    executor : Executor
        Executor that computes the forward pass.
    location : list of numpy.ndarray or dict of str to numpy.ndarray
        Argument values used as location to compute gradient
        Maps the name of arguments to the corresponding numpy.ndarray.
        Value of all the arguments must be provided.
    aux_states : None or list of numpy.ndarray or dict of str to numpy.ndarray, optional
        Auxiliary states values used as location to compute gradient
        Maps the name of aux_states to the corresponding numpy.ndarray.
        Value of all the auxiliary arguments must be provided.
    eps : float, optional
        Epsilon for the finite-difference method.
    use_forward_train : bool, optional
        Whether to use `is_train=True` in testing.
    dtype: np.float16 or np.float32 or np.float64
        Datatype for mx.nd.array.

    References
    ---------
    ..[1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """
    def as_stype(var, stype, dtype):
        return mx.nd.cast_storage(mx.nd.array(var, dtype=dtype), stype=stype)

    assert dtype in (np.float16, np.float32, np.float64)
    approx_grads = {k: np.zeros(v.shape, dtype=dtype)
                    for k, v in location.items()}
    for k, v in location.items():
        stype = executor.arg_dict[k].stype
        if stype == 'default':
            executor.arg_dict[k][:] = as_stype(v, stype, dtype=dtype)
    for k in location:
        location[k] = np.asarray(location[k], order='C')
    for k, v in location.items():
        if v.dtype.kind != 'f':
            continue
        stype = executor.arg_dict[k].stype
        old_value = v.copy()
        for i in range(np.prod(v.shape)):
            # inplace update
            v.ravel()[i] += eps/2.0
            executor.arg_dict[k][:] = as_stype(v, stype, dtype=dtype)
            if aux_states is not None:
                for key, val in aux_states.items():
                    executor.aux_dict[key][:] = val
            executor.forward(is_train=use_forward_train)
            f_peps = executor.outputs[0].asnumpy()

            v.ravel()[i] -= eps
            executor.arg_dict[k][:] = as_stype(v, stype, dtype=dtype)
            if aux_states is not None:
                for key, val in aux_states.items():
                    adstype = executor.aux_dict[key].stype
                    executor.aux_dict[key][:] = as_stype(val, adstype, dtype=dtype)
            executor.forward(is_train=use_forward_train)
            f_neps = executor.outputs[0].asnumpy()

            approx_grad = (f_peps - f_neps).sum() / eps
            approx_grads[k].ravel()[i] = approx_grad
            v.ravel()[i] = old_value.ravel()[i]
        # copy back the original value
        executor.arg_dict[k][:] = as_stype(old_value, stype, dtype=dtype)

    return approx_grads


def check_numeric_gradient(sym, location, aux_states=None, numeric_eps=1e-3, rtol=1e-2,
                           atol=None, grad_nodes=None, use_forward_train=True, ctx=None,
                           grad_stype_dict=None, dtype=default_dtype()):
    """Verify an operation by checking backward pass via finite difference method.

    Based on Theano's `theano.gradient.verify_grad` [1]

    Parameters
    ----------
    sym : Symbol
        Symbol containing op to test
    location : list or tuple or dict
        Argument values used as location to compute gradient

        - if type is list of numpy.ndarray, \
            inner elements should have the same order as mxnet.sym.list_arguments().

        - if type is dict of str -> numpy.ndarray, \
            maps the name of arguments to the corresponding numpy.ndarray.

        *In either case, value of all the arguments must be provided.*
    aux_states : list or tuple or dict, optional
        The auxiliary states required when generating the executor for the symbol.
    numeric_eps : float, optional
        Delta for the finite difference method that approximates the gradient.
    check_eps : float, optional
        relative error eps used when comparing numeric grad to symbolic grad.
    grad_nodes : None or list or tuple or dict, optional
        Names of the nodes to check gradient on
    use_forward_train : bool
        Whether to use is_train=True when computing the finite-difference.
    ctx : Context, optional
        Check the gradient computation on the specified device.
    grad_stype_dict : dict of str->str, optional
        Storage type dictionary for gradient ndarrays.
    dtype: np.float16 or np.float32 or np.float64
        Datatype for mx.nd.array.

    References
    ---------
    [1] https://github.com/Theano/Theano/blob/master/theano/gradient.py
    """
    assert dtype in (np.float16, np.float32, np.float64)
    # cannot use finite differences with small eps without high precision
    if dtype in (np.float32, np.float16):
        assert numeric_eps >= 1e-5
    if ctx is None:
        ctx = default_context()

    def random_projection(shape):
        """Get a random weight matrix with not too small elements

        Parameters
        ----------
        shape : list or tuple
        """
        # random_projection should not have elements too small,
        # otherwise too much precision is lost in numerical gradient
        plain = np.random.rand(*shape) + 0.1
        return plain

    location = _parse_location(sym=sym, location=location, ctx=ctx, dtype=dtype)
    location_npy = {k:v.asnumpy() for k, v in location.items()}
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx,
                                   dtype=dtype)
    if aux_states is not None:
        aux_states_npy = {k: v.asnumpy() for k, v in aux_states.items()}
    else:
        aux_states_npy = None
    if grad_nodes is None:
        grad_nodes = sym.list_arguments()
        grad_req = {k: 'write' for k in grad_nodes}
    elif isinstance(grad_nodes, (list, tuple)):
        grad_nodes = list(grad_nodes)
        grad_req = {k: 'write' for k in grad_nodes}
    elif isinstance(grad_nodes, dict):
        grad_req = grad_nodes.copy()
        grad_nodes = grad_nodes.keys()
    else:
        raise ValueError

    input_shape = {k: v.shape for k, v in location.items()}
    _, out_shape, _ = sym.infer_shape(**input_shape)
    proj = mx.sym.Variable("__random_proj")
    out = sym * proj
    out = mx.sym.make_loss(out)

    location = dict(list(location.items()) +
                    [("__random_proj", mx.nd.array(random_projection(out_shape[0]),
                                                   ctx=ctx, dtype=dtype))])
    args_grad_npy = dict([(k, np.random.normal(0, 0.01, size=location[k].shape))
                          for k in grad_nodes]
                         + [("__random_proj", np.random.normal(0, 0.01, size=out_shape[0]))])

    args_grad = {k: mx.nd.array(v, ctx=ctx, dtype=dtype) for k, v in args_grad_npy.items()}
    if grad_stype_dict is not None:
        assert isinstance(grad_stype_dict, dict), "grad_stype_dict must be a dict"
        for k, v in grad_stype_dict.items():
            if k in args_grad and v in _STORAGE_TYPE_STR_TO_ID and v != 'default':
                # create an uninitialized sparse ndarray for executor
                # if the symbolic grad is expected to be zero, it should not be initialized at all
                args_grad[k] = mx.nd.zeros(args_grad[k].shape, args_grad[k].context,
                                           args_grad[k].dtype, v)

    executor = out.bind(ctx, grad_req=grad_req,
                        args=location, args_grad=args_grad, aux_states=aux_states)

    inps = executor.arg_arrays
    if len(inps) != len(location):
        raise ValueError("Executor arg_arrays and and location len do not match."
                         "Got %d inputs and %d locations"%(len(inps), len(location)))
    assert len(executor.outputs) == 1

    executor.forward(is_train=True)
    executor.backward()
    symbolic_grads = {k:executor.grad_dict[k].asnumpy() for k in grad_nodes}

    numeric_gradients = numeric_grad(
        executor, location_npy, aux_states_npy,
        eps=numeric_eps, use_forward_train=use_forward_train, dtype=dtype)

    for name in grad_nodes:
        fd_grad = numeric_gradients[name]
        orig_grad = args_grad_npy[name]
        sym_grad = symbolic_grads[name]
        if grad_req[name] == 'write':
            assert_almost_equal(fd_grad, sym_grad, rtol, atol,
                                ("NUMERICAL_%s"%name, "BACKWARD_%s"%name))
        elif grad_req[name] == 'add':
            assert_almost_equal(fd_grad, sym_grad - orig_grad, rtol, atol,
                                ("NUMERICAL_%s"%name, "BACKWARD_%s"%name))
        elif grad_req[name] == 'null':
            assert_almost_equal(orig_grad, sym_grad, rtol, atol,
                                ("NUMERICAL_%s"%name, "BACKWARD_%s"%name))
        else:
            raise ValueError("Invalid grad_req %s for argument %s"%(grad_req[name], name))


def check_symbolic_forward(sym, location, expected, rtol=1E-4, atol=None,
                           aux_states=None, ctx=None, equal_nan=False,
                           dtype=default_dtype()):
    """Compares a symbol's forward results with the expected ones.
    Prints error messages if the forward results are not the same as the expected ones.

    Parameters
    ---------
    sym : Symbol
        output symbol
    location : list of np.ndarray or dict of str to np.ndarray
        The evaluation point

        - if type is list of np.ndarray
            Contains all the numpy arrays corresponding to `sym.list_arguments()`.
        - if type is dict of str to np.ndarray
            Contains the mapping between argument names and their values.
    expected : list of np.ndarray or dict of str to np.ndarray
        The expected output value

        - if type is list of np.ndarray
            Contains arrays corresponding to exe.outputs.
        - if type is dict of str to np.ndarray
            Contains mapping between sym.list_output() and exe.outputs.
    check_eps : float, optional
        Relative error to check to.
    aux_states : list of np.ndarray of dict, optional
        - if type is list of np.ndarray
            Contains all the NumPy arrays corresponding to sym.list_auxiliary_states
        - if type is dict of str to np.ndarray
            Contains the mapping between names of auxiliary states and their values.
    ctx : Context, optional
        running context
    dtype: "asnumpy" or np.float16 or np.float32 or np.float64
        If dtype is "asnumpy" then the mx.nd.array created will have the same
        type as th numpy array from which it is copied.
        Otherwise, dtype is the explicit datatype for all mx.nd.array objects
        created in this function.

    equal_nan: Boolean
        if True, `nan` is a valid value for checking equivalency (ie `nan` == `nan`)

    Example
    -------
    >>> shape = (2, 2)
    >>> lhs = mx.symbol.Variable('lhs')
    >>> rhs = mx.symbol.Variable('rhs')
    >>> sym_dot = mx.symbol.dot(lhs, rhs)
    >>> mat1 = np.array([[1, 2], [3, 4]])
    >>> mat2 = np.array([[5, 6], [7, 8]])
    >>> ret_expected = np.array([[19, 22], [43, 50]])
    >>> check_symbolic_forward(sym_dot, [mat1, mat2], [ret_expected])
    """
    assert dtype == "asnumpy" or dtype in (np.float16, np.float32, np.float64)
    if ctx is None:
        ctx = default_context()

    location = _parse_location(sym=sym, location=location, ctx=ctx, dtype=dtype)
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx,
                                   dtype=dtype)
    if isinstance(expected, dict):
        expected = [expected[k] for k in sym.list_outputs()]
    args_grad_data = {k:mx.nd.empty(v.shape, ctx=ctx, dtype=v.dtype if dtype == "asnumpy" else dtype) \
                      for k, v in location.items()}

    executor = sym.bind(ctx=ctx, args=location, args_grad=args_grad_data, aux_states=aux_states)
    for g in executor.grad_arrays:
        g[:] = 0

    executor.forward(is_train=False)

    outputs = [x.asnumpy() for x in executor.outputs]
    for output_name, expect, output in zip(sym.list_outputs(), expected, outputs):
        assert_almost_equal(expect, output, rtol, atol,
                            ("EXPECTED_%s"%output_name, "FORWARD_%s"%output_name),
                            equal_nan=equal_nan)
    return executor.outputs

def check_symbolic_backward(sym, location, out_grads, expected, rtol=1e-5, atol=None,
                            aux_states=None, grad_req='write', ctx=None, grad_stypes=None,
                            equal_nan=False, dtype=default_dtype()):
    """Compares a symbol's backward results with the expected ones.
    Prints error messages if the backward results are not the same as the expected results.

    Parameters
    ---------
    sym : Symbol
        output symbol
    location : list of np.ndarray or dict of str to np.ndarray
        The evaluation point

        - if type is list of np.ndarray
            Contains all the NumPy arrays corresponding to ``mx.sym.list_arguments``.
        - if type is dict of str to np.ndarray
            Contains the mapping between argument names and their values.
    out_grads : None or list of np.ndarray or dict of str to np.ndarray
        NumPys arrays corresponding to sym.outputs for incomming gradient.

        - if type is list of np.ndarray
            Contains arrays corresponding to ``exe.outputs``.
        - if type is dict of str to np.ndarray
            contains mapping between mxnet.sym.list_output() and Executor.outputs
    expected : list of np.ndarray or dict of str to np.ndarray
        expected gradient values

        - if type is list of np.ndarray
            Contains arrays corresponding to exe.grad_arrays
        - if type is dict of str to np.ndarray
            Contains mapping between ``sym.list_arguments()`` and exe.outputs.
    check_eps: float, optional
        Relative error to check to.
    aux_states : list of np.ndarray or dict of str to np.ndarray
    grad_req : str or list of str or dict of str to str, optional
        Gradient requirements. 'write', 'add' or 'null'.
    ctx : Context, optional
        Running context.
    grad_stypes: dict of str->str
        dictionary of mapping argument name to stype for the gradient
    equal_nan: Boolean
        if True, `nan` is a valid value for checking equivalency (ie `nan` == `nan`)
    dtype: np.float16 or np.float32 or np.float64
        Datatype for mx.nd.array.

    Example
    -------
    >>> lhs = mx.symbol.Variable('lhs')
    >>> rhs = mx.symbol.Variable('rhs')
    >>> sym_add = mx.symbol.elemwise_add(lhs, rhs)
    >>> mat1 = np.array([[1, 2], [3, 4]])
    >>> mat2 = np.array([[5, 6], [7, 8]])
    >>> grad1 = mx.nd.zeros(shape)
    >>> grad2 = mx.nd.zeros(shape)
    >>> exec_add = sym_add.bind(default_context(), args={'lhs': mat1, 'rhs': mat2},
    ... args_grad={'lhs': grad1, 'rhs': grad2}, grad_req={'lhs': 'write', 'rhs': 'write'})
    >>> exec_add.forward(is_train=True)
    >>> ograd = mx.nd.ones(shape)
    >>> grad_expected = ograd.copy().asnumpy()
    >>> check_symbolic_backward(sym_add, [mat1, mat2], [ograd], [grad_expected, grad_expected])
    """
    assert dtype in (np.float16, np.float32, np.float64)
    if ctx is None:
        ctx = default_context()

    location = _parse_location(sym=sym, location=location, ctx=ctx, dtype=dtype)
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx,
                                   dtype=dtype)
    if isinstance(expected, (list, tuple)):
        expected = {k:v for k, v in zip(sym.list_arguments(), expected)}

    args_grad_npy = {k:np.random.normal(size=v.shape) for k, v in expected.items()}
    args_grad_data = {}
    for k, v in args_grad_npy.items():
        nd = mx.nd.array(v, ctx=ctx, dtype=dtype)
        if grad_stypes is not None and k in grad_stypes:
            stype = grad_stypes[k]
            if stype is not None and stype != 'default':
                out = create_sparse_array(v.shape, stype, density=0.0)
            else:
                out = nd
            args_grad_data[k] = out
        else:
            args_grad_data[k] = nd

    if isinstance(grad_req, str):
        grad_req = {k:grad_req for k in sym.list_arguments()}
    elif isinstance(grad_req, (list, tuple)):
        grad_req = {k:v for k, v in zip(sym.list_arguments(), grad_req)}

    executor = sym.bind(ctx=ctx, args=location, args_grad=args_grad_data,
                        aux_states=aux_states, grad_req=grad_req)
    executor.forward(is_train=True)

    if isinstance(out_grads, (tuple, list)):
        outg = list()
        for arr in out_grads:
            if isinstance(arr, np.ndarray):
                outg.append(mx.nd.array(arr, ctx=ctx, dtype=dtype))
            else:
                outg.append(arr)
        out_grads = outg
    elif isinstance(out_grads, dict):
        outg = dict()
        for k, v in out_grads.items():
            if isinstance(v, np.ndarray):
                outg[k] = mx.nd.array(v, ctx=ctx, dtype=dtype)
            else:
                outg[k] = v
        out_grads = outg
    else:
        assert out_grads is None

    executor.backward(out_grads)

    grads = {k: v.asnumpy() for k, v in args_grad_data.items()}

    for name in expected:
        if grad_req[name] == 'write':
            assert_almost_equal(expected[name], grads[name], rtol, atol,
                                ("EXPECTED_%s"%name, "BACKWARD_%s"%name),
                                equal_nan=equal_nan)
        elif grad_req[name] == 'add':
            assert_almost_equal(expected[name], grads[name] - args_grad_npy[name],
                                rtol, atol, ("EXPECTED_%s"%name, "BACKWARD_%s"%name),
                                equal_nan=equal_nan)
        elif grad_req[name] == 'null':
            assert_almost_equal(args_grad_npy[name], grads[name],
                                rtol, atol, ("EXPECTED_%s"%name, "BACKWARD_%s"%name),
                                equal_nan=equal_nan)
        else:
            raise ValueError("Invalid grad_req %s for argument %s"%(grad_req[name], name))
    return args_grad_data

def check_speed(sym, location=None, ctx=None, N=20, grad_req=None, typ="whole",
                **kwargs):
    """Check the running speed of a symbol.

    Parameters
    ----------
    sym : Symbol
        Symbol to run the speed test.
    location : none or dict of str to np.ndarray
        Location to evaluate the inner executor.
    ctx : Context
        Running context.
    N : int, optional
        Repeat times.
    grad_req : None or str or list of str or dict of str to str, optional
        Gradient requirements.
    typ : str, optional
        "whole" or "forward"

        - "whole"
            Test the forward_backward speed.
        - "forward"
            Only test the forward speed.
    """
    if ctx is None:
        ctx = default_context()

    if grad_req is None:
        grad_req = 'write'
    if location is None:
        exe = sym.simple_bind(grad_req=grad_req, ctx=ctx, **kwargs)
        location = {k: np.random.normal(size=arr.shape, scale=1.0) for k, arr in
                    exe.arg_dict.items()}
    else:
        assert isinstance(location, dict), "Expect dict, get \"location\"=%s" %str(location)
        exe = sym.simple_bind(grad_req=grad_req, ctx=ctx,
                              **{k: v.shape for k, v in location.items()})

    for name, iarr in location.items():
        exe.arg_dict[name][:] = iarr.astype(exe.arg_dict[name].dtype)

    if typ == "whole":
        # Warm up
        exe.forward(is_train=True)
        exe.backward(out_grads=exe.outputs)
        for output in exe.outputs:
            output.wait_to_read()
        # Test forward + backward
        tic = time.time()
        for _ in range(N):
            exe.forward(is_train=True)
            exe.backward(out_grads=exe.outputs)
        mx.nd.waitall()
        toc = time.time()
        forward_backward_time = (toc - tic) * 1.0 / N
        return forward_backward_time
    elif typ == "forward":
        # Warm up
        exe.forward(is_train=False)
        for output in exe.outputs:
            output.wait_to_read()

        # Test forward only
        tic = time.time()
        for _ in range(N):
            exe.forward(is_train=False)
        mx.nd.waitall()
        toc = time.time()
        forward_time = (toc - tic) * 1.0 / N
        return forward_time
    else:
        raise ValueError('typ can only be "whole" or "forward".')

def check_consistency(sym, ctx_list, scale=1.0, grad_req='write',
                      arg_params=None, aux_params=None, tol=None,
                      raise_on_err=True, ground_truth=None, equal_nan=False,
                      use_uniform=False, rand_type=np.float64):
    """Check symbol gives the same output for different running context

    Parameters
    ----------
    sym : Symbol or list of Symbols
        Symbol(s) to run the consistency test.
    ctx_list : list
        Running context. See example for more detail.
    scale : float, optional
        Standard deviation of the inner normal distribution. Used in initialization.
    grad_req : str or list of str or dict of str to str
        Gradient requirement.
    use_unifrom: bool
        Optional, When flag set to true,
        random input data generated follows uniform distribution,
        not normal distribution
    rand_type: np.dtype
        casts the randomly generated data to this type
        Optional, when input data is passed via arg_params,
        defaults to np.float64 (numpy float default)

    Examples
    --------
    >>> # create the symbol
    >>> sym = mx.sym.Convolution(num_filter=3, kernel=(3,3), name='conv')
    >>> # initialize the running context
    >>> ctx_list =\
[{'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},\
 {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}},\
 {'ctx': mx.gpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float16}},\
 {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float64}},\
 {'ctx': mx.cpu(0), 'conv_data': (2, 2, 10, 10), 'type_dict': {'conv_data': np.float32}}]
    >>> check_consistency(sym, ctx_list)
    >>> sym = mx.sym.Concat(name='concat', num_args=2)
    >>> ctx_list = \
[{'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float64, 'concat_arg1': np.float64}},\
 {'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float32, 'concat_arg1': np.float32}},\
 {'ctx': mx.gpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float16, 'concat_arg1': np.float16}},\
 {'ctx': mx.cpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float64, 'concat_arg1': np.float64}},\
 {'ctx': mx.cpu(0), 'concat_arg1': (2, 10), 'concat_arg0': (2, 10),\
  'type_dict': {'concat_arg0': np.float32, 'concat_arg1': np.float32}}]
    >>> check_consistency(sym, ctx_list)
    """
    if tol is None:
        tol = {np.dtype(np.float16): 1e-1,
               np.dtype(np.float32): 1e-3,
               np.dtype(np.float64): 1e-5,
               np.dtype(np.uint8): 0,
               np.dtype(np.int32): 0,
               np.dtype(np.int64): 0}
    elif isinstance(tol, numbers.Number):
        tol = {np.dtype(np.float16): tol,
               np.dtype(np.float32): tol,
               np.dtype(np.float64): tol,
               np.dtype(np.uint8): tol,
               np.dtype(np.int32): tol,
               np.dtype(np.int64): tol}

    assert len(ctx_list) > 1
    if isinstance(sym, Symbol):
        sym = [sym]*len(ctx_list)
    else:
        assert len(sym) == len(ctx_list)

    output_names = sym[0].list_outputs()
    arg_names = sym[0].list_arguments()
    exe_list = []
    for s, ctx in zip(sym, ctx_list):
        assert s.list_arguments() == arg_names
        assert s.list_outputs() == output_names
        exe_list.append(s.simple_bind(grad_req=grad_req, **ctx))

    arg_params = {} if arg_params is None else arg_params
    aux_params = {} if aux_params is None else aux_params
    for n, arr in exe_list[0].arg_dict.items():
        if n not in arg_params:
            if use_uniform:
                arg_params[n] = np.random.uniform(low=-0.92, high=0.92,
                                                  size=arr.shape).astype(rand_type)
            else:
                arg_params[n] = np.random.normal(size=arr.shape,
                                                 scale=scale).astype(rand_type)
    for n, arr in exe_list[0].aux_dict.items():
        if n not in aux_params:
            aux_params[n] = 0
    for exe in exe_list:
        for name, arr in exe.arg_dict.items():
            arr[:] = arg_params[name]
        for name, arr in exe.aux_dict.items():
            arr[:] = aux_params[name]
        # We need to initialize the gradient arrays if it's add.
        if (grad_req == "add"):
            for arr in exe.grad_arrays:
                arr[:] = np.zeros(arr.shape, dtype=arr.dtype)

    dtypes = [np.dtype(exe.outputs[0].dtype) for exe in exe_list]
    max_idx = np.argmax(dtypes)
    gt = ground_truth
    if gt is None:
        gt = exe_list[max_idx].output_dict.copy()
        if grad_req != 'null':
            gt.update(exe_list[max_idx].grad_dict)

    # test
    for exe in exe_list:
        exe.forward(is_train=False)

    for i, exe in enumerate(exe_list):
        if i == max_idx:
            continue
        for name, arr in zip(output_names, exe.outputs):
            gtarr = gt[name].astype(dtypes[i]).asnumpy()
            arr = arr.asnumpy()
            try:
                assert_almost_equal(arr, gtarr, rtol=tol[dtypes[i]], atol=tol[dtypes[i]],
                                    equal_nan=equal_nan)
            except AssertionError as e:
                print('Predict Err: ctx %d vs ctx %d at %s'%(i, max_idx, name))
                traceback.print_exc()
                if raise_on_err: # pylint: disable=no-else-raise
                    raise e
                else:
                    print(str(e))

    # train
    if grad_req != 'null':
        for exe in exe_list:
            exe.forward(is_train=True)
            exe.backward(exe.outputs)

        for i, exe in enumerate(exe_list):
            if i == max_idx:
                continue
            curr = zip(output_names + arg_names, exe.outputs + exe.grad_arrays)
            for name, arr in curr:
                if gt[name] is None:
                    assert arr is None
                    continue
                gtarr = gt[name].astype(dtypes[i]).asnumpy()
                arr = arr.asnumpy()
                try:
                    assert_almost_equal(arr, gtarr, rtol=tol[dtypes[i]], atol=tol[dtypes[i]],
                                        equal_nan=equal_nan)
                except AssertionError as e:
                    print('Train Err: ctx %d vs ctx %d at %s'%(i, max_idx, name))
                    traceback.print_exc()
                    if raise_on_err: # pylint: disable=no-else-raise
                        raise e
                    else:
                        print(str(e))

    return gt

def list_gpus():
    """Return a list of GPUs

    Returns
    -------
    list of int:
        If there are n GPUs, then return a list [0,1,...,n-1]. Otherwise returns
        [].
    """
    return range(mx.util.get_gpu_count())

def download(url, fname=None, dirname=None, overwrite=False, retries=5):
    """Download an given URL

    Parameters
    ----------

    url : str
        URL to download
    fname : str, optional
        filename of the downloaded file. If None, then will guess a filename
        from url.
    dirname : str, optional
        output directory name. If None, then guess from fname or use the current
        directory
    overwrite : bool, optional
        Default is false, which means skipping download if the local file
        exists. If true, then download the url to overwrite the local file if
        exists.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes

    Returns
    -------
    str
        The filename of the downloaded file
    """

    assert retries >= 0, "Number of retries should be at least 0"

    if fname is None:
        fname = url.split('/')[-1]

    if dirname is None:
        dirname = os.path.dirname(fname)
    else:
        fname = os.path.join(dirname, fname)
    if dirname != "":
        if not os.path.exists(dirname):
            try:
                logging.info('create directory %s', dirname)
                os.makedirs(dirname)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise OSError('failed to create ' + dirname)

    if not overwrite and os.path.exists(fname):
        logging.info("%s exists, skipping download", fname)
        return fname

    while retries+1 > 0:
        # Disable pyling too broad Exception
        # pylint: disable=W0703
        try:
            r = requests.get(url, stream=True)
            assert r.status_code == 200, "failed to open %s" % url
            with open(fname, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                break
        except Exception as e:
            retries -= 1
            if retries <= 0: # pylint: disable=no-else-raise
                raise e
            else:
                print("download failed, retrying, {} attempt{} left"
                      .format(retries, 's' if retries > 1 else ''))
    logging.info("downloaded %s into %s successfully", url, fname)
    return fname

def get_mnist():
    """Download and load the MNIST dataset

    Returns
    -------
    dict
        A dict containing the data
    """
    def read_data(label_url, image_url):
        with gzip.open(mx.test_utils.download(label_url)) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.frombuffer(flbl.read(), dtype=np.int8)
        with gzip.open(mx.test_utils.download(image_url), 'rb') as fimg:
            _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
            image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32)/255
        return (label, image)

    # changed to mxnet.io for more stable hosting
    # path = 'http://yann.lecun.com/exdb/mnist/'
    path = 'http://data.mxnet.io/data/mnist/'
    (train_lbl, train_img) = read_data(
        path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
    (test_lbl, test_img) = read_data(
        path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')
    return {'train_data':train_img, 'train_label':train_lbl,
            'test_data':test_img, 'test_label':test_lbl}

def get_mnist_pkl():
    """Downloads MNIST dataset as a pkl.gz into a directory in the current directory
    with the name `data`
    """
    if not os.path.isdir("data"):
        os.makedirs('data')
    if not os.path.exists('data/mnist.pkl.gz'):
        download('http://deeplearning.net/data/mnist/mnist.pkl.gz',
                 dirname='data')

def get_mnist_ubyte():
    """Downloads ubyte version of the MNIST dataset into a directory in the current directory
    with the name `data` and extracts all files in the zip archive to this directory.
    """
    if not os.path.isdir("data"):
        os.makedirs('data')
    if (not os.path.exists('data/train-images-idx3-ubyte')) or \
            (not os.path.exists('data/train-labels-idx1-ubyte')) or \
            (not os.path.exists('data/t10k-images-idx3-ubyte')) or \
            (not os.path.exists('data/t10k-labels-idx1-ubyte')):
        zip_file_path = download('http://data.mxnet.io/mxnet/data/mnist.zip',
                                 dirname='data')
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall('data')

def get_cifar10():
    """Downloads CIFAR10 dataset into a directory in the current directory with the name `data`,
    and then extracts all files into the directory `data/cifar`.
    """
    if not os.path.isdir("data"):
        os.makedirs('data')
    if (not os.path.exists('data/cifar/train.rec')) or \
            (not os.path.exists('data/cifar/test.rec')) or \
            (not os.path.exists('data/cifar/train.lst')) or \
            (not os.path.exists('data/cifar/test.lst')):
        zip_file_path = download('http://data.mxnet.io/mxnet/data/cifar10.zip',
                                 dirname='data')
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall('data')

def get_mnist_iterator(batch_size, input_shape, num_parts=1, part_index=0):
    """Returns training and validation iterators for MNIST dataset
    """

    get_mnist_ubyte()
    flat = False if len(input_shape) == 3 else True # pylint: disable=simplifiable-if-expression

    train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat,
        num_parts=num_parts,
        part_index=part_index)

    val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat,
        num_parts=num_parts,
        part_index=part_index)

    return (train_dataiter, val_dataiter)

def get_zip_data(data_dir, url, data_origin_name):
    """Download and extract zip data.

    Parameters
    ----------

    data_dir : str
        Absolute or relative path of the directory name to store zip files
    url : str
        URL to download data from
    data_origin_name : str
        Name of the downloaded zip file

    Examples
    --------
    >>> get_zip_data("data_dir",
                     "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
                     "ml-10m.zip")
    """
    data_origin_name = os.path.join(data_dir, data_origin_name)
    if not os.path.exists(data_origin_name):
        download(url, dirname=data_dir, overwrite=False)
        zip_file = zipfile.ZipFile(data_origin_name)
        zip_file.extractall(path=data_dir)

def get_bz2_data(data_dir, data_name, url, data_origin_name):
    """Download and extract bz2 data.

    Parameters
    ----------

    data_dir : str
        Absolute or relative path of the directory name to store bz2 files
    data_name : str
        Name of the output file in which bz2 contents will be extracted
    url : str
        URL to download data from
    data_origin_name : str
        Name of the downloaded b2 file

    Examples
    --------
    >>> get_bz2_data("data_dir", "kdda.t",
                     "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2",
                     "kdda.t.bz2")
    """

    data_name = os.path.join(data_dir, data_name)
    data_origin_name = os.path.join(data_dir, data_origin_name)
    if not os.path.exists(data_name):
        download(url, fname=data_origin_name, dirname=data_dir, overwrite=False)
        bz_file = bz2.BZ2File(data_origin_name, 'rb')
        with open(data_name, 'wb') as fout:
            for line in bz_file:
                fout.write(line)
            bz_file.close()
        os.remove(data_origin_name)

def set_env_var(key, val, default_val=""):
    """Set environment variable

    Parameters
    ----------

    key : str
        Env var to set
    val : str
        New value assigned to the env var
    default_val : str, optional
        Default value returned if the env var doesn't exist

    Returns
    -------
    str
        The value of env var before it is set to the new value
    """
    prev_val = os.environ.get(key, default_val)
    os.environ[key] = val
    return prev_val

def same_array(array1, array2):
    """Check whether two NDArrays sharing the same memory block

    Parameters
    ----------

    array1 : NDArray
        First NDArray to be checked
    array2 : NDArray
        Second NDArray to be checked

    Returns
    -------
    bool
        Whether two NDArrays share the same memory
    """
    array1[:] += 1
    if not same(array1.asnumpy(), array2.asnumpy()):
        array1[:] -= 1
        return False
    array1[:] -= 1
    return same(array1.asnumpy(), array2.asnumpy())

@contextmanager
def discard_stderr():
    """
    Discards error output of a routine if invoked as:

    with discard_stderr():
        ...
    """
    with open(os.devnull, 'w') as bit_bucket:
        try:
            stderr_fileno = sys.stderr.fileno()
            old_stderr = os.dup(stderr_fileno)
            try:
                os.dup2(bit_bucket.fileno(), stderr_fileno)
                yield
            finally:
                os.dup2(old_stderr, stderr_fileno)
        except AttributeError:
            # On some systems is stderr not a file descriptor but actually a virtual pipeline
            # that can not be copied
            yield


class DummyIter(mx.io.DataIter):
    """A dummy iterator that always returns the same batch of data
    (the first data batch of the real data iter). This is usually used for speed testing.

    Parameters
    ----------
    real_iter: mx.io.DataIter
        The real data iterator where the first batch of data comes from
    """
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size
        self.the_batch = next(real_iter)

    def __iter__(self):
        return self

    def next(self):
        """Get a data batch from iterator. The first data batch of real iter is always returned.
        StopIteration will never be raised.

        Returns
        -------
        DataBatch
            The data of next batch.
        """
        return self.the_batch

def gen_buckets_probs_with_ppf(ppf, nbuckets):
    """Generate the buckets and probabilities for chi_square test when the ppf (Quantile function)
     is specified.

    Parameters
    ----------
    ppf : function
        The Quantile function that takes a probability and maps it back to a value.
        It's the inverse of the cdf function
    nbuckets : int
        size of the buckets

    Returns
    -------
    buckets : list of tuple
        The generated buckets
    probs : list
        The generate probabilities
    """
    assert nbuckets > 0
    probs = [1.0 / nbuckets for _ in range(nbuckets)]
    buckets = [(ppf(i / float(nbuckets)), ppf((i + 1) / float(nbuckets))) for i in range(nbuckets)]
    return buckets, probs

def mean_check(generator, mu, sigma, nsamples=1000000):
    """Test the generator by matching the mean.

    We test the sample mean by checking if it falls inside the range
        (mu - 3 * sigma / sqrt(n), mu + 3 * sigma / sqrt(n))

    References::

        @incollection{goucher2009beautiful,
              title={Beautiful Testing: Leading Professionals Reveal How They Improve Software},
              author={Goucher, Adam and Riley, Tim},
              year={2009},
              chapter=10
        }

    Examples::

        generator = lambda x: np.random.normal(0, 1.0, size=x)
        mean_check_ret = mean_check(generator, 0, 1.0)

    Parameters
    ----------
    generator : function
        The generator function. It's expected to generate N i.i.d samples by calling generator(N).
    mu : float
    sigma : float
    nsamples : int

    Returns
    -------
    ret : bool
        Whether the mean test succeeds
    """
    samples = np.array(generator(nsamples))
    sample_mean = samples.mean()
    ret = (sample_mean > mu - 3 * sigma / np.sqrt(nsamples)) and\
          (sample_mean < mu + 3 * sigma / np.sqrt(nsamples))
    return ret

def get_im2rec_path(home_env="MXNET_HOME"):
    """Get path to the im2rec.py tool

    Parameters
    ----------

    home_env : str
        Env variable that holds the path to the MXNET folder

    Returns
    -------
    str
        The path to im2rec.py
    """
    # Check first if the path to MXNET is passed as an env variable
    if home_env in os.environ:
        mxnet_path = os.environ[home_env]
    else:
        # Else use currently imported mxnet as reference
        mxnet_path = os.path.dirname(mx.__file__)
    # If MXNet was installed through pip, the location of im2rec.py
    im2rec_path = os.path.join(mxnet_path, 'tools', 'im2rec.py')
    if os.path.isfile(im2rec_path):
        return im2rec_path
    # If MXNet has been built locally
    im2rec_path = os.path.join(mxnet_path, '..', '..', 'tools', 'im2rec.py')
    if os.path.isfile(im2rec_path):
        return im2rec_path
    raise IOError('Could not find path to tools/im2rec.py')

def var_check(generator, sigma, nsamples=1000000):
    """Test the generator by matching the variance.
    It will need a large number of samples and is not recommended to use

    We test the sample variance by checking if it falls inside the range
        (sigma^2 - 3 * sqrt(2 * sigma^4 / (n-1)), sigma^2 + 3 * sqrt(2 * sigma^4 / (n-1)))

    References::

        @incollection{goucher2009beautiful,
              title={Beautiful Testing: Leading Professionals Reveal How They Improve Software},
              author={Goucher, Adam and Riley, Tim},
              year={2009},
              chapter=10
        }

    Examples::

        generator = lambda x: np.random.normal(0, 1.0, size=x)
        var_check_ret = var_check(generator, 0, 1.0)

    Parameters
    ----------
    generator : function
        The generator function. It's expected to generate N i.i.d samples by calling generator(N).
    sigma : float
    nsamples : int

    Returns
    -------
    ret : bool
        Whether the variance test succeeds
    """
    samples = np.array(generator(nsamples))
    sample_var = samples.var(ddof=1)
    ret = (sample_var > sigma ** 2 - 3 * np.sqrt(2 * sigma ** 4 / (nsamples - 1))) and\
          (sample_var < sigma ** 2 + 3 * np.sqrt(2 * sigma ** 4 / (nsamples - 1)))
    return ret

def chi_square_check(generator, buckets, probs, nsamples=1000000):
    """Run the chi-square test for the generator. The generator can be both continuous and discrete.

    If the generator is continuous, the buckets should contain tuples of (range_min, range_max) \
    and the probs should be the corresponding ideal probability within the specific ranges. \
    Otherwise, the buckets should contain all the possible values generated over the discrete distribution and the \
    probs should be groud-truth probability.

    Usually the user is required to specify the probs parameter.

    After obtaining the p value, we could further use the standard p > 0.05 (alpha) threshold to get \
    the final result.

    Examples::

      buckets, probs = gen_buckets_probs_with_ppf(lambda x: ss.norm.ppf(x, 0, 1), 5)
      generator = lambda x: np.random.normal(0, 1.0, size=x)
      p = chi_square_check(generator=generator, buckets=buckets, probs=probs)
      assert(p > 0.05)

    Parameters
    ----------
    generator: function
        A function that is assumed to generate i.i.d samples from a specific distribution.
        generator(N) should generate N random samples.
    buckets: list of tuple or list of number
        The buckets to run the chi-square the test. Make sure that the buckets cover
        the whole range of the distribution. Also, the buckets must be in ascending order and have
        no intersection
    probs: list or tuple
        The ground-truth probability of the random value fall in a specific bucket.
    nsamples:int
        The number of samples to generate for the testing

    Returns
    -------
    p : float
        p value that the generator has the expected distribution.
        A higher value indicates a larger confidence
    obs_freq : list
        Observed frequency of buckets
    expected_freq : list
        The expected (ground-truth) frequency of the buckets
    """
    if not ss:
        raise ImportError("scipy is not available."
                          " Please check if the scipy python bindings are installed.")
    assert isinstance(buckets, list)
    samples = generator(nsamples)
    assert len(probs) == len(buckets)
    if isinstance(buckets[0], (list, tuple)):
        # Check whether the buckets are valid and fill them into a npy array
        continuous_dist = True
        buckets_npy = np.zeros((len(buckets) * 2, ), dtype=np.float32)
        for i, _ in enumerate(buckets):
            assert(buckets[i][0] <= buckets[i][1])
            if i < len(buckets) - 1:
                assert(buckets[i][1] <= buckets[i + 1][0])
            buckets_npy[i * 2] = buckets[i][0]
            buckets_npy[i * 2 + 1] = buckets[i][1]
    else:
        continuous_dist = False
    expected_freq = (nsamples * np.array(probs, dtype=np.float32)).astype(np.int32)
    if continuous_dist:
        sample_bucket_ids = np.searchsorted(buckets_npy, samples, side='right')
    else:
        sample_bucket_ids = np.array(samples)
    if continuous_dist:
        sample_bucket_ids = sample_bucket_ids // 2
    obs_freq = np.zeros(shape=len(buckets), dtype=np.int)
    for i, _ in enumerate(buckets):
        if continuous_dist:
            obs_freq[i] = (sample_bucket_ids == i).sum()
        else:
            obs_freq[i] = (sample_bucket_ids == buckets[i]).sum()
    _, p = ss.chisquare(f_obs=obs_freq, f_exp=expected_freq)
    return p, obs_freq, expected_freq

def verify_generator(generator, buckets, probs, nsamples=1000000, nrepeat=5, success_rate=0.2, alpha=0.05):
    """Verify whether the generator is correct using chi-square testing.

    The test is repeated for "nrepeat" times and we check if the success rate is
     above the threshold (25% by default).

    Parameters
    ----------
    generator: function
        A function that is assumed to generate i.i.d samples from a specific distribution.
            generator(N) should generate N random samples.
    buckets: list of tuple or list of number
        The buckets to run the chi-square the test. Make sure that the buckets cover
         the whole range of the distribution. Also, the buckets must be in ascending order and
         have no intersection
    probs: list or tuple
        The ground-truth probability of the random value fall in a specific bucket.
    nsamples: int
        The number of samples to generate for the testing
    nrepeat: int
        The times to repeat the test
    success_rate: float
        The desired success rate
    alpha: float
        The desired threshold for type-I error i.e. when a true null hypothesis is rejected

    Returns
    -------
    cs_ret_l: list
        The p values of the chi-square test.
    """
    cs_ret_l = []
    obs_freq_l = []
    expected_freq_l = []
    for _ in range(nrepeat):
        cs_ret, obs_freq, expected_freq = chi_square_check(generator=generator, buckets=buckets,
                                                           probs=probs, nsamples=nsamples)
        cs_ret_l.append(cs_ret)
        obs_freq_l.append(obs_freq)
        expected_freq_l.append(expected_freq)
    success_num = (np.array(cs_ret_l) > alpha).sum()
    if success_num < nrepeat * success_rate:
        raise AssertionError("Generator test fails, Chi-square p=%s, obs_freq=%s, expected_freq=%s."
                             "\nbuckets=%s, probs=%s"
                             % (str(cs_ret_l), str(obs_freq_l), str(expected_freq_l),
                                str(buckets), str(probs)))
    return cs_ret_l

def compare_ndarray_tuple(t1, t2, rtol=None, atol=None):
    """Compare ndarray tuple."""
    if t1 is not None and t2 is not None:
        if isinstance(t1, tuple):
            for s1, s2 in zip(t1, t2):
                compare_ndarray_tuple(s1, s2, rtol, atol)
        else:
            assert_almost_equal(t1.asnumpy(), t2.asnumpy(), rtol=rtol, atol=atol)


def compare_optimizer(opt1, opt2, shape, dtype, w_stype='default', g_stype='default',
                      rtol=1e-4, atol=1e-5, compare_states=True):
    """Compare opt1 and opt2."""
    if w_stype == 'default':
        w2 = mx.random.uniform(shape=shape, ctx=default_context(), dtype=dtype)
        w1 = w2.copyto(default_context())
    elif w_stype in ('row_sparse', 'csr'):
        w2 = rand_ndarray(shape, w_stype, density=1, dtype=dtype)
        w1 = w2.copyto(default_context()).tostype('default')
    else:
        raise Exception("type not supported yet")
    if g_stype == 'default':
        g2 = mx.random.uniform(shape=shape, ctx=default_context(), dtype=dtype)
        g1 = g2.copyto(default_context())
    elif g_stype in ('row_sparse', 'csr'):
        g2 = rand_ndarray(shape, g_stype, dtype=dtype)
        g1 = g2.copyto(default_context()).tostype('default')
    else:
        raise Exception("type not supported yet")

    state1 = opt1.create_state_multi_precision(0, w1)
    state2 = opt2.create_state_multi_precision(0, w2)
    if compare_states:
        compare_ndarray_tuple(state1, state2)

    opt1.update_multi_precision(0, w1, g1, state1)
    opt2.update_multi_precision(0, w2, g2, state2)
    if compare_states:
        compare_ndarray_tuple(state1, state2, rtol=rtol, atol=atol)
    assert_almost_equal(w1.asnumpy(), w2.asnumpy(), rtol=rtol, atol=atol)

class EnvManager(object):
    """Environment variable setter and unsetter via with idiom"""
    def __init__(self, key, val):
        self._key = key
        self._next_val = val
        self._prev_val = None

    def __enter__(self):
        self._prev_val = os.environ.get(self._key)
        os.environ[self._key] = self._next_val

    def __exit__(self, ptype, value, trace):
        if self._prev_val:
            os.environ[self._key] = self._prev_val
        else:
            del os.environ[self._key]
