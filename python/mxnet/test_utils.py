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
import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
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
from .device import current_device
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, get_dtype_name
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, use_np_default_dtype, getenv, setenv  # pylint: disable=unused-import
from .util import get_max_supported_compute_capability, get_rtc_compile_opts # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability


def default_device():
    """Get default device for regression test."""
    # _TODO: get device from environment variable to support
    # testing with GPUs
    return current_device()


def set_default_device(device):
    """Set default device."""
    mx.device._current.set(device)


def default_dtype():
    """Get default data type for regression test."""
    # _TODO: get default dtype from environment variable
    return np.float32

def default_rtols():
    """Get default relative tolerances for data comparisons involving each data type."""
    return {np.dtype(np.float16): 1e-2,
            np.dtype(np.float32): 1e-4,
            np.dtype(np.float64): 1e-5,
            np.dtype(np.bool): 0,
            np.dtype(np.int8): 0,
            np.dtype(np.uint8): 0,
            np.dtype(np.int32): 0,
            np.dtype(np.uint32): 0,
            np.dtype(np.int64): 0,
            np.dtype(np.uint64): 0}

def default_atols():
    """Get default absolute tolerances for data comparisons involving each data type."""
    return {np.dtype(np.float16): 1e-1,
            np.dtype(np.float32): 1e-3,
            np.dtype(np.float64): 1e-20,
            np.dtype(np.bool): 0,
            np.dtype(np.int8): 0,
            np.dtype(np.uint8): 0,
            np.dtype(np.int32): 0,
            np.dtype(np.uint32): 0,
            np.dtype(np.int64): 0,
            np.dtype(np.uint64): 0}

def default_numeric_eps():
    """Get default epsilon for finite difference gradient calculations with data type."""
    # prefer a power-of-two eps, since no bits are dropped when serving as an input delta
    return {np.dtype(np.float16): 1.0 / 2**6,
            np.dtype(np.float32): 1.0 / 2**9,
            np.dtype(np.float64): 1.0 / 2**14}


def effective_dtype(dat):
    """ Return the most appropriate dtype for determining the tolerance used in dat comparisons
    Parameters
    ----------
    dat : np.ndarray or mx.nd.array or mx.np.ndarray
    """
    # On arch 80 gpus or later, a float32-io gemm or conv op will trim the mantissa of
    # data inputs to be of comparable precision to a float16, so float16 becomes the
    # 'effective dtype' for tolerance tests involving such op outputs.

    # Is TF32 enabled in the device (the default on arch 80 GPUs)
    def is_TF32_enabled(device):
        try:
            return (device.device_type == 'gpu' and
                    get_cuda_compute_capability(device) >= 80 and
                    os.environ.get('NVIDIA_TF32_OVERRIDE') != '0')
        except:  # pylint: disable=bare-except
            return False

    device = dat.device if hasattr(dat, 'device') else None
    dtype = np.dtype(dat.dtype)
    if dtype == np.dtype(np.float32) and is_TF32_enabled(device):
        return np.dtype(np.float16)
    else:
        return dtype


def get_tolerance(dat, tol, default_tol):
    """ Return the tolerance to be used for dat comparisons based on the given tol, datatype and device.
    Parameters
    ----------
    dat : np.ndarray or mx.nd.array or mx.np.ndarray
    tol : float, or a dict of dtype->float
    default_tol : default dict of dtype->float for all types
    """

    if isinstance(tol, numbers.Number):
        return tol

    # If the caller has supplied a tol dict, use that if it has an entry for dtype,
    # else use the supplied default tol dict.
    dtype = effective_dtype(dat)
    tol = {} if tol is None else tol
    return tol.get(dtype, default_tol[dtype])


def get_tols(x, y, rtol, atol):
    """For comparing two datasets 'x' and 'y', what relative and absolute tolerances should be used."""
    # Tolerance analysis needs 'dtype' of 'x' and 'y', so convert numbers to numpy scalars as needed
    if isinstance(x, numbers.Number):
        x = np.array(x)
    if isinstance(y, numbers.Number):
        y = np.array(y)

    # If tols are not specified, use the largest default tol for 'x' and 'y' based on their ctx and dtype.
    rtol = max(get_tolerance(x, rtol, default_rtols()),
               get_tolerance(y, rtol, default_rtols()))
    atol = max(get_tolerance(x, atol, default_atols()),
               get_tolerance(y, atol, default_atols()))

    return rtol, atol


def get_atol(atol=None, dtype=np.dtype(np.float64)):
    """Get default numerical threshold for regression test."""
    return default_atols()[dtype] if atol is None else atol

def get_rtol(rtol=None, dtype=np.dtype(np.float64)):
    """Get default numerical threshold for regression test."""
    return default_rtols()[dtype] if rtol is None else rtol

def get_etol(etol=None):
    """Get default numerical threshold for regression test."""
    # _TODO: get from env variable, different threshold might
    # be needed for different device and dtype
    return 0 if etol is None else etol

def random_arrays(*shapes):
    """Generate some random numpy arrays."""
    arrays = [np.array(np.random.randn(), dtype=default_dtype())
              if len(s) == 0 else np.random.randn(*s).astype(default_dtype())
              for s in shapes]
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def random_uniform_arrays(*shapes, **kwargs):
    """Generate some random numpy arrays."""
    low = kwargs.pop('low', 0.0)
    high = kwargs.pop('high', 1.0)
    dtype = kwargs.pop('dtype', default_dtype())
    if len(kwargs) > 0:
        raise TypeError('Got unexpected argument/s : ' + str(kwargs.keys()))
    arrays = [np.random.uniform(low, high, size=s).astype(dtype)
              for s in shapes]
    return arrays


def random_sample(population, k):
    """Return a k length list of the elements chosen from the population sequence."""
    assert 0 <= k <= len(population)
    population_copy = population[:]
    np.random.shuffle(population_copy)
    return population_copy[0:k]


def _sorted_items(d):
    """Return (key, value) pairs of dict 'd' in a deterministic order (sorted by key)."""
    return sorted(d.items(), key=lambda t: t[0])


def _sorted_dict(d):
    """Return ordered dictionary containing items ordered by their keys."""
    return OrderedDict(_sorted_items(d))


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
            raise ValueError(f"not supported for this density: {density}"
                             f" for this shape ({num_rows}, {num_cols})"
                             " Please keep :"
                             " num_rows * num_cols * density >= 2 * num_rows")


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

    if unused_nnz > 0:
        raise ValueError(f"not supported for this density: {density}"
                         f" for this shape ({num_rows},{num_cols})")

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

def create_2d_np_tensor(rows, columns, dtype=np.int64):
    inp = mx.np.arange(0, rows, dtype=dtype).reshape(rows, 1)
    inp = mx.np.broadcast_to(inp, shape=(inp.shape[0], columns))
    return inp

# For testing Large Tensors having total size > 2^32 elements
def create_2d_tensor(rows, columns, dtype=np.int64):
    a = mx.nd.arange(0, rows, dtype=dtype).reshape(rows, 1)
    b = mx.nd.broadcast_to(a, shape=(a.shape[0], columns))
    return b

# For testing Large Vectors having total size > 2^32 elements
def create_vector(size, dtype=np.int64):
    a = mx.nd.arange(0, size, dtype=dtype)
    return a

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
    ctx = ctx if ctx else default_device()
    density = rnd.rand() if density is None else density
    dtype = default_dtype() if dtype is None else dtype
    distribution = "uniform" if distribution is None else distribution
    if stype == 'row_sparse':
        assert (distribution == "uniform"), \
               f"Distribution {distribution} not supported for row_sparse"
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
            assert(False), f"Distribution not supported: {distribution}"
            return False
    else:
        assert(False), "unknown storage type"
        return False

def rand_ndarray(shape, stype='default', density=None, dtype=None, modifier_func=None,
                 shuffle_csr_indices=False, distribution=None, ctx=None):
    """Generate a random sparse ndarray. Returns the generated ndarray."""
    ctx = ctx if ctx else default_device()
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


def rand_shape_2d(dim0=10, dim1=10, allow_zero_size=False):
    low = 0 if allow_zero_size else 1
    return rnd.randint(low, dim0 + 1), rnd.randint(low, dim1 + 1)


def rand_shape_3d(dim0=10, dim1=10, dim2=10, allow_zero_size=False):
    low = 0 if allow_zero_size else 1
    return rnd.randint(low, dim0 + 1), rnd.randint(low, dim1 + 1), rnd.randint(low, dim2 + 1)


def rand_shape_nd(num_dim, dim=10, allow_zero_size=False):
    low = 0 if allow_zero_size else 1
    return tuple(rnd.randint(low, dim+1, size=num_dim))


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


def _find_max_violation(a, b, rtol, atol):
    """Finds and returns the location of maximum violation."""
    # 'smart' absdiff that considers inf's as equals (to match np.allclose)
    absdiff = np.where(np.equal(a, b), 0, np.abs(a-b))
    tol = atol + rtol*np.abs(b)
    violation = absdiff/(tol+1e-20)
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


def checkShapes(a, b):
    if a.shape != b.shape:
        msg = npt.build_err_msg([a, b],
                                err_msg="a.shape = {} and b.shape = {} are not equal"
                                .format(str(a.shape), str(b.shape)))
        raise AssertionError(msg)


def almost_equal(a, b, rtol=None, atol=None, equal_nan=False, use_broadcast=True):
    """Test if two numpy arrays are almost equal."""
    # pylint: disable=unexpected-keyword-arg
    if not use_broadcast:
        checkShapes(a, b)

    return np.allclose(a, b, rtol=get_rtol(rtol), atol=get_atol(atol), equal_nan=equal_nan)
    # pylint: enable=unexpected-keyword-arg

def locationError(a, b, index, names, maxError=False):
    """Create element mismatch comment

    Parameters
    ----------
    a, b : compared np.ndarray's
    index : tuple of coordinate arrays
        Location of violation
    names : tuple of names
        The names of compared arrays.
    maxError: boolean, optional
        Flag indicating that maximum error is reporting.
    """
    maximum = "maximum " if maxError else ""
    return f"Location of {maximum} error: {str(index)}, {names[0]}={a[index]:.8f}, {names[1]}={b[index]:.8f}"
def assert_almost_equal(a, b, rtol=None, atol=None, names=('a', 'b'), equal_nan=False,
                        use_broadcast=True, mismatches=(10, 10)):
    """Test that two numpy arrays are almost equal. Raise exception message if not.

    Parameters
    ----------
    a : np.ndarray or mx.nd.array
    b : np.ndarray or mx.nd.array
    rtol : None or float or dict of dtype -> float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float or dict of dtype -> float
        The absolute threshold. Default threshold will be used if set to ``None``.
    names : tuple of names, optional
        The names used in error message when an exception occurs
    equal_nan : boolean, optional
        The flag determining how to treat NAN values in comparison
    mismatches : tuple of mismatches
        Maximum number of mismatches to be printed (mismatches[0]) and determine (mismatches[1])
    """
    if not use_broadcast:
        checkShapes(a, b)

    rtol, atol = get_tols(a, b, rtol, atol)

    if isinstance(a, mx.numpy.ndarray):
        a = a.asnumpy()
    if isinstance(b, mx.numpy.ndarray):
        b = b.asnumpy()
    use_np_allclose = isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    if not use_np_allclose:
        if not (hasattr(a, 'ctx') and hasattr(b, 'ctx') and a.device == b.device and a.dtype == b.dtype):
            use_np_allclose = True
            if isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            if isinstance(b, mx.nd.NDArray):
                b = b.asnumpy()

    if use_np_allclose:
        if hasattr(a, 'dtype') and a.dtype == np.bool_ and hasattr(b, 'dtype') and b.dtype == np.bool_:
            np.testing.assert_equal(a, b)
            return
        if almost_equal(a, b, rtol, atol, equal_nan=equal_nan):
            return
    else:
        output = mx.nd.contrib.allclose(a, b, rtol, atol, equal_nan)
        if output.asnumpy() == 1:
            return

        a = a.asnumpy()
        b = b.asnumpy()

    index, rel = _find_max_violation(a, b, rtol, atol)
    if index != ():
        # a, b are the numpy arrays
        indexErr = index
        relErr = rel

        print('\n*** Maximum errors for vector of size {}:  rtol={}, atol={}\n'.format(a.size, rtol, atol))
        aTmp = a.copy()
        bTmp = b.copy()
        i = 1
        while i <= a.size:
            if i <= mismatches[0]:
                print(f"{i:3d}: Error {rel}  {locationError(a, b, index, names)}")

            aTmp[index] = bTmp[index] = 0
            if almost_equal(aTmp, bTmp, rtol, atol, equal_nan=equal_nan):
                break

            i += 1
            if i <= mismatches[1] or mismatches[1] <= 0:
                index, rel = _find_max_violation(aTmp, bTmp, rtol, atol)
            else:
                break

        mismatchDegree = "at least " if mismatches[1] > 0 and i > mismatches[1] else ""
        errMsg = f"Error {relErr} exceeds tolerance rtol={rtol:e}, atol={atol:e} " \
                 f"(mismatch {mismatchDegree}{100*i/a.size}%).\n" \
                 f"{locationError(a, b, indexErr, names, maxError=True)}"
    else:
        errMsg = f"Error {rel} exceeds tolerance rtol={rtol:e}, atol={atol:e}.\n"

    np.set_printoptions(threshold=4, suppress=True)
    msg = npt.build_err_msg([a, b], err_msg=errMsg)

    raise AssertionError(msg)


def assert_allclose(a, b, rtol=1e-07, atol=0, equal_nan=True):
    assert_almost_equal(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def assert_almost_equal_with_err(a, b, rtol=None, atol=None, etol=None,
                                 names=('a', 'b'), equal_nan=False, mismatches=(10, 10)):
    """Test that two numpy arrays are almost equal within given error rate. Raise exception message if not.

    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    rtol : None or float or dict of dtype -> float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float or dict of dtype -> float
        The absolute threshold. Default threshold will be used if set to ``None``.
    etol : None or float
        The error rate threshold. If etol is float, return true if error_rate < etol even if
        any error is found.
    names : tuple of names, optional
        The names used in error message when an exception occurs
    equal_nan : boolean, optional
        The flag determining how to treat NAN values in comparison
    mismatches : tuple of mismatches
        Maximum number of mismatches to be printed (mismatches[0]) and determine (mismatches[1])
    """
    etol = get_etol(etol)
    if etol > 0:
        rtol, atol = get_tols(a, b, rtol, atol)
        if isinstance(a, mx.nd.NDArray):
            a = a.asnumpy()
        if isinstance(b, mx.nd.NDArray):
            b = b.asnumpy()
        equals = np.isclose(a, b, rtol=rtol, atol=atol)
        err = 1 - np.count_nonzero(equals) / equals.size
        if err > etol:
            index, rel = _find_max_violation(a, b, rtol, atol)
            indexErr = index
            relErr = rel

            print('\n*** Maximum errors for vector of size {}:  rtol={}, atol={}\n'.format(a.size, rtol, atol))
            aTmp = a.copy()
            bTmp = b.copy()
            i = 1
            while i <= a.size:
                if i <= mismatches[0]:
                    print(f"{i:3d}: Error {rel}  {locationError(a, b, index, names)}")

                aTmp[index] = bTmp[index] = 0
                if almost_equal(aTmp, bTmp, rtol, atol, equal_nan=equal_nan):
                    break

                i += 1
                if i <= mismatches[1] or mismatches[1] <= 0:
                    index, rel = _find_max_violation(aTmp, bTmp, rtol, atol)
                else:
                    break

            mismatchDegree = "at least " if mismatches[1] > 0 and i > mismatches[1] else ""
            errMsg = f"Error {relErr} exceeds tolerance rtol={rtol:e}, atol={atol:e} " \
                     f"(mismatch {mismatchDegree}{100*i/a.size}%).\n" \
                     f"{locationError(a, b, indexErr, names, maxError=True)}"
            np.set_printoptions(threshold=4, suppress=True)
            msg = npt.build_err_msg([a, b], err_msg=errMsg)
            raise AssertionError(msg)
    else:
        assert_almost_equal(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


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


def _parse_location(sym, location, ctx, dtype=default_dtype()):
    """Parses the given location to a ordered dictionary.

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
    ctx : Device
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
                             f"symbol args:{str(set(sym.list_arguments()))}, location.keys():{str(set(location.keys()))}")
    else:
        location = {k: v for k, v in zip(sym.list_arguments(), location)}
    location = {k: mx.nd.array(v, ctx=ctx, dtype=v.dtype if dtype == "asnumpy" else dtype) \
               if isinstance(v, np.ndarray) else v for k, v in location.items()}
    return _sorted_dict(location)


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
    ctx : Device
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
                                 f"symbol aux_names:{str(set(sym.list_auxiliary_states()))}, aux_states.keys:{str(set(aux_states.keys()))}")
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
        for i in range(int(np.prod(v.shape))):
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

def check_numeric_gradient(sym, location, aux_states=None, numeric_eps=None, rtol=None,
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
    rtol : None or float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float
        The absolute threshold. Default threshold will be used if set to ``None``.
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
    if ctx is None:
        ctx = default_device()

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
    is_np_sym = bool(isinstance(sym, np_symbol))
    if is_np_sym:  # convert to np symbol for using element-wise multiplication
        proj = proj.as_np_ndarray()
    out = sym * proj
    if is_np_sym:  # convert to classic symbol so that make_loss can be used
        out = out.as_nd_ndarray()
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

    grad_req["__random_proj"] = 'write'
    executor = out._bind(ctx, grad_req=grad_req,
                         args=location, args_grad=args_grad, aux_states=aux_states)

    inps = executor.arg_arrays
    if len(inps) != len(location):
        raise ValueError("Executor arg_arrays and and location len do not match."
                         f"Got {len(inps)} inputs and {len(location)} locations")

    executor.forward(is_train=True)
    assert len(executor.outputs) == 1

    eps = get_tolerance(executor.outputs[0], numeric_eps, default_numeric_eps())
    # cannot use finite differences with small eps without high precision
    if dtype in (np.float32, np.float16):
        assert eps >= 1e-5

    executor.backward()
    symbolic_grads = executor.grad_dict

    numeric_gradients = numeric_grad(
        executor, location_npy, aux_states_npy,
        eps=eps, use_forward_train=use_forward_train, dtype=dtype)

    for name in grad_nodes:
        fd_grad = numeric_gradients[name]
        orig_grad = args_grad_npy[name]
        sym_grad = symbolic_grads[name]
        if grad_req[name] == 'write':
            assert_almost_equal(fd_grad, sym_grad, rtol, atol,
                                (f"NUMERICAL_{name}", f"BACKWARD_{name}"))
        elif grad_req[name] == 'add':
            if isinstance(sym_grad, mx.nd.NDArray):
                sym_grad = sym_grad.asnumpy()
            assert_almost_equal(fd_grad, sym_grad - orig_grad, rtol, atol,
                                (f"NUMERICAL_{name}", f"BACKWARD_{name}"))
        elif grad_req[name] == 'null':
            assert sym_grad is None
        else:
            raise ValueError(f"Invalid grad_req {grad_req[name]} for argument {name}")


def check_symbolic_forward(sym, location, expected, rtol=None, atol=None,
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
    rtol : None or float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float
        The absolute threshold. Default threshold will be used if set to ``None``.
    aux_states : list of np.ndarray of dict, optional
        - if type is list of np.ndarray
            Contains all the NumPy arrays corresponding to sym.list_auxiliary_states
        - if type is dict of str to np.ndarray
            Contains the mapping between names of auxiliary states and their values.
    device : Device, optional
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
        ctx = default_device()

    location = _parse_location(sym=sym, location=location, ctx=ctx, dtype=dtype)
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx,
                                   dtype=dtype)
    if isinstance(expected, dict):
        expected = [expected[k] for k in sym.list_outputs()]
    args_grad_data = {k:mx.nd.empty(v.shape, ctx=ctx, dtype=v.dtype if dtype == "asnumpy" else dtype) \
                      for k, v in location.items()}

    executor = sym._bind(ctx=ctx, args=location, args_grad=args_grad_data, aux_states=aux_states)
    for g in executor.grad_arrays:
        if g.ndim == 0:
            g[()] = 0
        else:
            g[:] = 0

    executor.forward(is_train=False)

    outputs = executor.outputs
    for output_name, expect, output in zip(sym.list_outputs(), expected, outputs):
        assert_almost_equal(expect, output, rtol, atol,
                            (f"EXPECTED_{output_name}", f"FORWARD_{output_name}"),
                            equal_nan=equal_nan)
    return executor.outputs

def check_symbolic_backward(sym, location, out_grads, expected, rtol=None, atol=None,
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
    rtol : None or float
        The relative threshold. Default threshold will be used if set to ``None``.
    atol : None or float
        The absolute threshold. Default threshold will be used if set to ``None``.
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
    >>> exec_add = sym_add._bind(default_device(), args={'lhs': mat1, 'rhs': mat2},
    ... args_grad={'lhs': grad1, 'rhs': grad2}, grad_req={'lhs': 'write', 'rhs': 'write'})
    >>> exec_add.forward(is_train=True)
    >>> ograd = mx.nd.ones(shape)
    >>> grad_expected = ograd.copy().asnumpy()
    >>> check_symbolic_backward(sym_add, [mat1, mat2], [ograd], [grad_expected, grad_expected])
    """
    assert dtype == 'asnumpy' or dtype in (np.float16, np.float32, np.float64)
    if ctx is None:
        ctx = default_device()

    location = _parse_location(sym=sym, location=location, ctx=ctx, dtype=dtype)
    aux_states = _parse_aux_states(sym=sym, aux_states=aux_states, ctx=ctx,
                                   dtype=dtype)
    if isinstance(expected, (list, tuple)):
        expected = {k:v for k, v in zip(sym.list_arguments(), expected)}

    # Dirty the output buffer deterministically, for reproducibility.
    args_grad_npy = {k:np.random.normal(size=v.shape) for k, v in _sorted_items(expected)}
    args_grad_data = {}
    for k, v in args_grad_npy.items():
        nd = mx.nd.array(v, ctx=ctx, dtype=expected[k].dtype if dtype == "asnumpy" else dtype)
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

    executor = sym._bind(ctx=ctx, args=location, args_grad=args_grad_data,
                         aux_states=aux_states, grad_req=grad_req)
    outputs = executor.forward(is_train=True)

    if isinstance(out_grads, (tuple, list)):
        outg = list()
        for i, arr in enumerate(out_grads):
            stype = outputs[i].stype
            if isinstance(arr, np.ndarray):
                dtype = arr.dtype if dtype == "asnumpy" else dtype
                outg.append(mx.nd.array(arr, ctx=ctx, dtype=dtype).tostype(stype))
            else:
                outg.append(arr.tostype(stype))
        out_grads = outg
    elif isinstance(out_grads, dict):
        outg = dict()
        for k, v in out_grads.items():
            if isinstance(v, np.ndarray):
                dtype = v.dtype if dtype == "asnumpy" else dtype
                outg[k] = mx.nd.array(v, ctx=ctx, dtype=dtype)
            else:
                outg[k] = v
        out_grads = outg
    else:
        assert out_grads is None
    executor.backward(out_grads)

    grads = args_grad_data

    for name in expected:
        if grad_req[name] == 'write':
            assert_almost_equal(expected[name], grads[name], rtol, atol,
                                (f"EXPECTED_{name}", f"BACKWARD_{name}"),
                                equal_nan=equal_nan)
        elif grad_req[name] == 'add':
            grad = grads[name].asnumpy() if isinstance(grads[name], mx.nd.NDArray) else grads[name]
            assert_almost_equal(expected[name], grad - args_grad_npy[name],
                                rtol, atol, (f"EXPECTED_{name}", f"BACKWARD_{name}"),
                                equal_nan=equal_nan)
        elif grad_req[name] == 'null':
            assert_almost_equal(args_grad_npy[name], grads[name],
                                rtol, atol, (f"EXPECTED_{name}", f"BACKWARD_{name}"),
                                equal_nan=equal_nan)
        else:
            raise ValueError(f"Invalid grad_req {grad_req[name]} for argument {name}")
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
        ctx = default_device()

    if grad_req is None:
        grad_req = 'write'
    if location is None:
        exe = sym._simple_bind(grad_req=grad_req, ctx=ctx, **kwargs)
        location = {k: np.random.normal(size=arr.shape, scale=1.0) for k, arr in
                    exe.arg_dict.items()}
    else:
        assert isinstance(location, dict), f'Expect dict, get "location"={str(location)}'
        exe = sym._simple_bind(grad_req=grad_req, ctx=ctx,
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
                      arg_params=None, aux_params=None, rtol=None, atol=None,
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
    arg_params : dict of input name -> input data
        data to use for non-aux inputs
    aux_params : dict of input name -> input data
        data to use for aux inputs
    rtol : float or dictionary dtype->float, optional
        The relative error tolerance.
    atol : float or dictionary dtype->float, optional
        The absolute error tolerance.
    raise_on_err : bool, optional, defaults to True
        Should an error raise an exception (or just output exception message)
    ground_truth : dict of output name -> data, optional
        Provided ideal result to be compared against
    equal_nan : bool, optional, defaults to False
        Should nans be treated as equal in the comparison
    use_uniform: bool
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
        exe_list.append(s._simple_bind(grad_req=grad_req, **ctx))

    arg_params = {} if arg_params is None else arg_params
    aux_params = {} if aux_params is None else aux_params

    # returns the least precise of two dtypes
    def smaller_dtype(dt1, dt2):
        return dt1 if dt2 is None or np.dtype(dt1).itemsize < np.dtype(dt2).itemsize else dt2

    # It's important to assign random inputs in a deterministic order, for reproducibility.
    for n, arr in _sorted_items(exe_list[0].arg_dict):
        if n not in arg_params:
            if use_uniform:
                arg_params[n] = np.random.uniform(low=-0.92 * scale, high=0.92 * scale,
                                                  size=arr.shape).astype(rand_type)
            else:
                arg_params[n] = np.random.normal(size=arr.shape,
                                                 scale=scale).astype(rand_type)
    for n in exe_list[0].aux_dict:
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

    # test
    for exe in exe_list:
        exe.forward(is_train=False)

    dtypes = [np.dtype(exe.outputs[0].dtype) for exe in exe_list]
    # Select the ground truth as the first model having the highest precision output[0]
    gt_idx = np.argmax(dtypes)
    gt = ground_truth
    if gt is None:
        gt = exe_list[gt_idx].output_dict.copy()

    for i, exe in enumerate(exe_list):
        if i == gt_idx:
            continue

        for name, arr in zip(output_names, exe.outputs):
            gtarr = gt[name]
            try:
                assert_almost_equal(arr, gtarr, rtol=rtol, atol=atol, equal_nan=equal_nan)
            except AssertionError as e:
                print(f'Predict Err: ctx {i} vs ctx {gt_idx} at {name}')
                traceback.print_exc()
                if raise_on_err:
                    raise e

                print(str(e))

    # train
    if grad_req != 'null':
        # Perform forward()
        for exe in exe_list:
            exe.forward(is_train=True)
        # Use the first executor's output data, cast to the least precise dtype,
        # as the gradient data to pass to all executor's backward() call.
        least_precise_dtype = [out.dtype for out in exe_list[0].outputs]
        for exe in exe_list:
            least_precise_dtype = [smaller_dtype(out1.dtype, dt) \
                                    for (out1, dt) in zip(exe.outputs, least_precise_dtype)]
        golden_data_np = [out.astype(dt).asnumpy() \
                          for (out, dt) in zip(exe_list[0].outputs, least_precise_dtype)]
        # Perform backward()
        for exe in exe_list:
            out_grads = [mx.nd.array(golden_np, ctx=exe._device,
                                     dtype=out.dtype).tostype(out.stype)
                         for (golden_np, out) in zip(golden_data_np, exe.outputs)]
            exe.backward(out_grads)

        gt = ground_truth
        if gt is None:
            gt = exe_list[gt_idx].output_dict.copy()
            if grad_req != 'null':
                gt.update(exe_list[gt_idx].grad_dict)
        for i, exe in enumerate(exe_list):
            if i == gt_idx:
                continue

            curr = zip(output_names + arg_names, exe.outputs + exe.grad_arrays)
            for name, arr in curr:
                if gt[name] is None:
                    assert arr is None, name
                    continue

                gtarr = gt[name]
                try:
                    rt, at = rtol, atol
                    # If the primary data i/o type is float16, then the tolerance used when
                    # comparing a float32 input gradient (e.g. batchnorm gamma) should be float16.
                    smaller_arr_dtype = smaller_dtype(arr.dtype, dtypes[i])
                    smaller_gt_dtype = smaller_dtype(gtarr.dtype, dtypes[gt_idx])
                    if smaller_arr_dtype != arr.dtype or \
                       smaller_gt_dtype != gtarr.dtype:
                        rt, at = get_tols(arr.astype(smaller_arr_dtype),
                                          gtarr.astype(smaller_gt_dtype), rtol, atol)
                    assert_almost_equal(arr, gtarr, rtol=rt, atol=at, equal_nan=equal_nan)
                except AssertionError as e:
                    print('Train Err: {} {} ctx {} vs {} {} ctx {} at {}'.format(
                        get_dtype_name(arr.dtype), arr.device, i,
                        get_dtype_name(gtarr.dtype), gtarr.device, gt_idx, name))
                    traceback.print_exc()
                    if raise_on_err:
                        raise e

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
            assert r.status_code == 200, f"failed to open {url}"
            with open(fname, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                break
        except Exception as e:
            retries -= 1
            if retries <= 0:
                raise e

            print("download failed, retrying, {} attempt{} left"
                  .format(retries, 's' if retries > 1 else ''))
    logging.info("downloaded %s into %s successfully", url, fname)
    return fname


def get_mnist(path='data'):
    """Download and load the MNIST dataset

    Parameters
    ----------
    path : str
        Path in which to save the files.

    Returns
    -------
    dict
        A dict containing the data.
    """
    def read_data(label_url, image_url):
        if not os.path.isdir(path):
            os.makedirs(path)
        with gzip.open(mx.gluon.utils.download(label_url, path=path)) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.frombuffer(flbl.read(), dtype=np.int8)
        with gzip.open(mx.gluon.utils.download(image_url, path=path), 'rb') as fimg:
            _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
            image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32)/255
        return (label, image)

    # changed to mxnet.io for more stable hosting
    url_path = 'https://repo.mxnet.io/gluon/dataset/mnist/'
    (train_lbl, train_img) = read_data(
        url_path+'train-labels-idx1-ubyte.gz', url_path+'train-images-idx3-ubyte.gz')
    (test_lbl, test_img) = read_data(
        url_path+'t10k-labels-idx1-ubyte.gz', url_path+'t10k-images-idx3-ubyte.gz')
    return {'train_data':train_img, 'train_label':train_lbl,
            'test_data':test_img, 'test_label':test_lbl}

def get_mnist_ubyte(path='data'):
    """Downloads ubyte version of the MNIST dataset into a directory in the current directory
    with the name `data` and extracts all files in the zip archive to this directory.
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    files = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
             't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
    if not all(os.path.exists(os.path.join(path, f)) for f in files):
        get_mnist(path)
        for f in files:
            ubyte_file_path = os.path.join(path, f)
            zip_file_path = ubyte_file_path + '.gz'
            with gzip.GzipFile(zip_file_path) as zf:
                with open(ubyte_file_path, 'wb') as ubyte_file:
                    ubyte_file.write(zf.read())

def get_cifar10(path='data'):
    """Downloads CIFAR10 dataset into a directory in the current directory with the name `data`,
    and then extracts all files into the directory `data/cifar`.
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    if (not os.path.exists(os.path.join(path, 'cifar', 'train.rec'))) or \
            (not os.path.exists(os.path.join(path, 'cifar', 'test.rec'))) or \
            (not os.path.exists(os.path.join(path, 'cifar', 'train.lst'))) or \
            (not os.path.exists(os.path.join(path, 'cifar', 'test.lst'))):
        url = 'https://repo.mxnet.io/gluon/dataset/cifar10/cifar10-b9ac2870.zip'
        sha1 = 'b9ac287012f2dad9dfb49d8271c39ecdd7db376c'
        zip_file_path = mx.gluon.utils.download(url, path=path, sha1_hash=sha1,
                                                verify_ssl=False)
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(path)

def get_mnist_iterator(batch_size, input_shape, num_parts=1, part_index=0, path='data'):
    """Returns training and validation iterators for MNIST dataset
    """

    get_mnist_ubyte(path)
    flat = len(input_shape) != 3

    train_dataiter = mx.io.MNISTIter(
        image=os.path.join(path, "train-images-idx3-ubyte"),
        label=os.path.join(path, "train-labels-idx1-ubyte"),
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat,
        num_parts=num_parts,
        part_index=part_index)

    val_dataiter = mx.io.MNISTIter(
        image=os.path.join(path, "t10k-images-idx3-ubyte"),
        label=os.path.join(path, "t10k-labels-idx1-ubyte"),
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat,
        num_parts=num_parts,
        part_index=part_index)

    return (train_dataiter, val_dataiter)

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
        raise AssertionError(f"Generator test fails, Chi-square p={str(cs_ret_l)}, "
                             f"obs_freq={str(obs_freq_l)}, expected_freq={str(expected_freq_l)}."
                             f"\nbuckets={str(buckets)}, probs={str(probs)}")
    return cs_ret_l


def compare_ndarray_tuple(t1, t2, rtol=None, atol=None):
    """Compare ndarray tuple."""
    if t1 is None or t2 is None:
        return

    if isinstance(t1, tuple):
        for s1, s2 in zip(t1, t2):
            compare_ndarray_tuple(s1, s2, rtol, atol)
    else:
        assert_almost_equal(t1, t2, rtol=rtol, atol=atol)


def compare_optimizer(opt1, opt2, shapes, dtype, w_stype='default', g_stype='default',
                      rtol=1e-4, atol=1e-5, compare_states=True):
    """Compare opt1 and opt2."""

    w1_list, w2_list = [], []
    g1_list, g2_list = [], []
    s1_list, s2_list = [], []
    for i, shape in enumerate(shapes):
        if w_stype == 'default':
            w2 = mx.random.uniform(shape=shape, ctx=default_device(), dtype=dtype)
            w1 = w2.copyto(default_device())
        elif w_stype in ('row_sparse', 'csr'):
            w2 = rand_ndarray(shape, w_stype, density=1, dtype=dtype)
            w1 = w2.copyto(default_device()).tostype('default')
        else:
            raise Exception("type not supported yet")
        if g_stype == 'default':
            g2 = mx.random.uniform(shape=shape, ctx=default_device(), dtype=dtype)
            g1 = g2.copyto(default_device())
        elif g_stype in ('row_sparse', 'csr'):
            g2 = rand_ndarray(shape, g_stype, dtype=dtype)
            g1 = g2.copyto(default_device()).tostype('default')
        else:
            raise Exception("type not supported yet")
        s1 = opt1.create_state_multi_precision(i, w1)
        s2 = opt2.create_state_multi_precision(i, w2)

        if compare_states:
            compare_ndarray_tuple(s1, s2)

        w1_list.append(w1)
        w2_list.append(w2)
        g1_list.append(g1)
        g2_list.append(g2)
        s1_list.append(s1)
        s2_list.append(s2)

    indices = list(range(len(shapes)))
    opt1.update_multi_precision(indices, w1_list, g1_list, s1_list)
    opt2.update_multi_precision(indices, w2_list, g2_list, s2_list)
    if compare_states:
        compare_ndarray_tuple(tuple(s1_list), tuple(s2_list), rtol=rtol, atol=atol)
    compare_ndarray_tuple(tuple(w1_list), tuple(w2_list), rtol=rtol, atol=atol)


def compare_optimizer_noise_seeded(opt1, opt2, shapes, dtype, noise_seed,
                                   w_stype='default', g_stype='default',
                                   rtol=1e-4, atol=1e-5, compare_states=True):
    """Compare opt1 and opt2 with the added functionality that the seed for generating random noise
    in the SGLD optimizer update is set so that the same noise is used in opt1 and opt2.

    """
    w1_list, w2_list = [], []
    g1_list, g2_list = [], []
    s1_list, s2_list = [], []
    for i, shape in enumerate(shapes):
        if w_stype == 'default':
            w2 = mx.random.uniform(shape=shape, ctx=default_device(), dtype=dtype)
            w1 = w2.copyto(default_device())
        elif w_stype in ('row_sparse', 'csr'):
            w2 = rand_ndarray(shape, w_stype, density=1, dtype=dtype)
            w1 = w2.copyto(default_device()).tostype('default')
        else:
            raise Exception("type not supported yet")
        if g_stype == 'default':
            g2 = mx.random.uniform(shape=shape, ctx=default_device(), dtype=dtype)
            g1 = g2.copyto(default_device())
        elif g_stype in ('row_sparse', 'csr'):
            g2 = rand_ndarray(shape, g_stype, dtype=dtype)
            g1 = g2.copyto(default_device()).tostype('default')
        else:
            raise Exception("type not supported yet")
        s1 = opt1.create_state_multi_precision(i, w1)
        s2 = opt2.create_state_multi_precision(i, w2)

        if compare_states:
            compare_ndarray_tuple(s1, s2)

        w1_list.append(w1)
        w2_list.append(w2)
        g1_list.append(g1)
        g2_list.append(g2)
        s1_list.append(s1)
        s2_list.append(s2)

    indices = list(range(len(shapes)))
    # set seed for Gaussian noise replication
    mx.random.seed(noise_seed)
    opt1.update_multi_precision(indices, w1_list, g1_list, s1_list)
    mx.random.seed(noise_seed)
    opt2.update_multi_precision(indices, w2_list, g2_list, s2_list)
    if compare_states:
        compare_ndarray_tuple(tuple(s1_list), tuple(s2_list), rtol=rtol, atol=atol)
    compare_ndarray_tuple(tuple(w1_list), tuple(w2_list), rtol=rtol, atol=atol)


def same_symbol_structure(sym1, sym2):
    """Compare two symbols to check if they have the same computation graph structure.
    Returns true if operator corresponding to a particular node id is same in both
    symbols for all nodes
    """
    conf = json.loads(sym1.tojson())
    nodes = conf["nodes"]
    conf2 = json.loads(sym2.tojson())
    nodes2 = conf2["nodes"]
    for node1, node2 in zip(nodes, nodes2):
        if node1["op"] != node2["op"]:
            return False
    return True


@contextmanager
def environment(*args):
    """
    Environment variable setter and unsetter via `with` idiom.

    Takes a specification of env var names and desired values and adds those
    settings to the environment in advance of running the body of the `with`
    statement.  The original environment state is restored afterwards, even
    if exceptions are raised in the `with` body.

    Parameters
    ----------
    args:
        if 2 args are passed:
            name, desired_value strings of the single env var to update, or
        if 1 arg is passed:
            a dict of name:desired_value for env var's to update

    """

    # On Linux, env var changes made through python's os.environ are seen
    # by the backend.  On Windows though, the C runtime gets a snapshot
    # of the environment that cannot be altered by os.environ.  Here we
    # check, using a wrapped version of the backend's getenv(), that
    # the desired env var value is seen by the backend, and otherwise use
    # a wrapped setenv() to establish that value in the backend.

    # Also on Windows, a set env var can never have the value '', since
    # the command 'set FOO= ' is used to unset the variable.  Perhaps
    # as a result, the wrapped dmlc::GetEnv() routine returns the same
    # value for unset variables and those set to ''.  As a result, we
    # ignore discrepancy.
    def validate_backend_setting(name, value, can_use_setenv=True):
        backend_value = getenv(name)
        if value == backend_value or \
           value == '' and backend_value is None and platform.system() == 'Windows':
            return
        if not can_use_setenv:
            raise RuntimeError('Could not set env var {}={} within C Runtime'.format(name, value))
        setenv(name, value)
        validate_backend_setting(name, value, can_use_setenv=False)

    # Core routine to alter environment from a dict of env_var_name, env_var_value pairs
    def set_environ(env_var_dict):
        for env_var_name, env_var_value in env_var_dict.items():
            if env_var_value is None:
                os.environ.pop(env_var_name, None)
            else:
                os.environ[env_var_name] = env_var_value
            validate_backend_setting(env_var_name, env_var_value)

    # Create env_var name:value dict from the two calling methods of this routine
    if len(args) == 1 and isinstance(args[0], dict):
        env_vars = args[0]
    else:
        assert len(args) == 2, 'Expecting one dict arg or two args: env var name and value'
        env_vars = {args[0]: args[1]}

    # Take a snapshot of the existing environment variable state
    # for those variables to be changed.  get() return None for unset keys.
    snapshot = {x: os.environ.get(x) for x in env_vars.keys()}

    # Alter the environment per the env_vars dict
    set_environ(env_vars)

    # Now run the wrapped code
    try:
        yield
    finally:
        # the backend engines may still be referencing the changed env var state
        mx.nd.waitall()
        # reinstate original env_var state per the snapshot taken earlier
        set_environ(snapshot)


def collapse_sum_like(a, shape):
    """Given `a` as a numpy ndarray, perform reduce_sum on `a` over the axes that do not
    exist in `shape`. Note that an ndarray with `shape` must be broadcastable to `a`.
    """
    assert len(a.shape) >= len(shape)
    if np.prod(shape) == 0 or a.size == 0:
        return np.zeros(shape, dtype=a.dtype)
    axes = []
    ndim_diff = len(a.shape) - len(shape)
    for i in range(ndim_diff):
        axes.append(i)
    for i, s in enumerate(shape):
        if s != a.shape[i+ndim_diff]:
            assert s == 1
            axes.append(i+ndim_diff)
    return np.sum(a, axis=tuple(axes)).reshape(shape)


def is_cd_run():
    """Checks if the test is running as part of a Continuous Delivery run"""
    return os.environ.get("CD_JOB", 0) == "1"


_features = Features()


def has_tvm_ops():
    """Returns True if MXNet is compiled with TVM generated operators. If current ctx
    is GPU, it only returns True for CUDA compute capability > 52 where FP16 is supported.
    """
    built_with_tvm_op = _features.is_enabled("TVM_OP")
    device = current_device()
    if device.device_type == 'gpu':
        try:
            cc = get_cuda_compute_capability(device)
        except:  # pylint: disable=bare-except
            print('Failed to get CUDA compute capability for context {}. The operators '
                  'built with USE_TVM_OP=1 will not be run in unit tests.'.format(device))
            return False
        print('Cuda arch compute capability: sm_{}'.format(str(cc)))
        return built_with_tvm_op and cc >= 53
    return built_with_tvm_op


def is_op_runnable():
    """Returns True for all CPU tests. Returns True for GPU tests that are either of the following.
    1. Built with USE_TVM_OP=0.
    2. Built with USE_TVM_OP=1, but with compute capability >= 53.
    """
    device = current_device()
    if device.device_type == 'gpu':
        if not _features.is_enabled("TVM_OP"):
            return True
        else:
            try:
                cc = get_cuda_compute_capability(device)
            except:  # pylint: disable=bare-except
                print('Failed to get CUDA compute capability for context {}. The operators '
                      'built with USE_TVM_OP=1 will not be run in unit tests.'.format(device))
                return False
            print('Cuda arch compute capability: sm_{}'.format(str(cc)))
            return cc >= 53
    return True


@use_np
def check_gluon_hybridize_consistency(net_builder, data_l, numpy_func=None, test_grad=True,
                                      rtol=1E-4, atol=1E-4):
    """Check whether a HybridBlock has consistent output when hybridized or not hybridized

    The network should not contain any random number generators.

    Parameters
    ----------
    net_builder : function
        The builder of the HybridBlock that we are going to check the consistency.
        Inside the implementation, we will call net_builder() to construct the hybrid block.
        Also, the net_builder will need to support specifying the params
    data_l : list of mx.np.ndarray
        List of input ndarrays.
    numpy_func : function, optional
        The ground truth numpy function that has the same functionality as net_builder().
        Default None.
    test_grad : bool, optional
        Whether to test the consistency of the gradient. Default True.
    rtol : float, optional
        The relative error tolerance, default 1E-4. Default 1E-4.
    atol : float, optional
        The absolute error tolerance, default 1E-4. Default 1E-4.
    """
    saved_out_np = None
    saved_grad_np_l = None
    params_init = None
    use_autograd_flags = [False, True] if test_grad else [False]
    for hybridize in [False, True]:
        for use_autograd in use_autograd_flags:
            net = net_builder()
            if params_init is None:
                net.initialize()
            else:
                net.load_dict(params_init)
            if hybridize:
                net.hybridize()
            in_data_l = [ele.copy() for ele in data_l]
            if use_autograd:
                for ele in in_data_l:
                    ele.attach_grad()
                with mx.autograd.record():
                    out = net(*in_data_l)
                out.backward(out)
            else:
                out = net(*in_data_l)
            if params_init is None:  # Deferred initialization finished
                params_init = {k: v.data().asnumpy() for k, v in net.collect_params().items()}
            if saved_out_np is None:
                saved_out_np = out.asnumpy()
            else:
                # Check for correctness
                assert_almost_equal(out.asnumpy(), saved_out_np, rtol=rtol, atol=atol)
            if use_autograd:
                if saved_grad_np_l is None:
                    saved_grad_np_l = [ele.grad.asnumpy() for ele in in_data_l]
                else:
                    # Check for correctness
                    for data, saved_grad_np in zip(in_data_l, saved_grad_np_l):
                        assert_almost_equal(data.grad.asnumpy(), saved_grad_np,
                                            rtol=rtol, atol=atol)
    if numpy_func is not None:
        numpy_out = numpy_func(*[ele.asnumpy() for ele in data_l])
        assert_almost_equal(saved_out_np, numpy_out, rtol=rtol, atol=atol)


def new_matrix_with_real_eigvals_2d(n):
    """Generate a well-conditioned matrix with small real eigenvalues."""
    shape = (n, n)
    q = np.ones(shape)
    while 1:
        D = np.diag(np.random.uniform(-1.0, 1.0, shape[-1]))
        I = np.eye(shape[-1]).reshape(shape)
        v = np.random.uniform(-1., 1., shape[-1]).reshape(shape[:-1] + (1,))
        v = v / np.linalg.norm(v, axis=-2, keepdims=True)
        v_T = np.swapaxes(v, -1, -2)
        U = I - 2 * np.matmul(v, v_T)
        q = np.matmul(U, D)
        if (np.linalg.cond(q, 2) < 3):
            break
    D = np.diag(np.random.uniform(-10.0, 10.0, n))
    q_inv = np.linalg.inv(q)
    return np.matmul(np.matmul(q_inv, D), q)


def new_matrix_with_real_eigvals_nd(shape):
    """Generate well-conditioned matrices with small real eigenvalues."""
    n = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    return np.array([new_matrix_with_real_eigvals_2d(shape[-1]) for i in range(n)]).reshape(shape)


def new_orthonormal_matrix_2d(n):
    """Generate a orthonormal matrix."""
    x = np.random.randn(n, n)
    x_trans = x.T
    sym_mat = np.matmul(x_trans, x)
    return np.linalg.qr(sym_mat)[0]


def new_sym_matrix_with_real_eigvals_2d(n):
    """Generate a sym matrix with real eigenvalues."""
    q = new_orthonormal_matrix_2d(n)
    D = np.diag(np.random.uniform(-10.0, 10.0, n))
    return np.matmul(np.matmul(q.T, D), q)


def new_sym_matrix_with_real_eigvals_nd(shape):
    """Generate sym matrices with real eigenvalues."""
    n = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    return np.array([new_sym_matrix_with_real_eigvals_2d(shape[-1]) for i in range(n)]).reshape(shape)
