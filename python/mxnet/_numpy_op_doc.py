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

"""Doc placeholder for numpy ops with prefix _np."""


def _np_reshape(a, newshape, order='C'):
    """
    reshape(a, newshape, order='C')

    Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : ndarray
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {'C'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. Other order types such as 'F'/'A'
        may be added in the future.

    Returns
    -------
    reshaped_array : ndarray
        It will be always a copy of the original array. This behavior is different
        from the official NumPy package where views of the original array may be
        generated.

    See Also
    --------
    ndarray.reshape : Equivalent method.
    """
    pass


def _np_ones_like(a):
    """Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.
    """
    pass


def _np_zeros_like(a):
    """Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.
    """
    pass


def _np_repeat(a, repeats, axis=None):
    """Repeat elements of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    repeats : int or array of ints
        The number of repetitions for each element.  `repeats` is broadcasted
        to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values.  By default, use the
        flattened input array, and return a flat output array.

    Returns
    -------
    repeated_array : ndarray
        Output array which has the same shape as `a`, except along
        the given axis.
    """
    pass


def _npi_multinomial(a):
    """Draw samples from a multinomial distribution.

    The multinomial distribution is a multivariate generalisation of the binomial distribution.
    Take an experiment with one of ``p`` possible outcomes. An example of such an experiment is throwing a dice,
    where the outcome can be 1 through 6. Each sample drawn from the distribution represents n such experiments.
    Its values, ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the outcome was ``i``.


    Parameters
    ----------
    n : int
        Number of experiments.
    pvals : sequence of floats, length p
        Probabilities of each of the p different outcomes. These should sum to 1
        (however, the last element is always assumed to account for the remaining
        probability, as long as ``sum(pvals[:-1]) <= 1)``.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k`` sam-
        ples are drawn. Default is None, in which case a single value is returned.

    Returns
    -------
    out : ndarray
        The drawn samples, of shape size, if that was provided. If not, the shape is ``(N,)``.
        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional value drawn from the distribution.
    """
    pass


def _np_cumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See `doc.ufuncs`
        (Section "Output arguments") for more details.

    Returns
    -------
    cumsum_along_axis : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned. The
        result has the same size as `a`, and the same shape as `a` if
        `axis` is not None or `a` is a 1-d array.
    """
    pass


def _np_amax(a, axis=None, out=None):
    """
    amax(a, axis=None, out=None, keepdims=_Null, initial=_Null)
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
        Python native iterables not supported.

    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is
        used.

        Negative indices not supported.

    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `amax` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice.

        Not supported yet.

    Returns
    -------
    amax : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.amax(a)           # Maximum of the flattened array
    array(3)
    >>> np.amax(a, axis=0)   # Maxima along the first axis
    array([2, 3])
    >>> np.amax(a, axis=1)   # Maxima along the second axis
    array([1, 3])

    >>> np.amax(a, axis=1, keepdims=True)
    array([[1],
          [3]])

    Notes
    -----
    This function differs to the original `numpy.amax
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html>`_ in
    the following aspects:
    - The default value type is `float32` instead of `float64` in numpy.
    - `a` only supports ndarray.
    - `axis` doe snot support negative value.
    - `initial` is not supported.

    """
    pass


def _np_squeeze(a, axis=None):
    """
    Remove single-dimensional entries from the shape of an array.
    Parameters
    ----------
    a : ndarray
        Input data.

    axis : None or int or tuple of ints, optional

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`.
    Examples
    --------
    >>> x = np.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> np.squeeze(x).shape
    (3,)
    >>> np.squeeze(x, axis=(2,)).shape
    (1, 3)

    Notes
    -----
    This function differs to the original `numpy.squeeze
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html>`_ in
    the following aspects:
    - The default value type is `float32` instead of `float64` in numpy.
    - `a` only supports ndarray.
    """
    pass


def _npi_ones(shape, dtype=None, order='C'):
    """
    ones(shape, dtype=None, order='C')
    Return a new array of given shape and type, filled with ones.
    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `int`.  Default is
        `float32`.
    order : {'C', 'F'}, optional, default: C
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

        Not Supported yet.
    Returns
    -------
    out : ndarray
        Array of ones with the given shape, dtype, and order.
    Examples
    --------
    >>> np.ones(5)
    array([ 1.,  1.,  1.,  1.,  1.])
    >>> np.ones((5,), dtype=int)
    array([1, 1, 1, 1, 1])
    >>> np.ones((2, 1))
    array([[ 1.],
           [ 1.]])
    >>> s = (2,2)
    >>> np.ones(s)
    array([[ 1.,  1.],
           [ 1.,  1.]])

    Notes
    -----
    This function differs to the original `numpy.ones
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html>`_ in
    the following aspects:
    - The default value type is `float32` instead of `float64` in numpy.
    - `order` is not supported.
    """

def _npi_random_uniform(low=0.0, high=1.0, size=None):
    """
    uniform(low=0.0, high=1.0, size=None)
    Draw samples from a uniform distribution.
    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.
    Parameters
    ----------
    low : float, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float, optional
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.
        Otherwise, ``np.broadcast(low, high).size`` samples are drawn.
    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized uniform distribution.
    Notes
    -----
    The probability density function of the uniform distribution is
    .. math:: p(x) = \frac{1}{b - a}
    anywhere within the interval ``[a, b)``, and zero elsewhere.
    When ``high`` == ``low``, values of ``low`` will be returned.
    If ``high`` < ``low``, the results are officially undefined
    and may eventually raise an error, i.e. do not rely on this
    function to behave when passed arguments satisfying that
    inequality condition.
    Examples
    --------
    Draw samples from the distribution:
     s = np.random.uniform(-1, 0, 1000)
    All values are within the given interval.

    Notes
    -----
    This function differs to the original `numpy.random.uniform
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.uniform.html>`_ in
    the following aspects:
    - The default value type is `float32` instead of `float64` in numpy.
    - `low` and `high` do not accept array-like of floats.
    - `size` is not optional.
    """
    pass
