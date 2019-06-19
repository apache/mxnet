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


def _np_bitwise_and():
    """
    Compute the bit-wise AND of two arrays element-wise.

    Computes the bit-wise AND of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``&``.

    Parameters
    ----------
    x1, x2 : ndarray or boolean
        Only integer and boolean types are handled.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Result.

    See Also
    --------
    bitwise_or
    bitwise_xor

    Notes
    -------
    This function differs from the original `numpy.bitwise_and
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.bitwise_and.html>`_ in
    the following aspects:
    - Input type does not support Python native iterables(list, tuple, ...).


    Examples
    --------
    The number 13 is represented by ``00001101``.  Likewise, 17 is
    represented by ``00010001``.  The bit-wise AND of 13 and 17 is
    therefore ``000000001``, or 1:

    >>> x1 = np.array(13)
    >>> x2 = np.array(17)
    >>> np.bitwise_and(x1, x2)
    array(1.)

    >>> x1 = np.array([True, False, False])
    >>> x2 = np.array([True, True, False])
    >>> np.bitwise_and(x1, x2)
    array([1., 0., 0.])

    Only support ints as input dtype.

    >>> x1 = np.array(13.0)
    >>> x2 = np.array(17.0)
    >>> np.bitwise_and(x1, x2)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
    """
    pass
