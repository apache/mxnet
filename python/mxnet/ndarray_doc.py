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

# coding: utf-8
# pylint: disable=unused-argument, too-many-arguments, unnecessary-pass
"""Extra symbol documents"""
from __future__ import absolute_import as _abs
import re as _re

from .base import build_param_doc as _build_param_doc

class NDArrayDoc(object):
    """The basic class"""
    pass

class ReshapeDoc(NDArrayDoc):
    """
    Examples
    --------

    Reshapes the input array into a new shape.

    >>> x = mx.nd.array([1, 2, 3, 4])
    >>> y = mx.nd.reshape(x, shape=(2, 2))
    >>> x.shape
    (4L,)
    >>> y.shape
    (2L, 2L)
    >>> y.asnumpy()
    array([[ 1.,  2.],
           [ 3.,  4.]], dtype=float32)

    You can use ``0`` to copy a particular dimension from the input to the output shape
    and '-1' to infer the dimensions of the output.

    >>> x = mx.nd.ones((2, 3, 4))
    >>> x.shape
    (2L, 3L, 4L)
    >>> y = mx.nd.reshape(x, shape=(4, 0, -1))
    >>> y.shape
    (4L, 3L, 2L)
    """

class concatDoc(NDArrayDoc):
    """
    Examples
    --------

    Joins input arrays along a given axis.

    >>> x = mx.nd.array([[1,1],[2,2]])
    >>> y = mx.nd.array([[3,3],[4,4],[5,5]])
    >>> z = mx.nd.array([[6,6], [7,7],[8,8]])

    >>> mx.nd.concat(x,y,z,dim=0)
    [[1. 1.]
    [2. 2.]
    [3. 3.]
    [4. 4.]
    [5. 5.]
    [6. 6.]
    [7. 7.]
    [8. 8.]]
    <NDArray 8x2 @cpu(0)>

    >>> mx.nd.concat(y,z,dim=1)
    [[3. 3. 6. 6.]
    [4. 4. 7. 7.]
    [5. 5. 8. 8.]]
    <NDArray 3x4 @cpu(0)>
    """

class SwapAxisDoc(NDArrayDoc):
    """
    Examples
    --------

    Interchanges two axes of an array.

    >>> x = mx.nd.array([[1, 2, 3]])
    >>> mx.nd.swapaxes(x, 0, 1)
    [[1.]
    [2.]
    [3.]]

    >>> x = mx.nd.array([[[ 0, 1],[ 2, 3]],[[ 4, 5],[ 6, 7]]])
    >>> x
    [[[0. 1.]
      [2. 3.]]
    [[4. 5.]
      [6. 7.]]]
    <NDArray 2x2x2 @cpu(0)>
    >>> mx.nd.swapaxes(x, 0, 2)
    [[[0. 4.]
     [2. 6.]]
    [[1. 5.]
      [3. 7.]]]
    <NDArray 2x2x2 @cpu(0)>
    """

class whereDoc(NDArrayDoc):
    """
    Examples
    --------

    Return the elements, either from x or y, depending on the condition.

    >>> x = mx.nd.array([[1, 2], [3, 4]])
    >>> y = mx.nd.array([[5, 6], [7, 8]])

    >>> cond = mx.nd.array([[0, 1], [-1, 0]])
    >>> mx.nd.where(cond, x, y)
    [[5. 2.]
    [3. 8.]]
    <NDArray 2x2 @cpu(0)>

    >>> csr_cond = mx.nd.sparse.cast_storage(cond, 'csr')
    >>> mx.nd.sparse.where(csr_cond, x, y)
    [[5. 2.]
    [3. 8.]]
    <NDArray 2x2 @cpu(0)>
    """

class ReshapeLikeDoc(NDArrayDoc):
    """
    Reshape some or all dimensions of `lhs` to have the same shape as some or all dimensions of `rhs`.
    Example
    -------

    >>> x = mx.nd.array([1, 2, 3, 4, 5, 6])
    >>> y = mx.nd.array([[0, -4], [3, 2], [2, 2]])
    >>> mx.nd.reshape_like(x, y)
    [[1. 2.]
    [3. 4.]
    [5. 6.]]
    <NDArray 3x2 @cpu(0)>
    """

class shape_arrayDoc(NDArrayDoc):
    """
    Returns a 1D int64 array containing the shape of data.
    Example
    -------

    >>> x = mx.nd.array([[1,2,3,4], [5,6,7,8]])
    >>> mx.nd.shape_array(x)
    [2 4]
    <NDArray 2 @cpu(0)>
    """

class size_arrayDoc(NDArrayDoc):
    """
    Returns a 1D int64 array containing the size of data.
    Example
    -------

    >>> x = mx.nd.array([[1,2,3,4], [5,6,7,8]])
    >>> mx.nd.size_array(x)
    [8]
    <NDArray 1 @cpu(0)>
    """

class CastDoc(NDArrayDoc):
    """
    Casts all elements of the input to a new type.
    Example
    -------

    >>> x = mx.nd.array([0.9, 1.3])
    >>> mx.nd.cast(x, dtype='int32')
    [0 1]
    <NDArray 2 @cpu(0)>

    >>> x = mx.nd.array([1e20, 11.1])
    >>> mx.nd.cast(x, dtype='float16')
    [ inf 11.1]
    <NDArray 2 @cpu(0)>

    >>> x = mx.nd.array([300, 11.1, 10.9, -1, -3])
    >>> mx.nd.cast(x, dtype='uint8')
    [ 44  11  10 255 253]
    <NDArray 5 @cpu(0)>
    """

class reciprocalDoc(NDArrayDoc):
    """
    Returns the reciprocal of the argument, element-wise.
    Example
    -------

    >>> x = mx.nd.array([-2, 1, 3, 1.6, 0.2])
    >>> mx.nd.reciprocal(x)
    [-0.5         1.          0.33333334  0.625       5.        ]
    <NDArray 5 @cpu(0)>
    """

class absDoc(NDArrayDoc):
    """
    Returns element-wise absolute value of the input.
    Example
    -------

    >>> x = mx.nd.array([-2, 0, 3])
    >>> mx.nd.abs(x)
    [2. 0. 3.]
    <NDArray 3 @cpu(0)>
    """

class signDoc(NDArrayDoc):
    """
    Returns element-wise sign of the input.
    Example
    -------

    >>> x = mx.nd.array([-2, 0, 3])
    >>> mx.nd.sign(x)
    [-1.  0.  1.]
    <NDArray 3 @cpu(0)>
    """

class roundDoc(NDArrayDoc):
    """
    Returns element-wise rounded value to the nearest integer of the input.
    Example
    -------

    >>> x = mx.nd.array([-2.1, -1.9, 1.5, 1.9, 2.1])
    >>> mx.nd.round(x)
    [-2. -2.  2.  2.  2.]
    <NDArray 5 @cpu(0)>
    """

class rintDoc(NDArrayDoc):
    """
    Returns element-wise rounded value to the nearest integer of the input.
    Example
    -------

    >>> x = mx.nd.array([-2.1, -1.9, 1.5, 1.9, 2.1])
    >>> mx.nd.rint(x)
    [-2. -2.  1.  2.  2.]
    <NDArray 5 @cpu(0)>
    """

class ceilDoc(NDArrayDoc):
    """
    Returns element-wise ceiling of the input.
    Example
    -------

    >>> x = mx.nd.array([-2.1, -1.9, 1.5, 1.9, 2.1])
    >>> mx.nd.ceil(x)
    [-2. -1.  2.  2.  3.]
    <NDArray 5 @cpu(0)>
    """

class floorDoc(NDArrayDoc):
    """
    Returns element-wise floor of the input.
    Example
    -------

    >>> x = mx.nd.array([-2.1, -1.9, 1.5, 1.9, 2.1])
    >>> mx.nd.floor(x)
    [-3. -2.  1.  1.  2.]
    <NDArray 5 @cpu(0)>
    """

class truncDoc(NDArrayDoc):
    """
    Return the element-wise truncated value of the input.
    Example
    -------

    >>> x = mx.nd.array([-2.1, -1.9, 1.5, 1.9, 2.1])
    >>> mx.nd.trunc(x)
    [-2. -1.  1.  1.  2.]
    <NDArray 5 @cpu(0)>
    """

class zeros_likeDoc(NDArrayDoc):
    """
    Return an array of zeros with the same shape, type and storage type
    Example
    -------

    >>> x = mx.nd.array([[ 1.,  1.,  1.],[ 1.,  1.,  1.]])
    >>> x
    [[1. 1. 1.]
    [1. 1. 1.]]
    <NDArray 2x3 @cpu(0)>
    >>> mx.nd.zeros_like(x)
    [[0. 0. 0.]
    [0. 0. 0.]]
    <NDArray 2x3 @cpu(0)>
    """

class unravel_indexDoc(NDArrayDoc):
    """
    Converts an array of flat indices into a batch of index arrays.
    The operator follows numpy conventions so a single multi index is given by a column of the output matrix.
    Example
    -------

    >>> a = mx.nd.array([22,41,37])
    >>> mx.nd.unravel_index(a, shape=(7,6))
    [[3. 6. 6.]
    [4. 5. 1.]]
    <NDArray 2x3 @cpu(0)>
    """

class elemwise_addDoc(NDArrayDoc):
    """
    Example
    -------

    >>> x = mx.nd.array([1, 2, 3, 4])
    >>> y = mx.nd.array([1.1, 2.1, 3.1, 4.1])
    >>> mx.nd.elemwise_add(x, y).asnumpy()
    array([ 2.0999999 ,  4.0999999 ,  6.0999999 ,  8.10000038], dtype=float32)
    """

class BroadcastToDoc(NDArrayDoc):
    """
    Examples
    --------

    Broadcasts the input array into a new shape.

    >>> a = mx.nd.array(np.arange(6).reshape(6,1))
    >>> b = a.broadcast_to((6,2))
    >>> a.shape
    (6L, 1L)
    >>> b.shape
    (6L, 2L)
    >>> b.asnumpy()
    array([[ 0.,  0.],
       [ 1.,  1.],
       [ 2.,  2.],
       [ 3.,  3.],
       [ 4.,  4.],
       [ 5.,  5.]], dtype=float32)
    Broadcasts along axes 1 and 2.
    >>> c = a.reshape((2,1,1,3))
    >>> d = c.broadcast_to((2,2,2,3))
    >>> d.asnumpy()
    array([[[[ 0.,  1.,  2.],
         [ 0.,  1.,  2.]],

        [[ 0.,  1.,  2.],
         [ 0.,  1.,  2.]]],


       [[[ 3.,  4.,  5.],
         [ 3.,  4.,  5.]],

        [[ 3.,  4.,  5.],
         [ 3.,  4.,  5.]]]], dtype=float32)
    >>> c.shape
    (2L, 1L, 1L, 3L)
    >>> d.shape
    (2L, 2L, 2L, 3L)
    """

class StackDoc(NDArrayDoc):
    """
    Example
    --------

    Join a sequence of arrays along a new axis.

    >>> x = mx.nd.array([1, 2])
    >>> y = mx.nd.array([3, 4])
    >>> stack(x, y)
    [[1, 2],
     [3, 4]]
    >>> stack(x, y, axis=1)
    [[1, 3],
     [2, 4]]
    """

class CustomDoc(NDArrayDoc):
    """
    Example
    -------

    Applies a custom operator named `my_custom_operator` to `input`.

    >>> output = mx.symbol.Custom(op_type='my_custom_operator', data=input)
    """

def _build_doc(func_name,
               desc,
               arg_names,
               arg_types,
               arg_desc,
               key_var_num_args=None,
               ret_type=None):
    """Build docstring for imperative functions."""
    param_str = _build_param_doc(arg_names, arg_types, arg_desc)
    # if key_var_num_args:
    #     desc += '\nThis function support variable length of positional input.'
    doc_str = ('%s\n\n' +
               '%s\n' +
               'out : NDArray, optional\n' +
               '    The output NDArray to hold the result.\n\n'+
               'Returns\n' +
               '-------\n' +
               'out : NDArray or list of NDArrays\n' +
               '    The output of this function.')
    doc_str = doc_str % (desc, param_str)
    extra_doc = "\n" + '\n'.join([x.__doc__ for x in type.__subclasses__(NDArrayDoc)
                                  if x.__name__ == '%sDoc' % func_name])
    doc_str += _re.sub(_re.compile("    "), "", extra_doc)
    doc_str = _re.sub('NDArray-or-Symbol', 'NDArray', doc_str)

    return doc_str
