# coding: utf-8
# pylint: disable=unused-argument, too-many-arguments
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

class CustomDoc(NDArrayDoc):
    """
    Example
    -------
    Applies a custom operator named `my_custom_operator` to `input`.

    >>> output = mx.symbol.Custom(op_type='my_custom_operator', data=input)
    """

class DeconvolutionDoc(NDArrayDoc):
    """
    Examples
    --------
    >>> input_4x4 = mx.nd.normal(shape=[1, 16, 4, 4])
    >>> input_5x5 = mx.nd.normal(shape=[1, 16, 5, 5])
    >>> kernel_3x3 = mx.nd.normal(shape=[16, 16, 3, 3])
    >>>
    >>> # The transpose of convolving a 3 × 3 kernel over a 4 × 4 input using
    ... # unit strides (i.e., i = 4, k = 3, s = 1 and p = 0).
    ...
    >>> conv = mx.nd.Convolution(data=input_4x4, kernel=(3,3), weight=kernel_3x3,
    ... num_filter=16, no_bias=True)
    >>> conv
    <NDArray 1x16x2x2 @cpu(0)>
    >>> transpose = mx.nd.Deconvolution(data=conv, kernel=(3,3), weight=kernel_3x3,
    ... num_filter=16, no_bias=True)
    >>> transpose # should be of the same size as the input to convolution.
    <NDArray 1x16x4x4 @cpu(0)>
    >>>
    >>> # The transpose of convolving a 3×3 kernel over a 5×5 input padded with a
    ... # 1 × 1 border of zeros using 2 × 2 strides (i.e., i = 5, k = 3, s = 2 and p = 1).
    ...
    >>> conv = mx.nd.Convolution(data=input_5x5, kernel=(3,3), weight=kernel_3x3,
    ... stride=(2,2), pad=(1,1), num_filter=16, no_bias=True)
    >>> conv
    <NDArray 1x16x3x3 @cpu(0)>
    >>> transpose = mx.nd.Deconvolution(data=conv, kernel=(3,3), weight=kernel_3x3,
    ... stride=(2,2), pad=(1,1), dilate=(1,1), num_filter=16, no_bias=True)
    >>> transpose # should be of the same size as the input to convolution.
    <NDArray 1x16x5x5 @cpu(0)>
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
