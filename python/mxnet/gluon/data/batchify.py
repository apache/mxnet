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
# pylint: disable=reimported, consider-using-enumerate
"""Batchify function."""
import math
import warnings
import numpy as np

from ...device import Device, cpu
from ... import ndarray as nd
from ... import numpy as _np
from ...util import is_np_array

class Stack(object):
    r"""Stack the input data samples to construct the batch.
    The N input samples must have the same shape/length and will be stacked to construct a batch.
    Examples
    --------
    >>> from mxnet.gluon.data import batchify
    >>> # Stack multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6, 8]
    >>> c = [8, 9, 1, 2]
    >>> batchify.Stack()([a, b, c])
    [[1. 2. 3. 4.]
     [4. 5. 6. 8.]
     [8. 9. 1. 2.]]
    <NDArray 3x4 @cpu(0)>
    >>> # Stack multiple numpy.ndarrays
    >>> import numpy as np
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> batchify.Stack()([a, b])
    [[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
     [[5. 6. 7. 8.]
      [1. 2. 3. 4.]]]
    <NDArray 2x2x4 @cpu(0)>
    >>> # Stack multiple NDArrays
    >>> import mxnet as mx
    >>> a = nd.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = nd.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> batchify.Stack()([a, b])
    [[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
     [[5. 6. 7. 8.]
      [1. 2. 3. 4.]]]
    <NDArray 2x2x4 @cpu(0)>
    """
    def __init__(self, use_shared_mem=False):
        self._use_shared_mem = use_shared_mem

    def __call__(self, data):
        """Batchify the input data
        Parameters
        ----------
        data : list
            The input data samples
        Returns
        -------
        batch_data : NDArray
        """
        _arr = _np if is_np_array() else nd
        _arr_cls = _arr.ndarray if is_np_array() else _arr.NDArray
        if isinstance(data[0], _arr_cls):
            dtype = data[0].dtype
            if self._use_shared_mem:
                out = _arr.empty((len(data),) + data[0].shape, dtype=dtype,
                                 ctx=Device('cpu_shared', 0))
                return _arr.stack(data, out=out) if is_np_array() else _arr.stack(*data, out=out)
            else:
                return _arr.stack(data) if is_np_array() else _arr.stack(*data)
        elif isinstance(data[0], (tuple, list)):
            data = zip(*data)
            return [self.__call__(i) for i in data]
        else:
            out = np.asarray(data)
            dtype = out.dtype
            if self._use_shared_mem:
                return _arr.array(out, ctx=Device('cpu_shared', 0), dtype=dtype)
            else:
                return _arr.array(out, dtype=dtype)

    def __mx_handle__(self):
        from ._internal import StackBatchify
        return StackBatchify()

def _pad_arrs_to_max_length(arrs, pad_val, use_shared_mem, dtype, round_to=None):
    """Inner Implementation of the Pad batchify
    Parameters
    ----------
    arrs : list
    pad_val : number
    use_shared_mem : bool, default False
    round_to : int

    Returns
    -------
    ret : NDArray
    """
    _arr = _np if is_np_array() else nd
    _arr_cls = _np.ndarray if is_np_array() else nd.NDArray
    if isinstance(arrs[0], _arr_cls):
        dtype = arrs[0].dtype if dtype is None else dtype
        arrs = [arr.asnumpy() for arr in arrs]
    elif not isinstance(arrs[0], np.ndarray):
        arrs = [np.asarray(ele) for ele in arrs]
        dtype = arrs[0][0].dtype if dtype is None else dtype
    else:
        dtype = arrs[0].dtype if dtype is None else dtype

    ret_shape = list(arrs[0].shape)
    for pad_axis in range(len(ret_shape)):
        curr_lengths = [ele.shape[pad_axis] for ele in arrs]
        max_size = max(curr_lengths)
        if round_to is not None:
            max_size = round_to * math.ceil(max_size / round_to)
        ret_shape[pad_axis] = max_size
    ret_shape = (len(arrs), ) + tuple(ret_shape)

    ret = np.full(shape=ret_shape, fill_value=pad_val, dtype=dtype)

    for i, arr in enumerate(arrs):
        if arr.shape == ret_shape[1:]:
            ret[i] = arr
        else:
            slices = [slice(None) for _ in range(arr.ndim)]
            for pad_axis in range(arr.ndim):
                slices[pad_axis] = slice(0, arr.shape[pad_axis])
                assert slices[pad_axis].start != slices[pad_axis].stop
            slices = [slice(i, i + 1)] + slices
            ret[tuple(slices)] = arr


    device = Device('cpu_shared', 0) if use_shared_mem else cpu()
    ret = _arr.array(ret, ctx=device, dtype=dtype)

    return ret


class Pad(object):
    """Pad the input ndarrays along the specific padding axis and stack them to get the output.
    Input of the function will be N samples. Each sample should contain a single element that
    can be 1) numpy.ndarray, 2) mxnet.nd.NDArray, 3) list of numbers.
    You can set the `pad_val` to determine the padding value.

    The arrays will be padded to the largest dimensions(at most 5 dimensions to pad) and then
    stacked to form the final output.

    Parameters
    ----------
    val : float or int, default None
        The padding value.
    dtype : str or numpy.dtype, default None
        The value type of the output. If it is set to None, the input data type is used.
    round_to : int, default None
        If specified, the padded dimension will be rounded to be multiple of this argument.

    Examples
    --------
    >>> from mxnet.gluon.data import batchify
    >>> # Inputs are multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> batchify.Pad()([a, b, c])
    [[ 1  2  3  4]
     [ 4  5  6  0]
     [ 8  2  0  0]]
    <NDArray 3x4 @cpu(0)>
    >>> # Also output the lengths
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> # Inputs are multiple ndarrays
    >>> import numpy as np
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 8], [1, 2]])
    >>> batchify.Pad(val=-1)([a, b])
    [[[ 1  2  3  4]
      [ 5  6  7  8]]
     [[ 5  8 -1 -1]
      [ 1  2 -1 -1]]]
    <NDArray 2x2x4 @cpu(0)>
    >>> # Inputs are multiple NDArrays
    >>> import mxnet as mx
    >>> a = nd.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = nd.array([[5, 8], [1, 2]])
    >>> batchify.Pad(val=-1)([a, b])
    [[[ 1.  2.  3.  4.]
      [ 5.  6.  7.  8.]]
     [[ 5.  8. -1. -1.]
      [ 1.  2. -1. -1.]]]
    <NDArray 2x2x4 @cpu(0)>
    """
    def __init__(self, val=None, dtype=None, round_to=None, use_shared_mem=False):
        self._pad_val = 0 if val is None else val
        self._dtype = dtype
        self._warned = False
        self._round_to = round_to
        self._use_shared_mem = use_shared_mem

    def __call__(self, data):
        """Batchify the input data.

        The input can be list of numpy.ndarray, list of numbers or list of
        mxnet.nd.NDArray. Inputting mxnet.nd.NDArray is discouraged as each
        array need to be converted to numpy for efficient padding.
        The arrays will be padded to the largest dimension at `axis` and then
        stacked to form the final output.

        Parameters
        ----------
        data : List[np.ndarray] or List[List[dtype]] or List[nd.NDArray]
            List of samples to pad and stack.
        Returns
        -------
        batch_data: NDArray
            Data in the minibatch. Shape is (N, ...)
        """
        _arr = _np if is_np_array() else nd
        _arr_cls = _arr.ndarray if is_np_array() else _arr.NDArray
        if isinstance(data[0], _arr_cls) and not self._warned:
            self._warned = True
            warnings.warn(
                'Using Pad with NDArrays is discouraged for speed reasons. '
                'Instead you should pad your data while it is still a list '
                'and before converting to an NDArray. '
                'Alternatively you can consider inputting a numpy.ndarray.')
        if isinstance(data[0], (_arr_cls, np.ndarray, list)):
            padded_arr = _pad_arrs_to_max_length(data, self._pad_val,
                                                 self._use_shared_mem,
                                                 self._dtype, self._round_to)
            return padded_arr
        else:
            raise NotImplementedError(
                "Pad() does not support multiple items, use Group(Pad(), Pad(), ...) instead")

    def __mx_handle__(self):
        from ._internal import PadBatchify
        return PadBatchify(pad_val=self._pad_val, dtype=self._dtype if self._dtype is not None else -1)

def _append_arrs(arrs, use_shared_mem=False, expand=False, batch_axis=0):
    """Internal impl for returning appened arrays as list."""
    _arr = _np if is_np_array() else nd
    if isinstance(arrs[0], _arr.NDArray):
        if use_shared_mem:
            out = [x.as_in_context(Device('cpu_shared', 0)) for x in arrs]
        else:
            out = arrs
    else:
        if use_shared_mem:
            out = [_arr.array(x, ctx=Device('cpu_shared', 0)) for x in arrs]
        else:
            out = [_arr.array(x) for x in arrs]

    # add batch axis
    if expand:
        out = [x.expand_dims(axis=batch_axis) for x in out]
    return out


class Append(object):
    r"""Loosely return list of the input data samples.
    There is no constraint of shape for any of the input samples, however, you will
    only be able to apply single batch operations since the output have different shapes.
    Examples
    --------
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> batchify.Append()([a, b, c])
    [
    [[1. 2. 3. 4.]]
    <NDArray 1x4 @cpu_shared(0)>,
    [[4. 5. 6.]]
    <NDArray 1x3 @cpu_shared(0)>,
    [[8. 2.]]
    <NDArray 1x2 @cpu_shared(0)>
    ]
    """

    def __init__(self, expand=True, batch_axis=0, use_shared_mem=False):
        self._expand = expand
        self._batch_axis = batch_axis
        self._use_shared_mem = use_shared_mem

    def __call__(self, data):
        """Batchify the input data.
        Parameters
        ----------
        data : list
            The input data samples
        Returns
        -------
        batch_data : NDArray
        """
        return _append_arrs(data, use_shared_mem=self._use_shared_mem,
                            expand=self._expand, batch_axis=self._batch_axis)

class Group(object):
    """Wrap multiple batchify functions together. The input functions will be applied
    to the corresponding input fields.
    Each data sample should be a list or tuple containing multiple attributes. The `i`th batchify
    function stored in `Group` will be applied on the `i`th attribute. For example, each
    data sample is (nd_data, label). You can wrap two batchify functions using
    `Group(DataBatchify, LabelBatchify)` to batchify nd_data and label correspondingly.
    Parameters
    ----------
    fn : list or tuple or callable
        The batchify functions to wrap.
    *args : tuple of callable
        The additional batchify functions to wrap.
    Examples
    --------
    >>> a = ([1, 2, 3, 4], 0)
    >>> b = ([5, 7], 1)
    >>> c = ([1, 2, 3, 4, 5, 6, 7], 0)
    >>> f1, f2 = Group(Pad(val=0),
    ...                Stack())([a, b])
    >>> f1
    <BLANKLINE>
    [[1. 2. 3. 4.]
     [5. 7. 0. 0.]]
    <NDArray 2x4 @cpu_shared(0)>
    >>> f2
    <BLANKLINE>
    [0 1]
    <NDArray 2 @cpu_shared(0)>
    """
    def __init__(self, fn, *args):
        self._handle = None
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Group can be ' \
                                   'Group(A, B, C) or Group([A, B, C]) or Group((A, B, C)). ' \
                                   f'Received fn={str(fn)}, args={str(args)}'
            self._fn = fn
        else:
            self._fn = (fn, ) + args
        for i, ele_fn in enumerate(self._fn):
            assert hasattr(ele_fn, '__call__'), 'Batchify functions must be callable! ' \
                                                f'type(fn[{i}]) = {str(type(ele_fn))}'

    def __call__(self, data):
        """Batchify the input data.
        Parameters
        ----------
        data : list
            The samples to batchfy. Each sample should contain N attributes.
        Returns
        -------
        ret : tuple
            A tuple of length N. Contains the batchified result of each attribute in the input.
        """
        assert len(data[0]) == len(self._fn),\
            'The number of attributes in each data sample should contains' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return tuple(ret)

    def __mx_handle__(self):
        if self._handle  is None:
            from ._internal import GroupBatchify
            try:
                mx_fn = [fn.__mx_handle__() for fn in self._fn]
                self._handle = GroupBatchify(functions=mx_fn)
            except Exception as e:
                raise NotImplementedError(
                    "GroupBatchify requires all internal batchify functions supported by backend."
                    + str(e))
        return self._handle

class AsList(object):
    """Simply forward the list of input data.
    This is particularly useful when the Dataset contains textual data
    and in conjonction with the `Group` batchify function.
    Examples
    --------
    >>> a = ([1, 2, 3, 4], "I am using MXNet")
    >>> b = ([5, 7, 2, 5], "Gluon rocks!")
    >>> c = ([1, 2, 3, 4], "Batchification!")
    >>> _, l = Group(Stack(), AsList())([a, b, c])
    >>> l
    ['I am using MXNet', 'Gluon rocks!', 'Batchification!']
    """
    def __call__(self, data):
        """
        Parameters
        ----------
        data : list
            The list of samples
        Returns
        -------
        ret : list
            The input list
        """
        return list(data)
