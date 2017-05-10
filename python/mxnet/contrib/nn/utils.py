# coding: utf-8
# pylint: disable=
"""Parallelization utility optimizer."""

from ... import ndarray

def split_data(data, num_slice, batch_axis=0, even_split=True):
    """Split a NDArray into num_slice slices along batch_axis.

    Parameters
    ----------
    data : NDArray
        A batch of data.
    num_slice : int
        Number of desired slices.
    batch_axis : int, default 0
        The axis along which to slice.
    even_split : bool, default True
        Whether to force all slices to have the same number of elements.

    Returns
    -------
    list of NDArray
    """
    assert even_split, "Only support even split for now"
    assert not even_split or data.shape[batch_axis] % num_slice == 0, \
        "data with shape %s cannot be evenly split into %d slices along axis %d. " \
        "Use a batch size that's multiple of %d or set even_split=False to enable " \
        "uneven partitioning of data."%(
            str(data.shape), num_slice, batch_axis, num_slice)
    size = data.shape[batch_axis] / num_slice
    if batch_axis == 0:
        slices = [data[i*size:(i+1)*size] for i in range(num_slice)]
    else:
        slices = [ndarray.slice_axis(data, i*size, (i+1)*size)
                  for i in range(num_slice)]
    return slices

def load_data(data, ctx_list, batch_axis=0, even_split=True):
    """Split a NDArray into multiple slices along batch_axis and copy
    each slice into a context.

    Parameters
    ----------
    data : NDArray
        A batch of data.
    ctx_list : list of Context
        A list of Context
    batch_axis : int, default 0
        The axis along which to slice.
    even_split : bool, default True
        Whether to force all slices to have the same number of elements.

    Returns
    -------
    list of NDArray, each corresponds to a context in ctx_list.
    """
    if len(ctx_list) == 1:
        if not isinstance(data, ndarray.NDArray):
            data = ndarray.array(data, ctx=ctx_list[0])
        return [data.as_in_context(ctx_list[0])]
    else:
        slices = split_data(data, len(ctx_list), batch_axis=batch_axis,
                            even_split=even_split)
        return [i.as_in_context(ctx) for i, ctx in zip(slices, ctx_list)]
