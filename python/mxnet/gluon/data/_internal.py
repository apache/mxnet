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
"""C++ Datasets for common data formats."""
import sys
import ctypes

from .dataset import Dataset
from .sampler import Sampler
from ...base import _LIB
from ...base import c_str_array, mx_uint, py_str
from ...base import DatasetHandle, NDArrayHandle, BatchifyFunctionhandle
from ...base import check_call, build_param_doc as _build_param_doc
from ...ndarray import NDArray
from ...ndarray import _ndarray_cls
from ...numpy.multiarray import _np_ndarray_cls
from ...util import is_np_array, default_array
from ...io import io as _io


class MXDataset(Dataset):
    """A python wrapper a C++ dataset.

    Parameters
    ----------
    handle : DatasetHandle, required
        The handle to the underlying C++ Dataset.

    """
    def __init__(self, handle, **kwargs):
        super(MXDataset, self).__init__()
        self.handle = handle
        self._kwargs = kwargs
        # get dataset size
        length = ctypes.c_uint64(0)
        check_call(_LIB.MXDatasetGetLen(self.handle, ctypes.byref(length)))
        self._len = length.value

    def __del__(self):
        check_call(_LIB.MXDatasetFree(self.handle))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        orig_idx = idx
        if idx < 0:
            idx += self._len
        # check bound
        if idx < 0 or idx >= self._len:
            raise IndexError("Index {} out of bound: (0, {})".format(orig_idx, self._len))
        create_ndarray_fn = _np_ndarray_cls if is_np_array() else _ndarray_cls
        output_vars = ctypes.POINTER(NDArrayHandle)()
        num_output = ctypes.c_int(0)
        check_call(_LIB.MXDatasetGetItems(self.handle,
                                          ctypes.c_uint64(idx),
                                          ctypes.byref(num_output),
                                          ctypes.byref(output_vars)))
        out = [create_ndarray_fn(ctypes.cast(output_vars[i], NDArrayHandle),
                                 False) for i in range(num_output.value)]
        for i in range(num_output.value):
            if out[i].size == 1:
                out[i] = out[i].asnumpy()
        if len(out) > 1:
            return tuple(out)
        return out[0]


class MXSampler(Sampler):
    """MXNet internal sampler implemented in c++.

    Parameters
    ----------
    name : str
        Name of the sampler.

    """
    def __init__(self, name, **kwargs):
        try:
            creator = getattr(_io, name)
        except AttributeError:
            raise ValueError('{} is not a valid MXDataIter class'.format(name))
        self._iter = creator(**kwargs)

    def __len__(self):
        try:
            size = len(self._iter)
        except TypeError:
            raise TypeError('Iterator {} does not provide length info'.format(self._iter))
        return size

    def __iter__(self):
        for item in self._iter:
            ret = item.data[0].asnumpy().flatten().tolist()
            pad = item.pad
            if pad > 0:
                # remove padded values
                ret = ret[:-pad]
            elif len(ret) == 1:
                ret = ret[0]
            yield ret
        self._iter.reset()


class MXBatchifyFunction(object):
    """MXNet batchify function implemented in C++.

    Parameters
    ----------
    handle : ctypes.c_void
        Object handle.

    """
    def __init__(self, handle, **kwargs):
        self._kwargs = kwargs
        self.handle = handle

    def __del__(self):
        if self.handle is not None:
            check_call(_LIB.MXBatchifyFunctionFree(self.handle))

    def __getstate__(self):
        """Override pickling behavior."""
        # pickling pointer is not allowed
        d = dict({'creator_name': self._kwargs['creator_name'],
                  '_kwargs': self._kwargs})
        return d

    def __setstate__(self, d):
        """Restore from pickled."""
        creator = d['_kwargs']['creator_name']
        d['_kwargs'].pop('creator_name')
        other = getattr(sys.modules[__name__], creator)(**d['_kwargs'])
        self.handle = other.handle
        self._kwargs = other._kwargs
        other.handle = None

    def __call__(self, data, num_out=1):
        if isinstance(data[0], NDArray):
            create_ndarray_fn = _np_ndarray_cls if is_np_array() else _ndarray_cls
            num_output = ctypes.c_int(num_out)
            input_arrs = (NDArrayHandle * len(data))()
            for i, d in enumerate(data):
                input_arrs[i] = d.handle
            input_vars = ctypes.cast(input_arrs, ctypes.POINTER(NDArrayHandle))
            batch_size = ctypes.c_int(len(data) // num_output.value)
            output_vars = ctypes.POINTER(NDArrayHandle)()
            check_call(_LIB.MXBatchifyFunctionInvoke(self.handle,
                                                     batch_size,
                                                     num_output,
                                                     input_vars,
                                                     ctypes.byref(output_vars)))
            out = [create_ndarray_fn(ctypes.cast(output_vars[i], NDArrayHandle), \
                False) for i in range(num_output.value)]
            if len(out) == 1:
                out = out[0]
            return out
        elif isinstance(data[0], (list, tuple)):
            return self.__call__([j for sub in data for j in sub], num_out=len(data[0]))
        else:
            data = [default_array(i) for i in data]
            return self.__call__(data, num_out=num_out)

def _make_internal_datasets(handle):
    """Create an io iterator by handle."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXDatasetGetDatasetInfo( \
            handle, ctypes.byref(name), ctypes.byref(desc), \
            ctypes.byref(num_args), \
            ctypes.byref(arg_names), \
            ctypes.byref(arg_types), \
            ctypes.byref(arg_descs)))
    iter_name = py_str(name.value)

    narg = int(num_args.value)
    param_str = _build_param_doc(
        [py_str(arg_names[i]) for i in range(narg)],
        [py_str(arg_types[i]) for i in range(narg)],
        [py_str(arg_descs[i]) for i in range(narg)])

    doc_str = (f'{desc.value}\n\n' +
               f'{param_str}\n' +
               'Returns\n' +
               '-------\n' +
               'MXDataset\n'+
               '    The result dataset.')

    def creator(*args, **kwargs):
        """Create a dataset.
        The parameters listed below can be passed in as keyword arguments.

        Parameters
        ----------
        name : string, required.
            Name of the resulting dataset.

        Returns
        -------
        dataset: Dataset
            The resulting dataset.
        """
        param_keys = []
        param_vals = []

        for k, val in kwargs.items():
            # convert ndarray to handle
            if hasattr(val, 'handle'):
                val = val.handle.value
            if isinstance(val, (tuple, list)):
                val = [vv.handle.value if hasattr(vv, 'handle') else vv for vv in val]
            param_keys.append(k)
            param_vals.append(str(val))
        # create atomic symbol
        param_keys = c_str_array(param_keys)
        param_vals = c_str_array(param_vals)
        dataset_handle = DatasetHandle()
        check_call(_LIB.MXDatasetCreateDataset(
            handle,
            mx_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(dataset_handle)))

        if len(args):
            raise TypeError(f'{iter_name} can only accept keyword arguments')

        return MXDataset(dataset_handle, **kwargs)

    creator.__name__ = iter_name
    creator.__doc__ = doc_str
    return creator

def _init_internal_dataset_module():
    """List and add all the datasets to current module."""
    plist = ctypes.POINTER(ctypes.c_void_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListDatasets(ctypes.byref(size), ctypes.byref(plist)))
    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = ctypes.c_void_p(plist[i])
        dataset = _make_internal_datasets(hdl)
        setattr(module_obj, dataset.__name__, dataset)

_init_internal_dataset_module()

def _make_internal_batchify_functions(handle):
    """Create an io iterator by handle."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXBatchifyFunctionGetFunctionInfo( \
            handle, ctypes.byref(name), ctypes.byref(desc), \
            ctypes.byref(num_args), \
            ctypes.byref(arg_names), \
            ctypes.byref(arg_types), \
            ctypes.byref(arg_descs)))
    bf_name = py_str(name.value)

    narg = int(num_args.value)
    param_str = _build_param_doc(
        [py_str(arg_names[i]) for i in range(narg)],
        [py_str(arg_types[i]) for i in range(narg)],
        [py_str(arg_descs[i]) for i in range(narg)])

    doc_str = (f'{desc.value}\n\n' +
               f'{param_str}\n' +
               'Returns\n' +
               '-------\n' +
               'MXBatchifyFunction\n'+
               '    The result batchify function.')

    def creator(*args, **kwargs):
        """Create an iterator.
        The parameters listed below can be passed in as keyword arguments.

        Parameters
        ----------
        name : string, required.
            Name of the resulting batchify function.

        Returns
        -------
        batchify_func: BatchifyFunction
            The resulting batchify function.
        """
        param_keys = []
        param_vals = []

        for k, val in kwargs.items():
            # convert ndarray to handle
            if hasattr(val, 'handle'):
                val = val.handle.value
            if isinstance(val, (tuple, list)):
                val = [vv.handle.value if hasattr(vv, 'handle') else vv for vv in val]
            param_keys.append(k)
            param_vals.append(str(val))
        # create atomic symbol
        param_keys = c_str_array(param_keys)
        param_vals = c_str_array(param_vals)
        batchify_fn_handle = BatchifyFunctionhandle()
        check_call(_LIB.MXBatchifyFunctionCreateFunction(
            handle,
            mx_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(batchify_fn_handle)))

        if len(args):
            raise TypeError(f'{bf_name} can only accept keyword arguments')

        return MXBatchifyFunction(batchify_fn_handle, creator_name=bf_name, **kwargs)

    creator.__name__ = bf_name
    creator.__doc__ = doc_str
    return creator

def _init_internal_batchify_function_module():
    """List and add all the batchify_functions to current module."""
    plist = ctypes.POINTER(ctypes.c_void_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListBatchifyFunctions(ctypes.byref(size), ctypes.byref(plist)))
    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = ctypes.c_void_p(plist[i])
        bf = _make_internal_batchify_functions(hdl)
        setattr(module_obj, bf.__name__, bf)

_init_internal_batchify_function_module()
