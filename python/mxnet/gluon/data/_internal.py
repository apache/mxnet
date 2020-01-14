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
# pylint: disable=unnecessary-pass
"""C++ Datasets for common data formats."""
from __future__ import absolute_import


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

    def __del__(self):
        check_call(_LIB.MXDatasetFree(self.handle))

    def __len__(self):
        len = ctypes.c_uint64(0)
        check_call(_LIB.MXDatasetGetLen(self.handle, ctypes.byref(len)))
        return len

    def __getitem__(self, idx):
        out_size = ctypes.c_int(0)
        check_call(_LIB.MXDatasetGetOutSize(self.handle, ctypes.byref(out_size)))
        assert out_size > 0, "Invalid number of outputs: {}".format(out_size)
        items = []
        for i in range(out_size):
            hdl = NDArrayHandle()
            check_call(_LIB.MXDatasetGetOutSize(self.handle,
                                                ctypes.c_uint64(idx),
                                                ctypes.c_int(i),
                                                ctypes.byref(hdl))))
            items.append(_ndarray_cls(hdl, False))
        if len(items) == 1:
            return items[0]
        return tuple(items)


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

    doc_str = ('%s\n\n' +
               '%s\n' +
               'Returns\n' +
               '-------\n' +
               'MXDataset\n'+
               '    The result dataset.')
    doc_str = doc_str % (desc.value, param_str)

    def creator(*args, **kwargs):
        """Create an iterator.
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
            raise TypeError('%s can only accept keyword arguments' % dataset_name)

        return MXDataIter(dataset_handle, **kwargs)

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
        dataset = _make_io_iterator(hdl)
        setattr(module_obj, dataiter.__name__, dataiter)

_init_internal_dataset_module()
