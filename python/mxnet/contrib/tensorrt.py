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

""" Module to enable the use of TensorRT optimized graphs."""
# pylint: skip-file

import ctypes

from ..base import _LIB, c_array, c_array_buf, c_str_array, c_handle_array
from ..base import mx_uint, py_str, string_types
from ..base import NDArrayHandle, ExecutorHandle
from ..base import check_call, MXNetError
from array import array
from ..ndarray import _ndarray_cls
from ..executor import Executor

import numpy as _numpy


def optimize_graph(sym, ctx, grad_req='write', type_dict=None, stype_dict=None,
                   group2ctx=None, shared_arg_names=None, shared_exec=None,
                   shared_buffer=None, **kwargs):
    num_provided_arg_types = 0
    provided_arg_type_names = ctypes.POINTER(ctypes.c_char_p)()  # provided type argument names
    provided_arg_type_data = ctypes.POINTER(mx_uint)()  # provided types
    if type_dict is not None:
        provided_arg_type_names = []
        provided_arg_type_data = []
        for k, v in type_dict.items():
            v = _numpy.dtype(v).type
            if v in _DTYPE_NP_TO_MX:
                provided_arg_type_names.append(k)
                provided_arg_type_data.append(_DTYPE_NP_TO_MX[v])
        num_provided_arg_types = mx_uint(len(provided_arg_type_names))
        provided_arg_type_names = c_str_array(provided_arg_type_names)
        provided_arg_type_data = c_array_buf(ctypes.c_int, array('i', provided_arg_type_data))

    # storage types
    num_provided_arg_stypes = 0
    # provided storage type argument names
    provided_arg_stype_names = ctypes.POINTER(ctypes.c_char_p)()
    provided_arg_stype_data = ctypes.POINTER(mx_uint)()  # provided storage types
    if stype_dict is not None:
        provided_arg_stype_names = []
        provided_arg_stype_data = []
        for k, v in stype_dict.items():
            if v in _STORAGE_TYPE_STR_TO_ID:
                provided_arg_stype_names.append(k)
                provided_arg_stype_data.append(_STORAGE_TYPE_STR_TO_ID[v])
        num_provided_arg_stypes = mx_uint(len(provided_arg_stype_names))
        provided_arg_stype_names = c_str_array(provided_arg_stype_names)
        provided_arg_stype_data = c_array_buf(ctypes.c_int, array('i', provided_arg_stype_data))

    provided_arg_shape_data = []  # shape data
    # argument shape index in sdata,
    # e.g. [sdata[indptr[0]], sdata[indptr[1]]) is the shape of the first arg
    provided_arg_shape_idx = [0]
    provided_arg_shape_names = []  # provided argument names
    for k, v in kwargs.items():
        # if k not in listed_arguments and k not in listed_aux_states:
        #   raise ValueError('arg name %s is not valid', k)
        if isinstance(v, tuple):
            provided_arg_shape_names.append(k)
            provided_arg_shape_data.extend(v)
            provided_arg_shape_idx.append(len(provided_arg_shape_data))

    provided_req_type_list_len = 0
    provided_grad_req_types = ctypes.POINTER(ctypes.c_char_p)()
    provided_grad_req_names = ctypes.POINTER(ctypes.c_char_p)()
    if grad_req is not None:
        if isinstance(grad_req, string_types):
            # use provided_req_type_list_len = 0 to indicate this situation
            provided_req_type_list_len = 0
            provided_grad_req_types = [grad_req]
        elif isinstance(grad_req, list):
            if len(grad_req) == 0:
                raise RuntimeError('grad_req in simple_bind cannot be an empty list')
            provided_grad_req_types = grad_req
            provided_req_type_list_len = len(provided_grad_req_types)
        elif isinstance(grad_req, dict):
            if len(grad_req) == 0:
                raise RuntimeError('grad_req in simple_bind cannot be an empty dict')
            provided_grad_req_names = []
            provided_grad_req_types = []
            for k, v in grad_req.items():
                provided_grad_req_names.append(k)
                provided_grad_req_types.append(v)
            provided_grad_req_names = c_str_array(provided_grad_req_names)
            provided_req_type_list_len = len(provided_grad_req_types)
        provided_grad_req_types = c_str_array(provided_grad_req_types)

    num_ctx_map_keys = mx_uint(0)
    ctx_map_keys = ctypes.POINTER(ctypes.c_char_p)()
    ctx_map_dev_types = ctypes.POINTER(ctypes.c_int)()
    ctx_map_dev_ids = ctypes.POINTER(ctypes.c_int)()
    if group2ctx is not None:
        ctx_map_keys = []
        ctx_map_dev_types = []
        ctx_map_dev_ids = []
        for key, val in group2ctx.items():
            ctx_map_keys.append(key)
            ctx_map_dev_types.append(val.device_typeid)
            ctx_map_dev_ids.append(val.device_id)
        num_ctx_map_keys = mx_uint(len(ctx_map_keys))
        ctx_map_keys = c_str_array(ctx_map_keys)
        ctx_map_dev_types = c_array(ctypes.c_int, array('i', ctx_map_dev_types))
        ctx_map_dev_ids = c_array(ctypes.c_int, array('i', ctx_map_dev_ids))

    # prepare param names
    shared_arg_name_list = []
    if shared_arg_names is not None:
        if not isinstance(shared_arg_names, list):
            raise ValueError('shared_arg_names in simple_bind must be a list or None')
        shared_arg_name_list = shared_arg_names

    # prepare shared_buffer
    if shared_buffer is None:
        shared_buffer_len = ctypes.c_int(-1)
        shared_buffer_names = ctypes.POINTER(ctypes.c_char_p)()
        shared_buffer_handles = ctypes.POINTER(NDArrayHandle)()
    else:
        if not isinstance(shared_buffer, dict):
            raise ValueError('shared_buffer in simple_bind must be dict or None')
        buffer_names = shared_buffer.keys()
        buffer_arrays = shared_buffer.values()
        for v in buffer_arrays:
            assert(v.stype == 'default'), \
                "shared_buffer is expected to only contain NDArrays with default storage"
        shared_buffer_names = c_str_array(buffer_names)
        shared_buffer_len = ctypes.c_int(len(buffer_arrays))
        shared_buffer_handles = c_handle_array(buffer_arrays)
    updated_shared_buffer_names = ctypes.POINTER(ctypes.c_char_p)()
    updated_shared_buffer_handles = ctypes.POINTER(NDArrayHandle)()

    # prepare shared_exec_handle
    shared_exec_handle = shared_exec.handle if shared_exec is not None else ExecutorHandle()

    # prepare current executor handle
    exe_handle = ExecutorHandle()

    # prepare current executor's in_args, arg_grads, and aux_states
    num_in_args = ctypes.c_uint()
    in_arg_handles = ctypes.POINTER(NDArrayHandle)()
    arg_grad_handles = ctypes.POINTER(NDArrayHandle)()
    num_aux_states = ctypes.c_uint()
    aux_state_handles = ctypes.POINTER(NDArrayHandle)()

    try:
        check_call(_LIB.MXExecutorTensorRTBind(sym.handle,
                                             ctypes.c_int(ctx.device_typeid),
                                             ctypes.c_int(ctx.device_id),
                                             num_ctx_map_keys,
                                             ctx_map_keys,
                                             ctx_map_dev_types,
                                             ctx_map_dev_ids,
                                             mx_uint(provided_req_type_list_len),
                                             provided_grad_req_names,
                                             provided_grad_req_types,
                                             mx_uint(len(provided_arg_shape_names)),
                                             c_str_array(provided_arg_shape_names),
                                             c_array_buf(mx_uint,
                                                         array('I', provided_arg_shape_data)),
                                             c_array_buf(mx_uint,
                                                         array('I', provided_arg_shape_idx)),
                                             num_provided_arg_types,
                                             provided_arg_type_names,
                                             provided_arg_type_data,
                                             num_provided_arg_stypes,
                                             provided_arg_stype_names,
                                             provided_arg_stype_data,
                                             mx_uint(len(shared_arg_name_list)),
                                             c_str_array(shared_arg_name_list),
                                             ctypes.byref(shared_buffer_len),
                                             shared_buffer_names,
                                             shared_buffer_handles,
                                             ctypes.byref(updated_shared_buffer_names),
                                             ctypes.byref(updated_shared_buffer_handles),
                                             ctypes.byref(num_in_args),
                                             ctypes.byref(in_arg_handles),
                                             ctypes.byref(arg_grad_handles),
                                             ctypes.byref(num_aux_states),
                                             ctypes.byref(aux_state_handles),
                                             shared_exec_handle,
                                             ctypes.byref(exe_handle)))
    except MXNetError as e:
        error_msg = "simple_bind error. Arguments:\n"
        for k, v in kwargs.items():
            error_msg += "%s: %s\n" % (k, v)
        error_msg += "%s" % e
        raise RuntimeError(error_msg)

    # update shared_buffer
    if shared_buffer is not None:
        for i in range(shared_buffer_len.value):
            k = py_str(updated_shared_buffer_names[i])
            v = NDArray(NDArrayHandle(updated_shared_buffer_handles[i]))
            shared_buffer[k] = v

    # create in_args, arg_grads, and aux_states for the current executor
    arg_arrays = [_ndarray_cls(NDArrayHandle(in_arg_handles[i]))
                  for i in range(num_in_args.value)]
    grad_arrays = [_ndarray_cls(NDArrayHandle(arg_grad_handles[i]))
                   if arg_grad_handles[i] is not None
                   else None for i in range(num_in_args.value)]
    aux_arrays = [_ndarray_cls(NDArrayHandle(aux_state_handles[i]))
                  for i in range(num_aux_states.value)]

    executor = Executor(exe_handle, sym, ctx, grad_req, group2ctx)
    executor.arg_arrays = arg_arrays
    executor.grad_arrays = grad_arrays
    executor.aux_arrays = aux_arrays
    return executor