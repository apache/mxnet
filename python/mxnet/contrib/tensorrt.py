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

import ctypes
import logging
import os

from mxnet.symbol import Symbol

from ..base import _LIB, SymbolHandle, MXNetError
from ..base import check_call


def set_use_tensorrt(status):
    """
    Set an environment variable which will enable or disable the use of TensorRT in the backend.
    Note: this is useful for A/B testing purposes.
    :param status: Boolean, true if TensorRT optimization should be applied, False for legacy
    behaviour.
    """
    os.environ["MXNET_USE_TENSORRT"] = str(int(status))


def get_use_tensorrt():
    """
    Get an environment variable which describes if TensorRT is currently enabled in the backend.
    Note: this is useful for A/B testing purposes.
    :return: Boolean, true if TensorRT optimization should be applied, False for legacy
    behaviour.
    """
    return bool(int(os.environ.get("MXNET_USE_TENSORRT", 0)) == 1)


def get_optimized_symbol(executor):
    """
    Take an executor's underlying symbol graph and return its generated optimized version.

    Parameters
    ----------
    executor :
        An executor for which you want to see an optimized symbol. Getting an optimized symbol
        is useful to compare and verify the work TensorRT has done against a legacy behaviour.

    Returns
    -------
    symbol : nnvm::Symbol
        The nnvm symbol optimized.
    """
    handle = SymbolHandle()
    try:
        check_call(_LIB.MXExecutorGetOptimizedSymbol(executor.handle, ctypes.byref(handle)))
        result = Symbol(handle=handle)
        return result
    except MXNetError:
        logging.error('Error while trying to fetch TRT optimized symbol for graph. Please ensure '
                      'build was compiled with MXNET_USE_TENSORRT enabled.')
        raise


def trt_bind(symbol, all_params, ctx, **kwargs):
    """Bind current symbol to get an optimized trt executor.

    Parameters
    ----------
    symbol : Symbol
        The symbol you wish to bind, and optimize with TensorRT.

    all_params : Dict of str->ndarray
        A dictionary of mappings from parameter names to parameter NDArrays.

    ctx : Context
        The device context the generated executor to run on.

    grad_req: string
        {'write', 'add', 'null'}, or list of str or dict of str to str, optional
        To specify how we should update the gradient to the `args_grad`.

        - 'write' means every time gradient is written to specified `args_grad` NDArray.
        - 'add' means every time gradient is added to the specified NDArray.
        - 'null' means no action is taken, the gradient may not be calculated.  This is the only
        mode supported by TensorRT

    type_dict  : Dict of str->numpy.dtype
        Input type dictionary, name->dtype

    stype_dict  : Dict of str->str
        Input storage type dictionary, name->storage_type

    group2ctx : Dict of string to mx.Context
        The dict mapping the `ctx_group` attribute to the context assignment.

    shared_arg_names : List of string
        The argument names whose `NDArray` of shared_exec can be reused for initializing
        the current executor.

    shared_exec : Executor
        The executor whose arg_arrays, arg_arrays, grad_arrays, and aux_arrays can be
        reused for initializing the current executor.

    shared_buffer : Dict of string to `NDArray`
        The dict mapping argument names to the `NDArray` that can be reused for initializing
        the current executor. This buffer will be checked for reuse if one argument name
        of the current executor is not found in `shared_arg_names`. The `NDArray`s are
        expected have default storage type.

    kwargs : Dict of str->shape
        Input shape dictionary, name->shape

    Returns
    -------
    executor : mxnet.Executor
        An optimized TensorRT executor.
    """
    kwargs['shared_buffer'] = all_params
    return symbol.simple_bind(ctx, **kwargs)
