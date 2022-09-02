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

# pylint: disable=invalid-name, unused-import
"""
Function namespace.
Acknowledgement: This file originates from incubator-tvm
"""
import os
import sys
import ctypes
from ..base import _LIB, check_call
from .base import py_str, c_str

try:
    if int(os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from ._ctypes.function import FunctionBase as _FunctionBase
        from ._ctypes.function import _set_class_packed_func, _get_global_func
        # To set RETURN_SWITCH for OBJECT_HANDLE
        from . import object
    else:
        from ._cy3.core import FunctionBase as _FunctionBase
        from ._cy3.core import _set_class_packed_func, _get_global_func
except ImportError:
    if int(os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from ._ctypes.function import FunctionBase as _FunctionBase
    from ._ctypes.function import _set_class_packed_func, _get_global_func
    # To set RETURN_SWITCH for OBJECT_HANDLE
    from . import object


class Function(_FunctionBase):
    """The PackedFunc object used in TVM.

    Function plays an key role to bridge front and backend in TVM.
    Function provide a type-erased interface, you can call function with positional arguments.

    The compiled module returns Function.
    TVM backend also registers and exposes its API as Functions.
    For example, the developer function exposed in tvm.ir_pass are actually
    C++ functions that are registered as PackedFunc

    The following are list of common usage scenario of tvm.Function.

    - Automatic exposure of C++ API into python
    - To call PackedFunc from python side
    - To call python callbacks to inspect results in generated code
    - Bring python hook into C++ backend

    See Also
    --------
    tvm.register_func: How to register global function.
    tvm.get_global_func: How to get global function.
    """


def get_global_func(name, allow_missing=False):
    """Get a global function by name

    Parameters
    ----------
    name : str
        The name of the global function

    allow_missing : bool
        Whether allow missing function or raise an error.

    Returns
    -------
    func : tvm.Function
        The function to be returned, None if function is missing.
    """
    return _get_global_func(name, allow_missing)


def list_global_func_names():
    """Get list of global functions registered.

    Returns
    -------
    names : list
       List of global functions names.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXNetFuncListGlobalNames(ctypes.byref(size),
                                             ctypes.byref(plist)))
    fnames = []
    for i in range(size.value):
        fnames.append(py_str(plist[i]))
    return fnames


def _get_api(f):
    flocal = f
    flocal.is_global = True
    return flocal


def _init_api(namespace, target_module_name=None):
    """Initialize api for a given module name

    namespace : str
       The namespace of the source registry

    target_module_name : str
       The target module name if different from namespace
    """
    target_module_name = (
        target_module_name if target_module_name else namespace)
    if namespace.startswith("mxnet."):
        _init_api_prefix(target_module_name, namespace[6:])
    else:
        _init_api_prefix(target_module_name, namespace)


def _init_api_prefix(module_name, prefix):
    module = sys.modules[module_name]

    for name in list_global_func_names():
        if prefix == "api":
            fname = name
            if name.startswith("_"):
                target_module = sys.modules["mxnet._api_internal"]
            else:
                target_module = module
        else:
            if not name.startswith(prefix):
                continue
            fname = name[len(prefix)+1:]
            target_module = module

        if fname.find(".") != -1:
            continue
        f = get_global_func(name)
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = (f"MXNet PackedFunc {fname}. ")
        setattr(target_module, ff.__name__, ff)

_set_class_packed_func(Function)
