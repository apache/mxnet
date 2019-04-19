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

"""Register backend ops in mxnet.ndarray namespace"""
from __future__ import absolute_import
import os
import sys
import ctypes

from ..base import check_call, _LIB, py_str, OpHandle, c_str
from ..ndarray.register import _make_ndarray_function


_NP_OP_SUBMODULE_LIST = ['_random_', '_linalg_']
_NP_OP_PREFIX = '_numpy_'


def _get_op_submodule_name(op_name):
    assert op_name.startswith(_NP_OP_PREFIX)
    for name in _NP_OP_SUBMODULE_LIST:
        if op_name[len(_NP_OP_PREFIX):].startswith(name):
            return name
    return ""


def _init_np_op_module(root_namespace, module_name, make_op_func):
    """
    Registers op functions created by `make_op_func` under
    `root_namespace.module_name.[submodule_name]`,
    where `submodule_name` is one of `_OP_SUBMODULE_NAME_LIST`.

    Parameters
    ----------
    root_namespace : str
        Top level module name, `mxnet` in the current cases.
    module_name : str
        Second level module name, `numpy` in the current case.
    make_op_func : function
        Function for creating op functions for `mxnet.numpy` module.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXListAllOpNames(ctypes.byref(size), ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        name = py_str(plist[i])
        if name.startswith(_NP_OP_PREFIX):
            op_names.append(name)

    module_op = sys.modules["%s.%s._op" % (root_namespace, module_name)]
    submodule_dict = {}
    for submodule_name in _NP_OP_SUBMODULE_LIST:
        submodule_dict[submodule_name] = \
            sys.modules["%s.%s.%s" % (root_namespace, module_name, submodule_name[1:-1])]
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        submodule_name = _get_op_submodule_name(name)
        module_name_local = module_name
        if len(submodule_name) > 0:
            func_name = name[(len(_NP_OP_PREFIX) + len(submodule_name)):]
            cur_module = submodule_dict[submodule_name]
            module_name_local = "%s.%s.%s" % (root_namespace, module_name, submodule_name[1:-1])
        else:
            func_name = name[len(_NP_OP_PREFIX):]
            cur_module = module_op

        function = make_op_func(hdl, name, func_name)
        function.__module__ = module_name_local
        setattr(cur_module, function.__name__, function)
        cur_module.__all__.append(function.__name__)


# def _generate_op_module_signature(root_namespace, module_name, op_code_gen_func):
#     """
#     Generate op functions created by `op_code_gen_func` and write to the source file
#     of `root_namespace.module_name.[submodule_name]`,
#     where `submodule_name` is one of `_OP_SUBMODULE_NAME_LIST`.
#
#     Parameters
#     ----------
#     root_namespace : str
#         Top level module name, `mxnet` in the current cases.
#     module_name : str
#         Second level module name, `numpy` in the current cases.
#     op_code_gen_func : function
#         Function for creating op functions for `mxnet.numpy` modules.
#     """
#     def get_module_file(module_name):
#         """Return the generated module file based on module name."""
#         path = os.path.dirname(__file__)
#         module_path = module_name.split('.')
#         module_path[-1] = 'gen_' + module_path[-1]
#         file_name = os.path.join(path, '..', *module_path) + '.py'
#         module_file = open(file_name, 'w')
#         dependencies = {'symbol': ['from ._internal import SymbolBase',
#                                    'from ..base import _Null'],
#                         'ndarray': ['from ._internal import NDArrayBase',
#                                     'from ..base import _Null']}
#         module_file.write('# File content is auto-generated. Do not modify.' + os.linesep)
#         module_file.write('# pylint: skip-file' + os.linesep)
#         module_file.write(os.linesep.join(dependencies[module_name.split('.')[1]]))
#         return module_file
#
#     def write_all_str(module_file, module_all_list):
#         """Write the proper __all__ based on available operators."""
#         module_file.write(os.linesep)
#         module_file.write(os.linesep)
#         all_str = '__all__ = [' + ', '.join(["'%s'"%s for s in module_all_list]) + ']'
#         module_file.write(all_str)
#
#     plist = ctypes.POINTER(ctypes.c_char_p)()
#     size = ctypes.c_uint()
#
#     check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
#                                      ctypes.byref(plist)))
#     op_names = []
#     for i in range(size.value):
#         name = py_str(plist[i])
#         if name.startswith(_NP_OP_PREFIX):
#             op_names.append(name)
#
#     module_op_file = get_module_file("%s.%s._op" % (root_namespace, module_name))
#     submodule_dict = {}
#     for submodule_name in _NP_OP_SUBMODULE_LIST:
#         submodule_dict[submodule_name] = \
#             sys.modules["%s.%s.%s" % (root_namespace, module_name, submodule_name[1:-1])]
#     for op_name_prefix in _OP_NAME_PREFIX_LIST:
#         submodule_dict[op_name_prefix] = \
#             (get_module_file("%s.%s.%s" % (root_namespace, module_name,
#                                            op_name_prefix[1:-1])), [])
#     for name in op_names:
#         hdl = OpHandle()
#         check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
#         op_name_prefix = _get_op_name_prefix(name)
#         if len(op_name_prefix) > 0:
#             func_name = name[len(op_name_prefix):]
#             cur_module_file, cur_module_all = submodule_dict[op_name_prefix]
#         elif name.startswith('_'):
#             func_name = name
#             cur_module_file = module_internal_file
#             cur_module_all = module_internal_all
#         else:
#             func_name = name
#             cur_module_file = module_op_file
#             cur_module_all = module_op_all
#
#         code, _ = op_code_gen_func(hdl, name, func_name, True)
#         cur_module_file.write(os.linesep)
#         cur_module_file.write(code)
#         cur_module_all.append(func_name)
#
#     for (submodule_f, submodule_all) in submodule_dict.values():
#         write_all_str(submodule_f, submodule_all)
#         submodule_f.close()
#     write_all_str(module_op_file, module_op_all)
#     module_op_file.close()
#     write_all_str(module_internal_file, module_internal_all)
#     module_internal_file.close()


_init_np_op_module('mxnet', 'numpy', _make_ndarray_function)
