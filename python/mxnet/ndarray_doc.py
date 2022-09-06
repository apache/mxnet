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
    doc_str = (f'{desc}\n\n' +
               f'{param_str}\n' +
               'out : NDArray, optional\n' +
               '    The output NDArray to hold the result.\n\n'+
               'Returns\n' +
               '-------\n' +
               'out : NDArray or list of NDArrays\n' +
               '    The output of this function.')
    extra_doc = "\n" + '\n'.join([x.__doc__ for x in type.__subclasses__(NDArrayDoc)
                                  if x.__name__ == f'{func_name}Doc'])
    doc_str += _re.sub(_re.compile("    "), "", extra_doc)
    doc_str = _re.sub('NDArray-or-Symbol', 'NDArray', doc_str)

    return doc_str
