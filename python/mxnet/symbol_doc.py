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
# pylint: disable=unused-argument, too-many-arguments
"""Extra symbol documents

Guidelines
----------

To add extra doc to the operator `XXX`, write a class `XXXDoc`, deriving
from the base class `SymbolDoc`, and put the extra doc as the docstring
of `XXXDoc`.

The document added here should be Python-specific. Documents that are useful
for all language bindings should be added to the C++ side where the operator
is defined / registered.

The code snippet in the docstring will be run using `doctest`. During running,
the environment will have access to

- all the global names in this file (e.g. `SymbolDoc`)
- all the operators (e.g. `FullyConnected`)
- the name `test_utils` for `mx.test_utils` (e.g. `test_utils.reldiff`)
- the name `mx` (e.g. `mx.nd.zeros`)
- the name `np`

The following documents are recommended:

- *Examples*: simple and short code snippet showing how to use this operator.
  It should show typical calling examples and behaviors (e.g. maps an input
  of what shape to an output of what shape).
"""
from __future__ import absolute_import as _abs
import re as _re
from .base import build_param_doc as _build_param_doc

class SymbolDoc(object):
    """The base class for attaching doc to operators."""

    @staticmethod
    def get_output_shape(sym, **input_shapes):
        """Get user friendly information of the output shapes."""
        _, s_outputs, _ = sym.infer_shape(**input_shapes)
        return dict(zip(sym.list_outputs(), s_outputs))

def _build_doc(func_name,
               desc,
               arg_names,
               arg_types,
               arg_desc,
               key_var_num_args=None,
               ret_type=None):
    """Build docstring for symbolic functions."""
    param_str = _build_param_doc(arg_names, arg_types, arg_desc)
    if key_var_num_args:
        desc += '\nThis function support variable length of positional input.'
    doc_str = (f'{desc}\n\n' +
               f'{param_str}\n' +
               'name : string, optional.\n' +
               '    Name of the resulting symbol.\n\n' +
               'Returns\n' +
               '-------\n' +
               'Symbol\n' +
               '    The result symbol.')
    extra_doc = "\n" + '\n'.join([x.__doc__ for x in type.__subclasses__(SymbolDoc)
                                  if x.__name__ == f'{func_name}Doc'])
    doc_str += _re.sub(_re.compile("    "), "", extra_doc)
    doc_str = _re.sub('NDArray-or-Symbol', 'Symbol', doc_str)
    return doc_str
