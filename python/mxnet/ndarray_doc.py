# coding: utf-8
# pylint: disable=unused-argument, too-many-arguments
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
    if key_var_num_args:
        desc += '\nThis function support variable length of positional input.'
    doc_str = ('%s\n\n' +
               '%s\n' +
               'out : NDArray, optional\n' +
               '    The output NDArray to hold the result.\n\n'+
               'Returns\n' +
               '-------\n' +
               'out : NDArray or list of NDArray\n' +
               '    The output of this function.')
    doc_str = doc_str % (desc, param_str)
    extra_doc = "\n" + '\n'.join([x.__doc__ for x in type.__subclasses__(NDArrayDoc)
                                  if x.__name__ == '%sDoc' % func_name])
    doc_str += _re.sub(_re.compile("    "), "", extra_doc)
    return doc_str
