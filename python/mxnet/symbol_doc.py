# coding: utf-8
# pylint: disable=unused-argument, too-many-arguments
"""Extra symbol documents"""
from __future__ import absolute_import as _abs
import re as _re
from .base import build_param_doc as _build_param_doc

class SymbolDoc(object):
    """The basic class"""
    pass


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
    doc_str = ('%s\n\n' +
               '%s\n' +
               'name : string, optional.\n' +
               '    Name of the resulting symbol.\n\n' +
               'Returns\n' +
               '-------\n' +
               'symbol: Symbol\n' +
               '    The result symbol.')
    doc_str = doc_str % (desc, param_str)
    extra_doc = "\n" + '\n'.join([x.__doc__ for x in type.__subclasses__(SymbolDoc)
                                  if x.__name__ == '%sDoc' % func_name])
    doc_str += _re.sub(_re.compile("    "), "", extra_doc)
    return doc_str


class ConcatDoc(SymbolDoc):
    """
    Examples
    --------
    >>> import mxnet as mx
    >>> data = mx.nd.array(range(6)).reshape((2,1,3))
    >>> print "input shape = %s" % data.shape
    >>> print "data = %s" % (data.asnumpy(), )
    input shape = (2L, 1L, 3L)
    data = [[[ 0.  1.  2.]]
     [[ 3.  4.  5.]]]

    >>> # concat two variables on different dimensions
    >>> a = mx.sym.Variable('a')
    >>> b = mx.sym.Variable('b')
    >>> for dim in range(3):
    ...     cat = mx.sym.Concat(a, b, dim=dim)
    ...     exe = cat.bind(ctx=mx.cpu(), args={'a':data, 'b':data})
    ...     exe.forward()
    ...     out = exe.outputs[0]
    ...     print "concat at dim = %d" % dim
    ...     print "shape = %s" % (out.shape, )
    ...     print "results = %s" % (out.asnumpy(), )
    concat at dim = 0
    shape = (4L, 1L, 3L)
    results = [[[ 0.  1.  2.]]
     [[ 3.  4.  5.]]
     [[ 0.  1.  2.]]
     [[ 3.  4.  5.]]]
    concat at dim = 1
    shape = (2L, 2L, 3L)
    results = [[[ 0.  1.  2.]
      [ 0.  1.  2.]]
     [[ 3.  4.  5.]
      [ 3.  4.  5.]]]
    concat at dim = 2
    shape = (2L, 1L, 6L)
    results = [[[ 0.  1.  2.  0.  1.  2.]]
     [[ 3.  4.  5.  3.  4.  5.]]]
    """
    pass

class BroadcastPlusDoc(SymbolDoc):
    """add with broadcast

    Examples
    --------
    >>> a = mx.sym.Variable('a')
    >>> b = mx.sym.Variable('b')
    >>> c = mx.sym.BroadcastPlus(a, b)

    >>> dev = mx.cpu();
    >>> x = c.bind(dev, args={'a': mx.nd.ones((2,2)), 'b' : mx.nd.ones((2,2))})
    >>> x.forward()
    >>> print x.outputs[0].asnumpy()
    [[ 2.  2.]
     [ 2.  2.]]
    >>> x = c.bind(dev, args={'a': mx.nd.ones((2,2)), 'b' : mx.nd.ones((1,1))})
    >>> x.forward()
    >>> print x.outputs[0].asnumpy()
    [[ 2.  2.]
     [ 2.  2.]]
    >>> x = c.bind(dev, args={'a': mx.nd.ones((2,1)), 'b' : mx.nd.ones((1,2))})
    >>> x.forward()
    >>> print x.outputs[0].asnumpy()
    [[ 2.  2.]
     [ 2.  2.]]
    >>> x = c.bind(dev, args={'a': mx.nd.ones((1,2)), 'b' : mx.nd.ones((2,1))})
    >>> x.forward()
    >>> print x.outputs[0].asnumpy()
    [[ 2.  2.]
     [ 2.  2.]]
    >>> x = c.bind(dev, args={'a': mx.nd.ones((2,2,2)), 'b' : mx.nd.ones((1,2,1))}
    >>> x.forward()
    >>> print x.outputs[0].asnumpy()
    [[[ 2.  2.]
      [ 2.  2.]]
     [[ 2.  2.]
      [ 2.  2.]]]
    >>> x = c.bind(dev, args={'a': mx.nd.ones((2,1,1)), 'b' : mx.nd.ones((2,2,2))})
    >>> x.forward()
    >>> print x.outputs[0].asnumpy()
    [[[ 2.  2.]
      [ 2.  2.]]
     [[ 2.  2.]
      [ 2.  2.]]]
    """
