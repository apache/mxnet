# coding: utf-8
"""Extra symbol documents"""

class SymbolDoc(object):
    """The basic class"""
    pass

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
