# coding: utf-8
"""Extra symbol documents"""

class SymbolDoc(object):
    """The base class for attaching doc to operators."""
    
    @staticmethod
    def get_output_shape(sym, **input_shapes):
        s_inputs, s_outputs, s_aux = sym.infer_shape(**input_shapes)
        return dict(zip(sym.list_outputs(), s_outputs))


class FullyConnectedDoc(SymbolDoc):
    """
    Examples
    --------
    Construct a fully connected operator with target dimension 512.

    >>> data = Variable('data')  # or some constructed NN
    >>> op = FullyConnected(data=data,
    ...                     num_hidden=512,
    ...                     name='FC1')
    >>> op
    <Symbol FC1>
    >>> SymbolDoc.get_output_shape(op, data=(128, 100))
    {'FC1_output': (128L, 512L)}

    A simple 3-layer MLP with ReLU activation:

    >>> net = Variable('data')
    >>> for i, dim in enumerate([128, 64]):
    ...     net = FullyConnected(data=net, num_hidden=dim, name='FC%d' % i)
    ...     net = Activation(data=net, act_type='relu', name='ReLU%d' % i)
    >>> # 10-class predictor (e.g. MNIST)
    >>> net = FullyConnected(data=net, num_hidden=10, name='pred')
    >>> net
    <Symbol pred>
    """
    pass


class ConcatDoc(SymbolDoc):
    """
    Examples
    --------
    Concat two (or more) inputs along a specific dimension:

    >>> a = Variable('a')
    >>> b = Variable('b')
    >>> c = Concat(a, b, dim=1, name='my-concat')
    >>> c
    <Symbol my-concat>
    >>> SymbolDoc.get_output_shape(c, a=(128, 10, 3, 3), b=(128, 15, 3, 3))
    {'my-concat_output': (128L, 25L, 3L, 3L)}

    Note the shape should be the same except on the dimension that is being
    concatenated.
    """
    pass


class BroadcastPlusDoc(SymbolDoc):
    """
    Examples
    --------

    >>> a = Variable('a')
    >>> b = Variable('b')
    >>> c = broadcast_plus(a, b)

    Normal summation with matching shapes:

    >>> dev = mxnet.context.cpu();
    >>> x = c.bind(dev, args={'a': mxnet.nd.ones((2, 2)), 'b' : mxnet.nd.ones((2, 2))})
    >>> x.forward()
    [<NDArray 2x2 @cpu(0)>]
    >>> print x.outputs[0].asnumpy()
    [[ 2.  2.]
     [ 2.  2.]]

    Broadcasting:

    >>> x = c.bind(dev, args={'a': mxnet.nd.ones((2, 2)), 'b' : mxnet.nd.ones((1, 1))})
    >>> x.forward()
    [<NDArray 2x2 @cpu(0)>]
    >>> print x.outputs[0].asnumpy()
    [[ 2.  2.]
     [ 2.  2.]]

    >>> x = c.bind(dev, args={'a': mxnet.nd.ones((2, 1)), 'b' : mxnet.nd.ones((1, 2))})
    >>> x.forward()
    [<NDArray 2x2 @cpu(0)>]
    >>> print x.outputs[0].asnumpy()
    [[ 2.  2.]
     [ 2.  2.]]

    >>> x = c.bind(dev, args={'a': mxnet.nd.ones((1, 2)), 'b' : mxnet.nd.ones((2, 1))})
    >>> x.forward()
    [<NDArray 2x2 @cpu(0)>]
    >>> print x.outputs[0].asnumpy()
    [[ 2.  2.]
     [ 2.  2.]]
    """
