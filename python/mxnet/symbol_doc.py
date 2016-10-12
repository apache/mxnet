# coding: utf-8
"""Extra symbol documents"""

import numpy
from .context import cpu, gpu
from .ndarray import array

class SymbolDoc(object):
    """The base class for attaching doc to operators."""
    
    @staticmethod
    def get_output_shape(sym, **input_shapes):
        """Get user friendly information of the output shapes."""
        s_inputs, s_outputs, s_aux = sym.infer_shape(**input_shapes)
        return dict(zip(sym.list_outputs(), s_outputs))

    @staticmethod
    def default_context():
        """Get default context for regression test."""
        # TODO: get context from environment variable to support
        # testing with GPUs
        return cpu()

    @staticmethod
    def default_dtype():
        """Get default data type for regression test."""
        # TODO: get default dtype from environment variable
        return numpy.float32

    @staticmethod
    def default_numerical_threshold():
        """Get default numerical threshold for regression test."""
        # TODO: get from env variable, different threshold might
        # be needed for different device and dtype
        return 1e-6

    @staticmethod
    def random_arrays(*shapes):
        """Generate some random numpy arrays."""
        return [numpy.random.randn(*s).astype(SymbolDoc.default_dtype())
                for s in shapes]

    @staticmethod
    def almost_equal(a, b, threshold=None):
        threshold = threshold or SymbolDoc.default_numerical_threshold()
        return numpy.max(numpy.abs(a-b)) <= threshold

    @staticmethod
    def simple_forward(sym, ctx=None, **inputs):
        ctx = ctx or SymbolDoc.default_context()
        inputs = {k: array(v) for k, v in inputs.iteritems()}
        exe = sym.bind(ctx, args=inputs)
        exe.forward()
        outputs = [x.asnumpy() for x in exe.outputs]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs


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

    Regression Test
    ---------------
    >>> dim_in, dim_out = (3, 4)
    >>> x, w, b = SymbolDoc.random_arrays((10, dim_in), (dim_out, dim_in), (dim_out,))
    >>> op = FullyConnected(num_hidden=dim_out, name='FC')
    >>> out = SymbolDoc.simple_forward(op, FC_data=x, FC_weight=w, FC_bias=b)
    >>> out_np = numpy.dot(x, w.T) + b
    >>> SymbolDoc.almost_equal(out, out_np)
    True
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
