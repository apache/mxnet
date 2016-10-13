# coding: utf-8
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
- the name `test_utils` for `mxnet.test_utils` (e.g. `test_utils.reldiff`)
- the name `mxnet` (e.g. `mxnet.nd.zeros`)

The following documents are recommended:

- *Examples*: simple and short code snippet showing how to use this operator.
  It should show typical calling examples and behaviors (e.g. maps an input
  of what shape to an output of what shape).
- *Regression Test*: longer test code for the operators. We normally do not
  expect the users to read those, but they will be executed by `doctest` to
  ensure the behavior of each operator does not change unintentionally.
"""

class SymbolDoc(object):
    """The base class for attaching doc to operators."""

    @staticmethod
    def get_output_shape(sym, **input_shapes):
        """Get user friendly information of the output shapes."""
        _, s_outputs, _ = sym.infer_shape(**input_shapes)
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

    Regression Test
    ---------------
    >>> dim_in, dim_out = (3, 4)
    >>> x, w, b = test_utils.random_arrays((10, dim_in), (dim_out, dim_in), (dim_out,))
    >>> op = FullyConnected(num_hidden=dim_out, name='FC')
    >>> out = test_utils.simple_forward(op, FC_data=x, FC_weight=w, FC_bias=b)
    >>> # numpy implementation of FullyConnected
    >>> out_np = numpy.dot(x, w.T) + b
    >>> test_utils.almost_equal(out, out_np)
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
