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
- the name `test_utils` for `mxnet.test_utils` (e.g. `test_utils.reldiff`)
- the name `mxnet` (e.g. `mxnet.nd.zeros`)
- the name `numpy`

The following documents are recommended:

- *Examples*: simple and short code snippet showing how to use this operator.
  It should show typical calling examples and behaviors (e.g. maps an input
  of what shape to an output of what shape).
- *Regression Test*: longer test code for the operators. We normally do not
  expect the users to read those, but they will be executed by `doctest` to
  ensure the behavior of each operator does not change unintentionally.
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


class ActivationDoc(SymbolDoc):
    """
    Examples
    --------
    A one-hidden-layer MLP with ReLU activation:

    >>> data = Variable('data')
    >>> mlp = FullyConnected(data=data, num_hidden=128, name='proj')
    >>> mlp = Activation(data=mlp, act_type='relu', name='activation')
    >>> mlp = FullyConnected(data=mlp, num_hidden=10, name='mlp')
    >>> mlp
    <Symbol mlp>

    Regression Test
    ---------------
    ReLU activation

    >>> test_suites = [
    ...     ('relu', lambda x: numpy.maximum(x, 0)),
    ...     ('sigmoid', lambda x: 1 / (1 + numpy.exp(-x))),
    ...     ('tanh', lambda x: numpy.tanh(x)),
    ...     ('softrelu', lambda x: numpy.log(1 + numpy.exp(x)))
    ... ]
    >>> x = test_utils.random_arrays((2, 3, 4))
    >>> for act_type, numpy_impl in test_suites:
    ...     op = Activation(act_type=act_type, name='act')
    ...     y = test_utils.simple_forward(op, act_data=x)
    ...     y_np = numpy_impl(x)
    ...     print('%s: %s' % (act_type, test_utils.almost_equal(y, y_np)))
    relu: True
    sigmoid: True
    tanh: True
    softrelu: True
    """


class DropoutDoc(SymbolDoc):
    """
    Examples
    --------
    Apply dropout to corrupt input as zero with probability 0.2:

    >>> data = Variable('data')
    >>> data_dp = Dropout(data=data, p=0.2)

    Regression Test
    ---------------
    >>> shape = (100, 100)  # take larger shapes to be more statistical stable
    >>> x = numpy.ones(shape)
    >>> op = Dropout(p=0.5, name='dp')
    >>> # dropout is identity during testing
    >>> y = test_utils.simple_forward(op, dp_data=x, is_train=False)
    >>> test_utils.almost_equal(x, y, threshold=0)
    True
    >>> y = test_utils.simple_forward(op, dp_data=x, is_train=True)
    >>> # expectation is (approximately) unchanged
    >>> numpy.abs(x.mean() - y.mean()) < 0.1
    True
    >>> set(numpy.unique(y)) == set([0, 2])
    True
    """


class EmbeddingDoc(SymbolDoc):
    """
    Examples
    --------
    Assume we want to map the 26 English alphabet letters to 16-dimensional
    vectorial representations.

    >>> vocabulary_size = 26
    >>> embed_dim = 16
    >>> seq_len, batch_size = (10, 64)
    >>> input = Variable('letters')
    >>> op = Embedding(data=input, input_dim=vocabulary_size, output_dim=embed_dim,
    ...                name='embed')
    >>> SymbolDoc.get_output_shape(op, letters=(seq_len, batch_size))
    {'embed_output': (10L, 64L, 16L)}

    Regression Test
    ---------------
    >>> vocab_size, embed_dim = (26, 16)
    >>> batch_size = 12
    >>> word_vecs = test_utils.random_arrays((vocab_size, embed_dim))
    >>> op = Embedding(name='embed', input_dim=vocab_size, output_dim=embed_dim)
    >>> x = numpy.random.choice(vocab_size, batch_size)
    >>> y = test_utils.simple_forward(op, embed_data=x, embed_weight=word_vecs)
    >>> y_np = word_vecs[x]
    >>> test_utils.almost_equal(y, y_np)
    True
    """


class FlattenDoc(SymbolDoc):
    """
    Examples
    --------
    Flatten is usually applied before `FullyConnected`, to reshape the 4D tensor
    produced by convolutional layers to 2D matrix:

    >>> data = Variable('data')  # say this is 4D from some conv/pool
    >>> flatten = Flatten(data=data, name='flat')  # now this is 2D
    >>> SymbolDoc.get_output_shape(flatten, data=(2, 3, 4, 5))
    {'flat_output': (2L, 60L)}

    Regression Test
    ---------------
    >>> test_dims = [(2, 3, 4, 5), (2, 3), (2,)]
    >>> op = Flatten(name='flat')
    >>> for dims in test_dims:
    ...     x = test_utils.random_arrays(dims)
    ...     y = test_utils.simple_forward(op, flat_data=x)
    ...     y_np = x.reshape((dims[0], numpy.prod(dims[1:])))
    ...     print('%s: %s' % (dims, test_utils.almost_equal(y, y_np)))
    (2, 3, 4, 5): True
    (2, 3): True
    (2,): True
    """


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
