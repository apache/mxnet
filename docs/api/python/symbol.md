# Symbol API

```eval_rst
    .. currentmodule:: mxnet.symbol
```

## Overview

This document lists the routines of the symbolic expression package:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.symbol
```

The `Symbol` API, defined in the `symbol` (or simply `sym`) package, provides
neural network graphs and auto-differentiation.
A symbol represents a multi-output symbolic expression.
They are composited by operators, such as simple matrix operations (e.g. “+”),
or a neural network layer (e.g. convolution layer).
An operator can take several input variables,
produce more than one output variables, and have internal state variables.
A variable can be either free, which we can bind with value later,
or an output of another symbol.

```python
>>> a = mx.sym.Variable('a')
>>> b = mx.sym.Variable('b')
>>> c = 2 * a + b
>>> type(c)
<class 'mxnet.symbol.Symbol'>
>>> e = c.bind(mx.cpu(), {'a': mx.nd.array([1,2]), 'b':mx.nd.array([2,3])})
>>> y = e.forward()
>>> y
[<NDArray 2 @cpu(0)>]
>>> y[0].asnumpy()
array([ 4.,  7.], dtype=float32)
```

A detailed tutorial is available at [Symbol - Neural network graphs and auto-differentiation](http://mxnet.io/tutorials/basic/symbol.html).

```eval_rst

.. note:: most operators provided in ``symbol`` are similar to ``ndarray``. But
   also note that ``symbol`` differs to ``ndarray`` in several aspects:

   - ``symbol`` adopts declarative programming. In other words, we need to first
     composite the computations, and then feed with data to execute.

   - Most binary operators such as ``+`` and ``>`` are not enabled broadcasting.
     We need to call the broadcasted version such as ``broadcast_plus``
     explicitly.

```

In the rest of this document, we first overview the methods provided by the
`symbol.Symbol` class, and then list other routines provided by the
`symbol` package.

## The `Symbol` class

### Composition

Composite multiple symbols into a new one by an operator.

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.__call__
```

#### Arithmetic operations

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.__add__
    Symbol.__sub__
    Symbol.__rsub__
    Symbol.__neg__
    Symbol.__mul__
    Symbol.__div__
    Symbol.__rdiv__
    Symbol.__pow__
```

#### Comparison operators

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.__lt__
    Symbol.__le__
    Symbol.__gt__
    Symbol.__ge__
    Symbol.__eq__
    Symbol.__ne__
```

### Query information


```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.name
    Symbol.list_arguments
    Symbol.list_outputs
    Symbol.list_auxiliary_states
    Symbol.list_attr
    Symbol.attr
    Symbol.attr_dict
```

### Get internal and output symbol

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.__getitem__
    Symbol.__iter__
    Symbol.get_internals
    Symbol.get_children
```

### Inference type and shape

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.infer_type
    Symbol.infer_shape
    Symbol.infer_shape_partial
```


### Bind

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.bind
    Symbol.simple_bind
```

### Save

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.save
    Symbol.tojson
    Symbol.debug_str
```

## Symbol creation routines

```eval_rst
.. autosummary::
    :nosignatures:

    var
    zeros
    ones
    arange
```

## Symbol manipulation routines

### Changing shape and type

```eval_rst
.. autosummary::
    :nosignatures:

    cast
    reshape
    flatten
    expand_dims
```

### Expanding elements

```eval_rst
.. autosummary::
    :nosignatures:

    broadcast_to
    broadcast_axes
    repeat
    tile
    pad
```

### Rearranging elements

```eval_rst
.. autosummary::
    :nosignatures:

    transpose
    swapaxes
    flip
```

### Joining and splitting symbols

```eval_rst
.. autosummary::
    :nosignatures:

    concat
    split
```

### Indexing routines

```eval_rst
.. autosummary::
    :nosignatures:

    slice
    slice_axis
    take
    batch_take
    one_hot
```

## Mathematical functions

### Arithmetic operations

```eval_rst
.. autosummary::
    :nosignatures:

    broadcast_add
    broadcast_sub
    broadcast_mul
    broadcast_div
    negative
    dot
    batch_dot
    add_n
```

### Trigonometric functions

```eval_rst
.. autosummary::
    :nosignatures:

    sin
    cos
    tan
    arcsin
    arccos
    arctan
    hypot
    broadcast_hypot
    degrees
    radians
```

### Hyperbolic functions

```eval_rst
.. autosummary::
    :nosignatures:

    sinh
    cosh
    tanh
    arcsinh
    arccosh
    arctanh
```

### Reduce functions

```eval_rst
.. autosummary::
    :nosignatures:

    sum
    nansum
    prod
    nanprod
    mean
    max
    min
    norm
```

### Rounding

```eval_rst
.. autosummary::
    :nosignatures:

    round
    rint
    fix
    floor
    ceil
```


### Exponents and logarithms

```eval_rst
.. autosummary::
    :nosignatures:

    exp
    expm1
    log
    log10
    log2
    log1p
```

### Powers

```eval_rst
.. autosummary::
    :nosignatures:

    broadcast_power
    sqrt
    rsqrt
    square
```

### Logic functions

```eval_rst
.. autosummary::
    :nosignatures:

    broadcast_equal
    broadcast_not_equal
    broadcast_greater
    broadcast_greater_equal
    broadcast_lesser
    broadcast_lesser_equal
```
### Random sampling

```eval_rst
.. autosummary::
    :nosignatures:

    random_uniform
    random_normal
    random_gamma
    random_exponential
    random_poisson
    random_negative_binomial
    random_generalized_negative_binomial
    sample_uniform
    sample_normal
    sample_gamma
    sample_exponential
    sample_poisson
    sample_negative_binomial
    sample_generalized_negative_binomial
    mxnet.random.seed
```

### Sorting and searching

```eval_rst
.. autosummary::
    :nosignatures:

    sort
    topk
    argsort
    argmax
    argmin
```

### Miscellaneous

```eval_rst
.. autosummary::
    :nosignatures:

    maximum
    minimum
    broadcast_maximum
    broadcast_minimum
    clip
    abs
    sign
    gamma
    gammaln
```

## Neural network

### Basic

```eval_rst
.. autosummary::
    :nosignatures:

    FullyConnected
    Convolution
    Activation
    BatchNorm
    Pooling
    SoftmaxOutput
    softmax
    log_softmax
```

### More

```eval_rst
.. autosummary::
    :nosignatures:

    Correlation
    Deconvolution
    RNN
    Embedding
    LeakyReLU
    InstanceNorm
    L2Normalization
    LRN
    ROIPooling
    SoftmaxActivation
    Dropout
    BilinearSampler
    GridGenerator
    UpSampling
    SpatialTransformer
    LinearRegressionOutput
    LogisticRegressionOutput
    MAERegressionOutput
    SVMOutput
    softmax_cross_entropy
    smooth_l1
    IdentityAttachKLSparseReg
    MakeLoss
    BlockGrad
    Custom
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.symbol
    :members:

```

<script>auto_index("api-reference");</script>
