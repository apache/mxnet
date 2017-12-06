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
<br><br>

```eval_rst

.. note:: most operators provided in ``symbol`` are similar to those in ``ndarray``
   although there are few differences:

   - ``symbol`` adopts declarative programming. In other words, we need to first
     compose the computations, and then feed it with data for execution whereas
     ndarray adopts imperative programming.

   - Most binary operators in ``symbol`` such as ``+`` and ``>`` don't broadcast.
     We need to call the broadcast version of the operator such as ``broadcast_plus``
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
    Symbol.__mod__
    Symbol.__rmod__
    Symbol.__pow__
```

#### Trigonometric functions

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.sin
    Symbol.cos
    Symbol.tan
    Symbol.arcsin
    Symbol.arccos
    Symbol.arctan
    Symbol.degrees
    Symbol.radians
```

#### Hyperbolic functions

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.sinh
    Symbol.cosh
    Symbol.tanh
    Symbol.arcsinh
    Symbol.arccosh
    Symbol.arctanh
```

#### Exponents and logarithms

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.exp
    Symbol.expm1
    Symbol.log
    Symbol.log10
    Symbol.log2
    Symbol.log1p
```

#### Powers

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.sqrt
    Symbol.rsqrt
    Symbol.cbrt
    Symbol.rcbrt
    Symbol.square
```

## Basic neural network functions

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.relu
    Symbol.sigmoid
    Symbol.softmax
    Symbol.log_softmax
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

### Symbol creation

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.zeros_like
    Symbol.ones_like
```

### Changing shape and type

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.astype
    Symbol.reshape
    Symbol.reshape_like
    Symbol.flatten
    Symbol.expand_dims
```

### Expanding elements

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.broadcast_to
    Symbol.broadcast_axes
    Symbol.tile
    Symbol.pad
```

### Rearranging elements

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.transpose
    Symbol.swapaxes
    Symbol.flip
```

### Reduce functions

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.sum
    Symbol.nansum
    Symbol.prod
    Symbol.nanprod
    Symbol.mean
    Symbol.max
    Symbol.min
    Symbol.norm
```

### Rounding

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.round
    Symbol.rint
    Symbol.fix
    Symbol.floor
    Symbol.ceil
    Symbol.trunc
```

### Sorting and searching

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.sort
    Symbol.argsort
    Symbol.topk
    Symbol.argmax
    Symbol.argmin
    Symbol.argmax_channel
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

### Indexing

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.slice
    Symbol.slice_axis
    Symbol.take
    Symbol.one_hot
    Symbol.pick
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

### Miscellaneous

```eval_rst
.. autosummary::
    :nosignatures:

    Symbol.clip
    Symbol.sign
```

## Symbol creation routines

```eval_rst
.. autosummary::
    :nosignatures:

    var
    zeros
    zeros_like
    ones
    ones_like
    arange
```

## Symbol manipulation routines

### Changing shape and type

```eval_rst
.. autosummary::
    :nosignatures:

    cast
    reshape
    reshape_like
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
    stack
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
    pick
    where
    gather_nd
    scatter_nd
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
    broadcast_mod
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
    trunc
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
    cbrt
    rcbrt
    square
    reciprocal
```

### Comparison

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

    mxnet.symbol.random.uniform
    mxnet.symbol.random.normal
    mxnet.symbol.random.gamma
    mxnet.symbol.random.exponential
    mxnet.symbol.random.poisson
    mxnet.symbol.random.negative_binomial
    mxnet.symbol.random.generalized_negative_binomial
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

### Sequence operation

```eval_rst
.. autosummary::
    :nosignatures:

    SequenceLast
    SequenceMask
    SequenceReverse
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
    relu
    sigmoid
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

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst

.. autoclass:: mxnet.symbol.Symbol
    :members:
    :special-members:

.. automodule:: mxnet.symbol
    :members:
    :imported-members:
    :special-members:
    :exclude-members: Symbol

.. automodule:: mxnet.symbol.random
    :members:

```

<script>auto_index("api-reference");</script>
