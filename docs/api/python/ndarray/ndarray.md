# NDArray API

```eval_rst
.. currentmodule:: mxnet.ndarray
```

## Overview

This document lists the routines of the *n*-dimensional array package:

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.ndarray
```

The `NDArray` API, defined in the `ndarray` (or simply `nd`) package, provides
imperative tensor operations on CPU/GPU.
An `NDArray` represents a multi-dimensional, fixed-size homogenous array.

```python
>>> x = mx.nd.array([[1, 2, 3], [4, 5, 6]])
>>> type(x)
<class 'mxnet.ndarray.NDArray'>
>>> x.shape
(2, 3)
>>> y = x + mx.nd.ones(x.shape)*3
>>> print(y.asnumpy())
[[ 4.  5.  6.]
 [ 7.  8.  9.]]
>>> z = y.as_in_context(mx.gpu(0))
>>> print(z)
<NDArray 2x3 @gpu(0)>
```

A detailed tutorial is available at
[NDArray - Imperative tensor operations on CPU/GPU](http://mxnet.io/tutorials/basic/ndarray.html).
<br><br>

```eval_rst

.. note:: ``mxnet.ndarray`` is similar to ``numpy.ndarray`` in some aspects. But the differences are not negligible. For instance:

   - ``mxnet.ndarray.NDArray.T`` does real data transpose to return new a copied
     array, instead of returning a view of the input array.
   - ``mxnet.ndarray.dot`` performs dot product between the last axis of the
     first input array and the first axis of the second input, while `numpy.dot`
     uses the second last axis of the input array.

   In addition, ``mxnet.ndarray.NDArray`` supports GPU computation and various neural
   network layers.

.. note:: ``ndarray`` provides almost the same routines as ``symbol``. Most
  routines between these two packages share the source code. But ``ndarray``
  differs from ``symbol`` in few aspects:

  - ``ndarray`` adopts imperative programming, namely sentences are executed
    step-by-step so that the results can be obtained immediately whereas
    ``symbol`` adopts declarative programming.

  - Most binary operators in ``ndarray`` such as ``+`` and ``>`` have
    broadcasting enabled by default.
```

In the rest of this document, we first overview the methods provided by the
`ndarray.NDArray` class, and then list other routines provided by the `ndarray` package.

The `ndarray` package provides several classes:

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray
    sparse.CSRNDArray
    sparse.RowSparseNDArray
```

## The `NDArray` class

### Array attributes

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.shape
    NDArray.size
    NDArray.context
    NDArray.dtype
    NDArray.stype
```

### Array conversion

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.copy
    NDArray.copyto
    NDArray.as_in_context
    NDArray.asnumpy
    NDArray.asscalar
    NDArray.astype
    NDArray.tostype
```

### Array creation

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.zeros_like
    NDArray.ones_like
```

### Array change shape

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.T
    NDArray.reshape
    NDArray.reshape_like
    NDArray.flatten
    NDArray.expand_dims
    NDArray.split
```

### Array expand elements

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.broadcast_to
    NDArray.broadcast_axes
    NDArray.tile
    NDArray.pad
```

### Array rearrange elements

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.transpose
    NDArray.swapaxes
    NDArray.flip
```

### Array reduction

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.sum
    NDArray.nansum
    NDArray.prod
    NDArray.nanprod
    NDArray.mean
    NDArray.max
    NDArray.min
    NDArray.norm
```

### Array rounding

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.round
    NDArray.rint
    NDArray.fix
    NDArray.floor
    NDArray.ceil
    NDArray.trunc
```

### Array sorting and searching

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.sort
    NDArray.argsort
    NDArray.topk
    NDArray.argmax
    NDArray.argmin
    NDArray.argmax_channel
```

### Arithmetic operations

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.__add__
    NDArray.__sub__
    NDArray.__rsub__
    NDArray.__neg__
    NDArray.__mul__
    NDArray.__div__
    NDArray.__rdiv__
    NDArray.__mod__
    NDArray.__rmod__
    NDArray.__pow__
```

### Trigonometric functions

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.sin
    NDArray.cos
    NDArray.tan
    NDArray.arcsin
    NDArray.arccos
    NDArray.arctan
    NDArray.degrees
    NDArray.radians
```

### Hyperbolic functions

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.sinh
    NDArray.cosh
    NDArray.tanh
    NDArray.arcsinh
    NDArray.arccosh
    NDArray.arctanh
```

### Exponents and logarithms

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.exp
    NDArray.expm1
    NDArray.log
    NDArray.log10
    NDArray.log2
    NDArray.log1p
```

### Powers

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.sqrt
    NDArray.rsqrt
    NDArray.cbrt
    NDArray.rcbrt
    NDArray.square
    NDArray.reciprocal
```

## Basic neural network functions

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.relu
    NDArray.sigmoid
    NDArray.softmax
    NDArray.log_softmax
```

### In-place arithmetic operations

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.__iadd__
    NDArray.__isub__
    NDArray.__imul__
    NDArray.__idiv__
    NDArray.__imod__
```

### Comparison operators

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.__lt__
    NDArray.__le__
    NDArray.__gt__
    NDArray.__ge__
    NDArray.__eq__
    NDArray.__ne__
```

### Indexing

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.__getitem__
    NDArray.__setitem__
    NDArray.slice
    NDArray.slice_axis
    NDArray.take
    NDArray.one_hot
    NDArray.pick
```

### Lazy evaluation

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.wait_to_read
```

### Miscellaneous

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.clip
    NDArray.sign
```

## Array creation routines

```eval_rst
.. autosummary::
    :nosignatures:

    array
    empty
    zeros
    zeros_like
    ones
    ones_like
    full
    arange
    load
    save
```

## Array manipulation routines

### Changing array shape and type

```eval_rst
.. autosummary::
    :nosignatures:

    cast
    reshape
    reshape_like
    flatten
    expand_dims
```

### Expanding array elements

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

### Joining and splitting arrays

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
```

## Mathematical functions

### Arithmetic operations

```eval_rst
.. autosummary::
    :nosignatures:

    add
    subtract
    negative
    multiply
    divide
    modulo
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

    power
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

    equal
    not_equal
    greater
    greater_equal
    lesser
    lesser_equal
```

### Random sampling

```eval_rst
.. autosummary::
    :nosignatures:

    mxnet.ndarray.random.uniform
    mxnet.ndarray.random.normal
    mxnet.ndarray.random.gamma
    mxnet.ndarray.random.exponential
    mxnet.ndarray.random.poisson
    mxnet.ndarray.random.negative_binomial
    mxnet.ndarray.random.generalized_negative_binomial
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

.. autoclass:: mxnet.ndarray.NDArray
    :members:
    :special-members:

.. automodule:: mxnet.ndarray
    :members:
    :imported-members:
    :special-members:
    :exclude-members: CachedOp, NDArray

.. automodule:: mxnet.random
    :members:

```

<script>auto_index("api-reference");</script>
