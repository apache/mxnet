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
`ndarray.NDArray` class, and then list other routines provided by the
`ndarray` package.


## The `NDArray` class

### Array attributes

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.shape
    NDArray.size
    NDArray.context
    NDArray.dtype
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
```

### Array change shape

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.T
    NDArray.reshape
    NDArray.broadcast_to
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
```

### Lazy evaluation

```eval_rst
.. autosummary::
    :nosignatures:

    NDArray.wait_to_read
```

## Array creation routines

```eval_rst
.. autosummary::
    :nosignatures:

    array
    empty
    zeros
    ones
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
    square
```

### Logic functions

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

    random_uniform
    random_normal
    random_gamma
    random_exponential
    random_poisson
    random_negative_binomial
    random_generalized_negative_binomial
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
.. automodule:: mxnet.ndarray
    :members:

.. automodule:: mxnet.random
    :members:

```

<script>auto_index("api-reference");</script>
