# NDArray API

```eval_rst
.. currentmodule:: mxnet.ndarray
```

This document lists the routines of `mxnet.ndarray` (or `mxnet.nd` for short)
grouped by functionality. Many docstrings contain example code, which
demonstrates the basic usage of the routine. The examples assume that `MXNet` is
imported with:

```python
>>> import mxnet as mx
```

```eval_rst

.. note:: A convenient way to execute examples is the ``%doctest_mode`` mode of
    Jupyter notebook, which allows for pasting of multi-line examples contains
    ``>>>`` and preserves indentation. Run ``%doctest_mode?`` in Jupyter notebook
    for more details.

```

A `NDArray` is a multidimensional container of items of the same type and
size. Various methods for data manipulation and computation are provided.

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

A more detailed tutorial is avaible at
[http://mxnet.io/tutorials/python/ndarray.html](http://mxnet.io/tutorials/python/ndarray.html)

```eval_rst

.. note:: ``NDArray`` is similar to ``numpy.ndarray`` in some aspects.
    happends
    - 111
    - 222

```

```eval_rst

.. note:: ``NDArray`` also provides almost same routines to ``Symbol``.
    adsfasdf sadf asdf

    * 123

```

In the rest of this document, we first overview the methods provided by the
`mxnet.ndarray.NDArray` class, and then list other routines provided by the
`mxnet.ndarray` package.


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
```

### Comparison operators:

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

## Array manipulation routines

### Changing array shape, type and elements

```eval_rst
.. autosummary::
    :nosignatures:

    cast
    reshape -> reshape
    Flatten -> flatten
```

### Changing both array shape and elements

```eval_rst
    .. autosummary::
        :nosignatures:

        broadcast_to
        broadcast_axis
        expand_dims
        Crop -> ?
        crop -> slice
        Pad -> ?
```


### Joining arrays

```eval_rst
    .. autosummary::
        :nosignatures:

        Concat -> hide key_var_num_args
```

### Splitting arrays

```eval_rst
    .. autosummary::
        :nosignatures:

        slice_axis -> special slice
        SliceChannel -> split
```

### Tiling arrays

```eval_rst
    .. autosummary::
        :nosignatures:

        repeat
        tile
```


### Rearranging elements

```eval_rst
    .. autosummary::
        :nosignatures:

        transpose
        SwapAxis ->swap_axis
        flip

```

## Indexing routines

```eval_rst
    .. autosummary::
        :nosignatures:

        take
        batch_take
        one_hot
        SequenceMask -->lower
        SequenceReverse -->lower
```

## Input and output

```eval_rst
    .. autosummary::
        :nosignatures:

        load
        save
```


## Random sampling

```eval_rst
    .. autosummary::
        :nosignatures:

        uniform
        normal
```

## Sorting and searching

```eval_rst
    .. autosummary::
        :nosignatures:

        sort
        topk
        argsort
        argmax
        argmin
```

## Statistics

```eval_rst
    .. autosummary::
        :nosignatures:

        norm
        mean
        max
        min
```

## Neural network

### Fully-connection

```eval_rst
    .. autosummary::
        :nosignatures:

        FullyConnected
```

### Convolution

```eval_rst
    .. autosummary::
        :nosignatures:

        Convolution
        Correlation --> http://dsp.stackexchange.com/questions/12684/difference-between-correlation-and-convolution-on-an-image
        Deconvolution
```

### Recurrent layers

```eval_rst
    .. autosummary::
        :nosignatures:

        RNN
```

### Embedding

```eval_rst
    .. autosummary::
        :nosignatures:

        Embedding
```

### Activation

```eval_rst
    .. autosummary::
        :nosignatures:

        Activation
        LeakyReLU
        SoftmaxActivation --> softmax
```

### Normalization

```eval_rst
    .. autosummary::
        :nosignatures:

        BatchNorm
        InstanceNorm
        L2Normalization
        LRN
```


### Sampling

```eval_rst
    .. autosummary::
        :nosignatures:

        Pooling
        ROIPooling
        Dropout
        BilinearSampler
        GridGenerator
        UpSampling
        SpatialTransformer
```

### Loss

```eval_rst
    .. autosummary::
        :nosignatures:

        SoftmaxOutput
        LinearRegressionOutput
        LogisticRegressionOutput
        MAERegressionOutput
        SVMOutput
        softmax_cross_entropy
```

### Regularization

```eval_rst
    .. autosummary::
        :nosignatures:

        smooth_l1
        IdentityAttachKLSparseReg
```
### Utilities

```eval_rst
    .. autosummary::
        :nosignatures:

        MakeLoss
        BlockGrad
        Custom
```

### Weight updating functions

remove from symbol

```eval_rst
    .. autosummary::
        :nosignatures:

        adam_update
        rmsprop_update
        sgd_mom_update
        sgd_update
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.ndarray
    :members:

```

<script>auto_index("api-reference");</script>
