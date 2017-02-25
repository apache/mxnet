# Symbol API

The symbol API provides a way to construct computation graphs.
A tutorial is avaible at [http://mxnet.io/tutorials/python/symbol.html](http://mxnet.io/tutorials/python/symbol.html)


```eval_rst
    .. currentmodule:: mxnet.symbol
```

## Creation routines

```eval_rst
    .. autosummary::
        :toctree: generated/

        Variable
        zeros
        ones
        arange
```

## Mathematical functions

### Arithmetic operations

```eval_rst
    .. autosummary::
        :toctree: generated/

        elemwise_add
        dot
        batch_dot
        broadcast_plus
        broadcast_add
        broadcast_sub
        broadcast_minus
        broadcast_mul
        broadcast_div
        broadcast_power
```

### Trigonometric functions

```eval_rst
    .. autosummary::
        :toctree: generated/

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

### Sums and products

```eval_rst
    .. autosummary::
        :toctree: generated/

        sum
        sum_axis
        ElementWiseSum
        nansum
        prod
        nanprod
```

### Hyperbolic functions

```eval_rst
    .. autosummary::
        :toctree: generated/

        sinh
        cosh
        tanh
        arcsinh
        arccosh
        arctanh
```

### Rounding

```eval_rst
    .. autosummary::
        :toctree: generated/

        round
        rint
        fix
        floor
        ceil
```


### Exponents and logarithms

```eval_rst
    .. autosummary::
        :toctree: generated/

        exp
        expm1
        log
        log10
        log2
        log1p
        sqrt
        rsqrt
        square
```

### Miscellaneous

```eval_rst
    .. autosummary::
        :toctree: generated/

        maximum
        minimum
        broadcast_maximum
        broadcast_minimum
        clip
        abs
        sign
        identity
        gamma
        gammaln
        smooth_l1
```

## manipulation routines

### Changing type

```eval_rst
    .. autosummary::
        :toctree: generated/

        cast
```

### Changing shape

```eval_rst
    .. autosummary::
        :toctree: generated/

        Reshape
        Flatten
```

### Changing both shape and elements

```eval_rst
    .. autosummary::
        :toctree: generated/

        broadcast_to
        broadcast_axis
        expand_dims
        Crop
        crop
        Pad
```


### Joining symbols

```eval_rst
    .. autosummary::
        :toctree: generated/

        concatenate
        Concat
```

### Splitting symbols

```eval_rst
    .. autosummary::
        :toctree: generated/

        slice_axis
        SliceChannel
```

### Tiling symbols

```eval_rst
    .. autosummary::
        :toctree: generated/

        repeat
        tile
```


### Rearranging elements

```eval_rst
    .. autosummary::
        :toctree: generated/

        transpose
        SwapAxis
        flip

```

## Indexing routines

```eval_rst
    .. autosummary::
        :toctree: generated/

        take
        batch_take
        choose_element_0index
        fill_element_0index
        one_hot
        onehot_encode
        SequenceLast
        SequenceMask
        SequenceReverse
```

## Input and output

```eval_rst
    .. autosummary::
        :toctree: generated/

        load
        Symbol.save
```

## Logic functions

```eval_rst
    .. autosummary::
        :toctree: generated/

        broadcast_equal
        broadcast_not_equal
        broadcast_greater
        broadcast_greater_equal
        broadcast_lesser
        broadcast_lesser_equal
```

## Random sampling

```eval_rst
    .. autosummary::
        :toctree: generated/

        uniform
        normal
```

## Sorting and searching

```eval_rst
    .. autosummary::
        :toctree: generated/

        sort
        topk
        argsort
        argmax
        argmax_channel
        argmin
```

## Statistics

```eval_rst
    .. autosummary::
        :toctree: generated/

        mean
        norm
        max
        min
        max_axis
        min_axis
```

## Neural network

### Fully-connection

```eval_rst
    .. autosummary::
        :toctree: generated/

        FullyConnected
```

### Convolution

```eval_rst
    .. autosummary::
        :toctree: generated/

        Convolution
        Correlation
        Deconvolution
```

### Recurrent layers

```eval_rst
    .. autosummary::
        :toctree: generated/

        RNN
```

### Embedding

```eval_rst
    .. autosummary::
        :toctree: generated/

        Embedding
```

### Activation

```eval_rst
    .. autosummary::
        :toctree: generated/

        Activation
        LeakyReLU
        SoftmaxActivation
```

### Normalization

```eval_rst
    .. autosummary::
        :toctree: generated/

        BatchNorm
        InstanceNorm
        L2Normalization
        LRN
```


### Sampling

```eval_rst
    .. autosummary::
        :toctree: generated/

        Pooling
        ROIPooling
        Dropout
        BilinearSampler
        GridGenerator
        UpSampling
        SpatialTransformer
```

### Output

```eval_rst
    .. autosummary::
        :toctree: generated/

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
        :toctree: generated/

        IdentityAttachKLSparseReg
```
### Utilities

```eval_rst
    .. autosummary::
        :toctree: generated/

        MakeLoss
        BlockGrad
        Custom
```

### Weight updating functions

```eval_rst
    .. autosummary::
        :toctree: generated/

        adam_update
        rmsprop_update
        sgd_mom_update
        sgd_update
```

## API Reference

<script type="text/javascript" src='../../_static/js/auto_module_index.js'></script>

```eval_rst
    .. automodule:: mxnet.symbol
        :members:

```

<script>auto_index("api-reference");</script>
